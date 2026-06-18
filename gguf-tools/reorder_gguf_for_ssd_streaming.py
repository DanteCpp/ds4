#!/usr/bin/env python3
"""Rewrite a GGUF so routed expert tensors are laid out per layer.

This is an offline layout optimizer for DS4 SSD streaming experiments.  It
does not dequantize, requantize, or rename tensors.  It rebuilds the GGUF
tensor directory and copies tensor payloads in a new order so each layer's
routed expert tensors are physically adjacent:

    blk.N.ffn_gate_exps.weight
    blk.N.ffn_up_exps.weight
    blk.N.ffn_down_exps.weight

Current DS4 expects each routed tensor to remain one contiguous tensor, so this
script intentionally does not interleave individual experts inside those
tensors.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


DEFAULT_INPUT = Path(
    "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)

GGUF_HEADER_SIZE = 24
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

GGUF_SCALAR_SIZES = {
    GGUF_VALUE_UINT8: 1,
    GGUF_VALUE_INT8: 1,
    GGUF_VALUE_UINT16: 2,
    GGUF_VALUE_INT16: 2,
    GGUF_VALUE_UINT32: 4,
    GGUF_VALUE_INT32: 4,
    GGUF_VALUE_FLOAT32: 4,
    GGUF_VALUE_BOOL: 1,
    GGUF_VALUE_UINT64: 8,
    GGUF_VALUE_INT64: 8,
    GGUF_VALUE_FLOAT64: 8,
}

# GGML quant type -> (block elements, bytes per block, display name).
# This mirrors the DS4 GGUF formats used by the local quantizer.
GGML_QUANT_SIZES = {
    0: (1, 4, "F32"),
    1: (1, 2, "F16"),
    8: (32, 34, "Q8_0"),
    10: (256, 84, "Q2_K"),
    12: (256, 144, "Q4_K"),
    16: (256, 66, "IQ2_XXS"),
    26: (1, 4, "I32"),
}

EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
EXPERT_PART_ORDER = ("gate", "up", "down")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    rel_offset: int
    data_offset: int
    n_bytes: int


@dataclass(frozen=True)
class GGUFInfo:
    path: Path
    version: int
    tensor_count: int
    kv_count: int
    kv_blob: bytes
    alignment: int
    tensors: list[TensorInfo]
    tensor_by_name: dict[str, TensorInfo]


@dataclass(frozen=True)
class PlanItem:
    tensor: TensorInfo
    new_rel_offset: int


def pad_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def read_exact(src: BinaryIO, n_bytes: int) -> bytes:
    data = src.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError("short read while parsing GGUF")
    return data


def read_u32_file(src: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(src, 4))[0]


def read_u64_file(src: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(src, 8))[0]


def skip_gguf_string_file(src: BinaryIO) -> None:
    n = read_u64_file(src)
    src.seek(n, os.SEEK_CUR)


def read_gguf_string_file(src: BinaryIO) -> str:
    n = read_u64_file(src)
    return read_exact(src, n).decode("utf-8")


def skip_value_payload_file(src: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_VALUE_STRING:
        skip_gguf_string_file(src)
        return
    if value_type == GGUF_VALUE_ARRAY:
        subtype = read_u32_file(src)
        count = read_u64_file(src)
        if subtype == GGUF_VALUE_STRING:
            for _ in range(count):
                skip_gguf_string_file(src)
            return
        if subtype == GGUF_VALUE_ARRAY:
            for _ in range(count):
                skip_value_payload_file(src, subtype)
            return
        size = GGUF_SCALAR_SIZES.get(subtype)
        if size is None:
            raise ValueError(f"unsupported GGUF array subtype {subtype}")
        src.seek(count * size, os.SEEK_CUR)
        return
    size = GGUF_SCALAR_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    src.seek(size, os.SEEK_CUR)


def tensor_nbytes(dims: tuple[int, ...], ggml_type: int) -> int:
    q = GGML_QUANT_SIZES.get(ggml_type)
    if q is None:
        raise ValueError(f"unsupported GGML tensor type {ggml_type}; add it to GGML_QUANT_SIZES")
    block_elems, block_bytes, _name = q
    n_elems = 1
    for dim in dims:
        n_elems *= dim
    if n_elems % block_elems != 0:
        raise ValueError(f"tensor element count {n_elems} is not divisible by block size {block_elems}")
    return (n_elems // block_elems) * block_bytes


def parse_gguf(path: Path) -> GGUFInfo:
    with path.open("rb") as src:
        if read_exact(src, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32_file(src)
        tensor_count = read_u64_file(src)
        kv_count = read_u64_file(src)

        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(kv_count):
            key = read_gguf_string_file(src)
            value_type = read_u32_file(src)
            value_start = src.tell()
            skip_value_payload_file(src, value_type)
            value_end = src.tell()
            if key == "general.alignment" and value_type == GGUF_VALUE_UINT32:
                src.seek(value_start)
                alignment = read_u32_file(src)
                src.seek(value_end)

        kv_end = src.tell()
        src.seek(GGUF_HEADER_SIZE)
        kv_blob = read_exact(src, kv_end - GGUF_HEADER_SIZE)
        src.seek(kv_end)

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = read_gguf_string_file(src)
            n_dims = read_u32_file(src)
            dims = tuple(read_u64_file(src) for _ in range(n_dims))
            ggml_type = read_u32_file(src)
            rel_offset = read_u64_file(src)
            raw_tensors.append((name, dims, ggml_type, rel_offset))

        data_start = pad_to(src.tell(), alignment)

    tensors: list[TensorInfo] = []
    for name, dims, ggml_type, rel_offset in raw_tensors:
        n_bytes = tensor_nbytes(dims, ggml_type)
        tensors.append(TensorInfo(name, dims, ggml_type, rel_offset, data_start + rel_offset, n_bytes))

    return GGUFInfo(
        path=path,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        kv_blob=kv_blob,
        alignment=alignment,
        tensors=tensors,
        tensor_by_name={t.name: t for t in tensors},
    )


def expert_match(name: str) -> tuple[int, str] | None:
    match = EXPERT_TENSOR_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def collect_expert_layers(info: GGUFInfo) -> dict[int, dict[str, TensorInfo]]:
    layers: dict[int, dict[str, TensorInfo]] = {}
    for tensor in info.tensors:
        matched = expert_match(tensor.name)
        if not matched:
            continue
        layer, part = matched
        layers.setdefault(layer, {})[part] = tensor

    for layer, parts in sorted(layers.items()):
        missing = [part for part in EXPERT_PART_ORDER if part not in parts]
        if missing:
            raise ValueError(f"layer {layer} is missing routed expert tensor(s): {', '.join(missing)}")
    return layers


def build_layer_triplet_order(info: GGUFInfo) -> list[TensorInfo]:
    layers = collect_expert_layers(info)
    emitted_layers: set[int] = set()
    ordered: list[TensorInfo] = []
    emitted_names: set[str] = set()

    for tensor in info.tensors:
        matched = expert_match(tensor.name)
        if not matched:
            ordered.append(tensor)
            emitted_names.add(tensor.name)
            continue

        layer, _part = matched
        if layer in emitted_layers:
            continue
        emitted_layers.add(layer)
        for part in EXPERT_PART_ORDER:
            expert_tensor = layers[layer][part]
            ordered.append(expert_tensor)
            emitted_names.add(expert_tensor.name)

    if len(ordered) != len(info.tensors):
        raise ValueError(f"internal ordering error: planned {len(ordered)} tensors, expected {len(info.tensors)}")
    if len(emitted_names) != len(info.tensors):
        raise ValueError("internal ordering error: duplicate tensor name in output plan")
    return ordered


def build_plan(info: GGUFInfo) -> list[PlanItem]:
    next_rel = 0
    plan: list[PlanItem] = []
    for tensor in build_layer_triplet_order(info):
        plan.append(PlanItem(tensor=tensor, new_rel_offset=next_rel))
        next_rel += pad_to(tensor.n_bytes, info.alignment)
    return plan


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_tensor_info(out: BinaryIO, item: PlanItem) -> None:
    tensor = item.tensor
    out.write(pack_string(tensor.name))
    out.write(struct.pack("<I", len(tensor.dims)))
    for dim in tensor.dims:
        out.write(struct.pack("<Q", dim))
    out.write(struct.pack("<I", tensor.ggml_type))
    out.write(struct.pack("<Q", item.new_rel_offset))


def copy_exact(src: BinaryIO, dst: BinaryIO, n_bytes: int, bufsize: int = 64 * 1024 * 1024) -> None:
    remaining = n_bytes
    while remaining:
        chunk = src.read(min(bufsize, remaining))
        if not chunk:
            raise EOFError("short read while copying tensor payload")
        dst.write(chunk)
        remaining -= len(chunk)


def write_padding(out: BinaryIO, n_bytes: int, alignment: int) -> None:
    pad = pad_to(n_bytes, alignment) - n_bytes
    if pad:
        out.write(b"\0" * pad)


def output_path_for(input_path: Path) -> Path:
    if input_path.suffix == ".gguf":
        return input_path.with_name(input_path.stem + "-ssd-layer-layout.gguf")
    return input_path.with_name(input_path.name + "-ssd-layer-layout.gguf")


def validate_layer_adjacency(plan: list[PlanItem], alignment: int) -> tuple[int, int]:
    by_name = {item.tensor.name: item for item in plan}
    layers = sorted({expert_match(item.tensor.name)[0] for item in plan if expert_match(item.tensor.name)})
    adjacent = 0
    for layer in layers:
        gate = by_name[f"blk.{layer}.ffn_gate_exps.weight"]
        up = by_name[f"blk.{layer}.ffn_up_exps.weight"]
        down = by_name[f"blk.{layer}.ffn_down_exps.weight"]
        expected_up = gate.new_rel_offset + pad_to(gate.tensor.n_bytes, alignment)
        expected_down = up.new_rel_offset + pad_to(up.tensor.n_bytes, alignment)
        if up.new_rel_offset == expected_up and down.new_rel_offset == expected_down:
            adjacent += 1
    return adjacent, len(layers)


def summarize(info: GGUFInfo, plan: list[PlanItem]) -> None:
    expert_tensors = [item.tensor for item in plan if expert_match(item.tensor.name)]
    moved = sum(
        1
        for index, item in enumerate(plan)
        if info.tensors[index].name != item.tensor.name
    )
    adjacent, layers = validate_layer_adjacency(plan, info.alignment)
    payload = sum(item.tensor.n_bytes for item in plan)

    print(f"input: {info.path}")
    print(f"gguf: v{info.version}, {info.kv_count} metadata keys, {info.tensor_count} tensors")
    print(f"alignment: {info.alignment}")
    print(f"routed expert tensors: {len(expert_tensors)} across {layers} layers")
    print(f"adjacent routed layer triplets in output: {adjacent}/{layers}")
    print(f"tensor directory positions changed: {moved}")
    print(f"tensor payload: {payload:,} bytes ({payload / (1024 ** 3):.2f} GiB)")


def write_reordered(info: GGUFInfo, plan: list[PlanItem], out_path: Path, force: bool) -> None:
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")
    if tmp_path.exists():
        raise FileExistsError(f"temporary file already exists: {tmp_path}")

    total_payload = sum(item.tensor.n_bytes for item in plan)
    copied = 0
    next_report = 0

    with info.path.open("rb") as src, tmp_path.open("wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", info.version))
        out.write(struct.pack("<Q", info.tensor_count))
        out.write(struct.pack("<Q", info.kv_count))
        out.write(info.kv_blob)
        for item in plan:
            write_tensor_info(out, item)
        write_padding(out, out.tell(), info.alignment)

        for item in plan:
            src.seek(item.tensor.data_offset)
            copy_exact(src, out, item.tensor.n_bytes)
            write_padding(out, item.tensor.n_bytes, info.alignment)
            copied += item.tensor.n_bytes
            pct = int(copied * 100 / total_payload) if total_payload else 100
            if pct >= next_report:
                print(
                    f"copied {copied / (1024 ** 3):.2f} GiB / "
                    f"{total_payload / (1024 ** 3):.2f} GiB ({pct}%)",
                    flush=True,
                )
                next_report += 5

        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp_path, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite a DS4 GGUF so routed expert tensors are adjacent as gate/up/down per layer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input GGUF, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output GGUF, default: INPUT with -ssd-layer-layout.gguf suffix",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the rewrite plan without writing")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    args = parser.parse_args()

    input_path = args.input
    out_path = args.out or output_path_for(input_path)

    info = parse_gguf(input_path)
    plan = build_plan(info)
    summarize(info, plan)
    print(f"output: {out_path}")

    if args.dry_run:
        return 0

    write_reordered(info, plan, out_path, args.force)
    final_size = out_path.stat().st_size
    print(f"wrote: {out_path}")
    print(f"file size: {final_size:,} bytes ({final_size / (1024 ** 3):.2f} GiB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
