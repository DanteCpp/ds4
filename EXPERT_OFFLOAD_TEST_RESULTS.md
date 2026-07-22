# Expert Offload — Two-Machine Test Results

Live record of the `expert-offload` branch bring-up tests across **mone**
(coordinator) and **mtwo** (worker), per `EXPERT_OFFLOAD_HANDOFF.md`.
Appended to as each phase is run. Dates are local.

---

## Test environment

### Machines (roles per the plan)

| Host | Role | en0 (LAN/WiFi) | bridge0 (Thunderbolt) | MTU |
|------|------|----------------|-----------------------|-----|
| `mone.local` | coordinator | `192.168.1.141/24` | `169.254.94.152/16` | 1500 |
| `mtwo.local` | worker | `192.168.1.132/24` | `169.254.14.255/16` | 1500 |

- Both on `branch expert-offload`. mone tip `0b0c60e` (+ uncommitted doc trims).
- bridge0 link confirmed: mone sees mtwo at `36:6b:fa:18:1f:40` on bridge0;
  mtwo's `bridge0` ether matches. Both on `169.254.0.0/16` (same subnet) → direct
  Thunderbolt path, no router.

### Model / binaries

- Model: `ds4flash.gguf` (DeepSeek-V4-Flash, IQ2XXS pack, ~86 GB).
  - On mone: symlink → `/Users/dante/ns4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
  - On mtwo: present in the ds4 build dir.
- Binaries built on both: `ds4`, `ds4-offload-ping`.
  - Offload flags compiled in (`--expert-server`, `--expert-offload <host>`,
    `--expert-offload-bind`, `--expert-offload-port`); not yet listed in `--help`.

### Build / code state at start of testing

- **Built:** transport (`ds4_offload.{c,h}`: HELLO/EXPERT_REQ/EXPERT_RESP, client,
  worker loop, residency bitmap); worker CLI (`ds4 --expert-server` →
  `run_offload_worker`); Unit B worker compute
  (`ds4_engine_offload_compute_experts` → `ds4_gpu_offload_run_layer`,
  reusing `ds4_gpu_routed_moe_one_tensor`) — **unverified**; Phase-0 ping tool.
- **NOT built:** Unit C coordinator decode splice — `--expert-offload <host>` is
  parsed (`ds4_cli.c:1945`) but never acted on (no `cfg.offload.host` branch in
  `main()`, no `ds4_offload_client_connect`, no `offload_active`, no splice at the
  6 routed-MoE sites `ds4.c:23824`/23009/23090/23271/23477/23569). A coordinator
  run today silently falls back to **solo**. Also missing: the Step-1 per-token
  hidden-state-hash verification harness.

---

## Phase 0 — bridge0 RTT gate (go/no-go)

`ds4-offload-ping`: 8192 B (n_embd f16) bounced over a `TCP_NODELAY` socket,
reporting p50/p99 RTT. The single number that picks the §7 throughput row.
**Path used: Thunderbolt bridge0** (plan: "Do NOT use en4 100 Mbit; bridge0 only").

### Connectivity (ICMP sanity, mone → mtwo)

| Path | min | avg | max | notes |
|------|-----|-----|-----|-------|
| bridge0 `169.254.14.255` | 0.889 ms | 0.978 ms | 1.067 ms | Thunderbolt, direct |
| LAN `192.168.1.132` | 8.274 ms | 18.427 ms | 28.579 ms | WiFi, ~18× slower |

### Ping-pong gate (TCP_NODELAY, 8 KB payload)

| Run | Iters | min | p50 | p99 | max | mean | verdict |
|-----|-------|-----|-----|-----|-----|------|---------|
| mone loopback `127.0.0.1` (baseline) | 200 | 0.021 | 0.121 | 0.161 | 0.269 | 0.102 | GO — ~26 t/s projected |
| **mone → mtwo over bridge0** `169.25.14.255:47300` | 2000 | 0.081 | **0.220** | 0.314 | 0.902 | 0.209 | **GO — ~26 t/s projected** |

Raw (cross-machine):
```
dante@mone ds4 % ./ds4-offload-ping --host 169.254.14.255 --port 47300 --iters 2000
ds4-offload-ping: 2000 round trips of 8192 B over 169.254.14.255:47300
  RTT ms:  min 0.081  p50 0.220  p99 0.314  max 0.902  mean 0.209
  gate: p50 0.220 ms -> GO (tuned row: ~26 t/s projected)
```

**Result: PASS.** Cross-machine Thunderbolt RTT (p50 0.22 ms) is only ~1.8×
loopback (0.12 ms) and well under the gate threshold. p99 0.31 ms — tight tail.
The 8 KB activation round trip is cheap enough to hide behind on-GPU local expert
compute (the plan's whole premise). WiFi (~18 ms) would have been a hard NO-GO,
confirming bridge0 is mandatory.

---

## Phase 1 — Worker bring-up on mtwo (loads model, serves experts)

_Status: PASS — worker up, model loaded (SSD streaming), listening and reachable from mone over bridge0._

Command run on mtwo:
```sh
./ds4 --expert-server --expert-offload-port 47300 \
      -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 512
```

Raw worker output (mtwo):
```
ds4: Metal device Apple M2 Max, 32.00 GiB RAM
ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices
ds4: drift-patch flags hc_stable=on norm_unify=on kv_raw_f32=off rope_exp2_log2=off math_safe=off tensor_matmul=off
ds4: Metal SSD streaming mode enabled; full model residency and warmup are skipped
ds4: WARNING: SSD streaming expert cache (512 experts) is under twice the per-token routed working set (43 layers x 6 experts = 258); expect heavy thrashing below 3.40 GiB
ds4: SSD streaming initial metal model map restricted to token embedding (1 spans, 0.99 GiB tensor span)
ds4: metal backend initialized for graph diagnostics
ds4: memory: KV 0.61 GiB (raw 0.36 + compressed 0.25) + buffers 0.25 GiB + resident model 0.99 GiB + expert cache 3.38 GiB = 5.22 GiB planned
ds4: memory detail: ctx=32768 prefill_cap=4096 raw_kv_rows=4352 compressed_kv_rows=8194 backend=metal
ds4-offload: expert-server listening on *:47300
```

| Check | Expected | Observed |
|-------|----------|----------|
| Worker starts, prints listening line | `expert-server listening on *:47300` | ✅ printed |
| Model loads (DeepSeek-V4-Flash) | n_embd 4096 / n_layer 43 guard passes | ✅ guard passed (worker reached `run_offload_worker`'s listen line, which aborts on mismatch) |
| Memory stable under cache 512 | no rising compressor/swapins | ✅ planned 5.22 GiB on 32 GB M2 Max (~16%); no pressure (vm_stat not continuously logged; memory plan is the evidence) |
| Port reachable from mone (bridge0) | TCP connect to `169.254.14.255:47300` succeeds | ✅ `nc -z -v -w2 169.254.14.255 47300` → `Connection ... succeeded!` (rc=0) |

Notes:
- **No OOM pressure.** Total planned 5.22 GiB on a 32 GB M2 Max (~26 GB free). SSD
  streaming on, so the ~86 GB model is never mlocked — only embedding + a 3.38 GiB
  expert cache are resident.
- **Thrashing WARNING is perf-only, not correctness.** Cache 512 experts (3.38 GiB)
  is just under 2× the per-token working set (43×6=258 → 2×=516), so the worker's
  cold-side cache may thrash under load. mtwo has ~26 GB free, so for Phase 2 we can
  bump `--ssd-streaming-cache-experts` to ≥1024 to clear the warning and cut SSD
  thrashing on the worker's cold experts.
- A real HELLO + `EXPERT_REQ` round trip still cannot be driven from mone — the
  coordinator client is Unit C (not built). Phase 1 only proves the worker *starts,
  loads, listens, and is reachable*. The `nc` probe made mtwo print a brief
  `coordinator connected` + HELLO/disconnect line; the worker re-listened after.

---

## Phase 2 — Offloaded decode correctness + latency hiding (needs Unit C)

_Status: blocked on building Unit C on mone (coordinator decode splice)._

Acceptance (handoff correctness invariant): with the worker returning the exact
experts the coordinator would otherwise compute locally, the **per-token
hidden-state hash of the offloaded run == solo run** (within IQ2 noise); then
decode t/s offloaded vs solo, and wire MB/token.

| Metric | Solo | Offloaded (bridge0) | Δ |
|--------|------|---------------------|---|
| Per-token hidden-state hash | | | |
| Decode t/s | | | |
| Wire MB/token | | | |
| Worker tax (mtwo GPU %, t/s served) | | | |
| Cache hit rate (residency) | | | |

---

## Open items before the real test can run

1. **Build Unit C on mone:** coordinator client bring-up in `ds4_cli.c main()`
   (`ds4_offload_client_connect` after `ds4_engine_open`, stash client +
   `offload_active` on engine, close on shutdown, solo fallback on connect fail);
   decode splice at the 6 routed-MoE sites gated on `offload_active`
   (residency split → issue remote `EXPERT_REQ` first, compute local experts with
   remote weights zeroed, `collect` + add f16 partial before shared/HC combine).
2. **Build Step-1 hidden-state-hash harness:** fixed hidden + expert set,
   compute weighted sum two ways — (a) normal decode routed forward,
   (b) `ds4_engine_offload_compute_experts` — assert match within IQ2 noise
   (single-process first, lower OOM risk). Also resolves the two Unit-B
   unknowns: `x` dtype (f32 vs f16) and command-buffer lifecycle.
3. (Phase 1 tail) Seed mone/mtwo split from `ds4_streaming_hotlist.inc` →
   residency bitmap + streaming cache warm-start.