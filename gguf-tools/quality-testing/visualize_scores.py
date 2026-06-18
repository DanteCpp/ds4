#!/usr/bin/env python3
"""Visualize expert-swap loss-degradation scores against a streaming baseline.

Consumes the TSV files emitted by ``score_official`` (columns: id,
prompt_tokens, target_tokens, nll, avg_nll, first_match, greedy_lcp) and
renders a multi-panel figure comparing one or more swap configurations to the
baseline. The summary metrics match ``compare_scores.py`` so the plot and the
text report agree.

Usage:
    visualize_scores.py BASELINE.tsv LABEL=FILE.tsv [LABEL=FILE.tsv ...] \
        [--out plot.png] [--title TEXT]

If a comparison argument has no ``LABEL=`` prefix, the file stem is used as the
label, e.g. ``swap-k12-minratio03.tsv`` -> ``k12-minratio03``.

Example:
    visualize_scores.py /tmp/swap-baseline.tsv \
        "k6 minratio0=/tmp/swap-k6-minratio0.tsv" \
        "k12 minratio0.3=/tmp/swap-k12-minratio03.tsv" \
        --out /tmp/expert-swap.png

Sweep mode
----------
``sweep.sh`` emits a directory holding ``baseline.tsv`` plus a 2-D grid of
``k<K>-minratio<R>.tsv`` cells (router window K x min_prob_ratio R). To render
that grid as a set of annotated heatmaps -- one per metric, with K on the rows
and min_prob_ratio on the columns -- use the ``sweep`` subcommand:

    visualize_scores.py sweep /tmp/expert-swap-sweep \
        --out /tmp/expert-swap-heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-9


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows: dict[str, dict[str, float]] = {}
        for row in csv.DictReader(fp, delimiter="\t"):
            rows[row["id"]] = {
                "target_tokens": int(row["target_tokens"]),
                "nll": float(row["nll"]),
                "avg_nll": float(row["avg_nll"]),
                "first_match": int(row["first_match"]),
                "greedy_lcp": int(row["greedy_lcp"]),
            }
        return rows


def parse_labeled(arg: str) -> tuple[str, Path]:
    """Split ``LABEL=path`` or fall back to the file stem as the label."""
    if "=" in arg:
        label, _, path = arg.partition("=")
        return label.strip(), Path(path.strip())
    path = Path(arg)
    return path.stem, path


class Comparison:
    """Aggregated baseline-vs-config metrics over the shared case set."""

    def __init__(self, label: str, baseline: dict, candidate: dict):
        self.label = label
        ids = sorted(set(baseline) & set(candidate))
        if not ids:
            raise SystemExit(f"{label}: no common cases with baseline")

        self.ids = ids
        self.base_avg = np.array([baseline[i]["avg_nll"] for i in ids])
        self.cand_avg = np.array([candidate[i]["avg_nll"] for i in ids])
        base_nll = np.array([baseline[i]["nll"] for i in ids])
        cand_nll = np.array([candidate[i]["nll"] for i in ids])
        toks = np.array([baseline[i]["target_tokens"] for i in ids])

        for i in ids:
            if baseline[i]["target_tokens"] != candidate[i]["target_tokens"]:
                raise SystemExit(f"{label}: token-count mismatch for {i}")

        self.tokens = int(toks.sum())
        self.case_delta = self.cand_avg - self.base_avg  # per-case avg_nll delta
        self.nll_delta = cand_nll - base_nll  # per-case total-nll delta
        self.base_token_avg = float(base_nll.sum() / self.tokens)
        self.cand_token_avg = float(cand_nll.sum() / self.tokens)
        self.delta_token_avg = self.cand_token_avg - self.base_token_avg
        self.rel_change = (self.cand_token_avg / self.base_token_avg - 1.0) * 100.0

        self.new_wins = int((self.nll_delta < -EPS).sum())
        self.old_wins = int((self.nll_delta > EPS).sum())
        self.ties = int((np.abs(self.nll_delta) <= EPS).sum())

        self.base_first = int(sum(baseline[i]["first_match"] for i in ids))
        self.cand_first = int(sum(candidate[i]["first_match"] for i in ids))
        self.base_lcp = float(np.mean([baseline[i]["greedy_lcp"] for i in ids]))
        self.cand_lcp = float(np.mean([candidate[i]["greedy_lcp"] for i in ids]))


def print_summary(comps: list[Comparison]) -> None:
    hdr = (
        f"{'config':<14}{'base_nll':>11}{'new_nll':>11}{'delta':>11}"
        f"{'rel%':>9}{'win/tie/loss':>16}{'1st(b/n)':>11}{'lcp(b/n)':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in comps:
        print(
            f"{c.label:<14}{c.base_token_avg:>11.6f}{c.cand_token_avg:>11.6f}"
            f"{c.delta_token_avg:>+11.6f}{c.rel_change:>+8.2f}%"
            f"{f'{c.new_wins}/{c.ties}/{c.old_wins}':>16}"
            f"{f'{c.base_first}/{c.cand_first}':>11}"
            f"{f'{c.base_lcp:.2f}/{c.cand_lcp:.2f}':>14}"
        )


def make_figure(comps: list[Comparison], title: str, out: Path) -> None:
    n = len(comps)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))
    fig = plt.figure(figsize=(15, 4 * n + 1.5))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1.1] + [3] * n)

    # --- top row: summary bars across configs ---------------------------------
    ax_delta = fig.add_subplot(gs[0, 0])
    labels = [c.label for c in comps]
    deltas = [c.delta_token_avg for c in comps]
    bar_colors = ["#d62728" if d > EPS else "#2ca02c" for d in deltas]
    ax_delta.bar(labels, deltas, color=bar_colors)
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.set_title("avg_nll delta (new - baseline)\n(+ = degradation)", fontsize=9)
    ax_delta.tick_params(axis="x", rotation=30, labelsize=8)
    ax_delta.grid(axis="y", alpha=0.3)

    ax_wtl = fig.add_subplot(gs[0, 1])
    x = np.arange(n)
    wins = [c.new_wins for c in comps]
    ties = [c.ties for c in comps]
    losses = [c.old_wins for c in comps]
    ax_wtl.bar(x, wins, label="new better", color="#2ca02c")
    ax_wtl.bar(x, ties, bottom=wins, label="tie", color="#999999")
    ax_wtl.bar(x, losses, bottom=np.add(wins, ties), label="new worse", color="#d62728")
    ax_wtl.set_xticks(x)
    ax_wtl.set_xticklabels(labels, rotation=30, fontsize=8)
    ax_wtl.set_title("per-case win / tie / loss", fontsize=9)
    ax_wtl.legend(fontsize=7)

    ax_q = fig.add_subplot(gs[0, 2])
    width = 0.35
    ax_q.bar(x - width / 2, [c.cand_first for c in comps], width,
             label="first-token match", color="#1f77b4")
    ax_q.bar(x + width / 2, [c.cand_lcp for c in comps], width,
             label="avg greedy LCP", color="#ff7f0e")
    ax_q.axhline(comps[0].base_first, color="#1f77b4", ls="--", lw=0.8,
                 label="baseline 1st")
    ax_q.axhline(comps[0].base_lcp, color="#ff7f0e", ls="--", lw=0.8,
                 label="baseline LCP")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(labels, rotation=30, fontsize=8)
    ax_q.set_title("greedy quality (vs baseline dashed)", fontsize=9)
    ax_q.legend(fontsize=7)

    # --- per-config rows ------------------------------------------------------
    for r, c in enumerate(comps, start=1):
        color = colors[r - 1]

        # scatter: baseline vs config per-case avg_nll
        ax_sc = fig.add_subplot(gs[r, 0])
        lo = float(min(c.base_avg.min(), c.cand_avg.min()))
        hi = float(max(c.base_avg.max(), c.cand_avg.max()))
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax_sc.scatter(c.base_avg, c.cand_avg, s=14, alpha=0.6, color=color)
        ax_sc.set_xlabel("baseline avg_nll", fontsize=8)
        ax_sc.set_ylabel(f"{c.label} avg_nll", fontsize=8)
        ax_sc.set_title(f"{c.label}: per-case avg_nll\n(above line = worse)", fontsize=9)
        ax_sc.grid(alpha=0.3)

        # histogram of per-case avg_nll deltas
        ax_h = fig.add_subplot(gs[r, 1])
        ax_h.hist(c.case_delta, bins=25, color=color, alpha=0.8)
        ax_h.axvline(0, color="k", lw=0.8)
        ax_h.axvline(c.case_delta.mean(), color="red", ls="--", lw=1.0,
                     label=f"mean {c.case_delta.mean():+.4f}")
        ax_h.set_xlabel("per-case avg_nll delta (new - baseline)", fontsize=8)
        ax_h.set_ylabel("cases", fontsize=8)
        ax_h.set_title(f"{c.label}: delta distribution", fontsize=9)
        ax_h.legend(fontsize=7)
        ax_h.grid(axis="y", alpha=0.3)

        # sorted per-case delta (degradation profile)
        ax_s = fig.add_subplot(gs[r, 2])
        srt = np.sort(c.case_delta)
        ax_s.bar(range(len(srt)), srt,
                 color=["#d62728" if d > EPS else "#2ca02c" for d in srt])
        ax_s.axhline(0, color="k", lw=0.8)
        ax_s.set_xlabel("cases (sorted by delta)", fontsize=8)
        ax_s.set_ylabel("avg_nll delta", fontsize=8)
        ax_s.set_title(
            f"{c.label}: delta={c.delta_token_avg:+.5f} ({c.rel_change:+.2f}%)",
            fontsize=9,
        )
        ax_s.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------- #
# Sweep heatmaps
# --------------------------------------------------------------------------- #

CELL_RE = re.compile(r"^k(\d+)-minratio([0-9]+(?:\.[0-9]+)?)\.tsv$")


def discover_sweep(directory: Path, baseline_name: str = "baseline.tsv"):
    """Load a sweep directory into a baseline plus a {(k, min_ratio): Comparison} grid.

    Returns ``(comps, ks, min_ratios)`` where ``comps`` is keyed by the parsed
    ``(k, min_ratio)`` pair and ``ks``/``min_ratios`` are the sorted unique axis
    values.
    """
    base_path = directory / baseline_name
    if not base_path.is_file():
        raise SystemExit(f"sweep: missing baseline {base_path}")
    baseline = load(base_path)

    comps: dict[tuple[int, float], Comparison] = {}
    for path in sorted(directory.glob("k*-minratio*.tsv")):
        m = CELL_RE.match(path.name)
        if not m:
            continue
        k, min_ratio = int(m.group(1)), float(m.group(2))
        label = f"k{k} minratio{min_ratio:g}"
        comps[(k, min_ratio)] = Comparison(label, baseline, load(path))

    if not comps:
        raise SystemExit(f"sweep: no k<K>-minratio<R>.tsv cells found in {directory}")

    ks = sorted({k for k, _ in comps})
    min_ratios = sorted({min_ratio for _, min_ratio in comps})
    return comps, ks, min_ratios


# Each entry: (title, attribute, value-format, colormap, diverging-around-zero,
# higher-is-better). ``higher_is_better`` only flips the colormap direction for
# non-diverging maps so that "good" is always green.
HEATMAP_METRICS = [
    ("avg_nll delta (new - base)\n+ = degradation",
     "delta_token_avg", "{:+.5f}", "RdYlGn_r", True, False),
    ("relative nll change (%)\n+ = degradation",
     "rel_change", "{:+.2f}", "RdYlGn_r", True, False),
    ("new candidate avg_nll\n(token-weighted)",
     "cand_token_avg", "{:.5f}", "viridis_r", False, False),
    ("per-case net wins (new - old)\n+ = new better",
     "net_wins", "{:+d}", "RdYlGn", True, True),
    ("first-token match delta (new - base)\n+ = better",
     "first_delta", "{:+d}", "RdYlGn", True, True),
    ("greedy LCP delta (new - base)\n+ = better",
     "lcp_delta", "{:+.2f}", "RdYlGn", True, True),
]


def metric_value(c: Comparison, attr: str) -> float:
    """Resolve a heatmap metric, including a few derived ones."""
    if attr == "net_wins":
        return c.new_wins - c.old_wins
    if attr == "first_delta":
        return c.cand_first - c.base_first
    if attr == "lcp_delta":
        return c.cand_lcp - c.base_lcp
    return getattr(c, attr)


def _draw_heatmap(ax, grid, ks, min_ratios, title, fmt, cmap, diverging, higher_better):
    finite = grid[np.isfinite(grid)]
    if diverging and finite.size:
        vmax = float(np.nanmax(np.abs(finite))) or 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if not diverging and higher_better:
            # plain colormaps already run low->high; "_r" handled via name
            pass
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(min_ratios)))
    ax.set_xticklabels([f"{min_ratio:g}" for min_ratio in min_ratios], fontsize=8)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks], fontsize=8)
    ax.set_xlabel("min_prob_ratio", fontsize=8)
    ax.set_ylabel("router window (k)", fontsize=8)
    ax.set_title(title, fontsize=9)

    # Annotate each cell, picking text color for contrast against the fill.
    rng = (vmax - vmin) or 1.0
    for i in range(len(ks)):
        for j in range(len(min_ratios)):
            v = grid[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=7, color="#888888")
                continue
            shade = (v - vmin) / rng
            txt_color = "white" if shade < 0.25 or shade > 0.78 else "black"
            label = fmt.format(int(v) if "d}" in fmt else v)
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7.5, color=txt_color)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)


def make_sweep_figure(comps, ks, min_ratios, title: str, out: Path) -> None:
    ncols = 3
    nrows = (len(HEATMAP_METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (mtitle, attr, fmt, cmap, diverging, higher) in zip(axes, HEATMAP_METRICS):
        grid = np.full((len(ks), len(min_ratios)), np.nan)
        for i, k in enumerate(ks):
            for j, min_ratio in enumerate(min_ratios):
                c = comps.get((k, min_ratio))
                if c is not None:
                    grid[i, j] = metric_value(c, attr)
        _draw_heatmap(ax, grid, ks, min_ratios, mtitle, fmt, cmap, diverging, higher)

    for ax in axes[len(HEATMAP_METRICS):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


def print_sweep_summary(comps, ks, min_ratios) -> None:
    print_summary([
        comps[(k, min_ratio)]
        for k in ks
        for min_ratio in min_ratios
        if (k, min_ratio) in comps
    ])


def run_sweep(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="visualize_scores.py sweep",
        description="Render sweep.sh k/min_prob_ratio grid output as metric heatmaps.")
    ap.add_argument("directory", help="sweep output dir (baseline.tsv + k*-minratio*.tsv)")
    ap.add_argument("--baseline", default="baseline.tsv",
                    help="baseline filename within the dir (default: %(default)s)")
    ap.add_argument("--out", default="expert-swap-heatmap.png",
                    help="output image path (default: %(default)s)")
    ap.add_argument("--title",
                    default="Expert-swap sweep vs streaming baseline (k x min_prob_ratio)")
    args = ap.parse_args(argv)

    comps, ks, min_ratios = discover_sweep(Path(args.directory), args.baseline)
    print_sweep_summary(comps, ks, min_ratios)
    make_sweep_figure(comps, ks, min_ratios, args.title, Path(args.out))
    return 0


def main() -> int:
    # ``sweep`` subcommand renders the k/min_prob_ratio grid as heatmaps; the bare form keeps
    # the original baseline + LABEL=file comparison panels.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        return run_sweep(sys.argv[2:])

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="baseline TSV from score_official")
    ap.add_argument("configs", nargs="+",
                    help="comparison TSVs, optionally LABEL=path")
    ap.add_argument("--out", default="expert-swap-degradation.png",
                    help="output image path (default: %(default)s)")
    ap.add_argument("--title", default="Expert-swap loss degradation vs streaming baseline")
    args = ap.parse_args()

    baseline = load(Path(args.baseline))
    comps = []
    for arg in args.configs:
        label, path = parse_labeled(arg)
        comps.append(Comparison(label, baseline, load(path)))

    print_summary(comps)
    make_figure(comps, args.title, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
