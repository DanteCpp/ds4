#!/bin/sh
# Two-dimensional expert-swap quality sweep.
#
#   k = router candidate window            -> --expert-swap <k> positional
#   min_prob_ratio = minimum probability ratio -> DS4_EXPERT_SWAP_MIN_PROB_RATIO env var
#
# Every cell is scored against a single streaming baseline (expert-swap OFF).
# Only the non-redundant cells are run: min_prob_ratio=0 disables substitution
# before k is consulted, and k=6 (==n_expert_used) has an empty backup pool, so
# both are behaviorally identical to the baseline and are not re-scored here.
#
# Usage:
#   ./sweep.sh OUT_DIR [MODEL] [MANIFEST] [CTX] [CACHE]
#
# Defaults: MODEL=../../ds4flash.gguf  MANIFEST=data/flash/manifest.tsv
#           CTX=4096  CACHE=16GB
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 OUT_DIR [MODEL] [MANIFEST] [CTX] [CACHE]" >&2
    exit 2
fi

# Resolve paths relative to this script, then run from the repo root: the Metal
# backend loads its shader sources from a `metal/` directory relative to the
# current working directory, and that only exists at the repo root. Running
# elsewhere fails with "metal backend unavailable; aborting startup".
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

OUT_DIR=$1
MODEL=${2:-$REPO_ROOT/ds4flash.gguf}
MANIFEST=${3:-$SCRIPT_DIR/data/flash/manifest.tsv}
CTX=${4:-4096}
CACHE=${5:-12GB}

# Make OUT_DIR absolute so it doesn't depend on the cwd we run the scorer from.
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$PWD/$OUT_DIR" ;;
esac

K_VALUES="12 18 24 30"
MIN_RATIO_VALUES="0.25 0.5 0.75 1"

SCORER="$SCRIPT_DIR/score_official"
SCORER_DEPS="
$SCRIPT_DIR/score_official.c
$REPO_ROOT/ds4.c
$REPO_ROOT/ds4.h
$REPO_ROOT/ds4_expert_swap.c
$REPO_ROOT/ds4_expert_swap.h
$REPO_ROOT/ds4_ssd.c
$REPO_ROOT/ds4_gpu.h
$SCRIPT_DIR/../Makefile
$REPO_ROOT/Makefile
"

scorer_needs_build() {
    [ ! -x "$SCORER" ] && return 0
    for dep in $SCORER_DEPS; do
        [ "$dep" -nt "$SCORER" ] && return 0
    done
    return 1
}

if scorer_needs_build; then
    echo "building/updating $SCORER" >&2
    make -C "$SCRIPT_DIR/.." quality-score
fi

[ -x "$SCORER" ] || { echo "error: $SCORER not built (run: make -C $SCRIPT_DIR/.. quality-score)" >&2; exit 1; }
if strings "$SCORER" | grep -q -- "--expert-swap needs K and P"; then
    echo "error: $SCORER is stale; expected --expert-swap [K] with DS4_EXPERT_SWAP_MIN_PROB_RATIO" >&2
    echo "       rebuild with: make -C $SCRIPT_DIR/.. quality-score" >&2
    exit 1
fi

# --- progress helpers -------------------------------------------------------
# Total cells = baseline + every (k, min_prob_ratio) combination.
N_K=$(echo $K_VALUES | wc -w | tr -d ' ')
N_RATIO=$(echo $MIN_RATIO_VALUES | wc -w | tr -d ' ')
TOTAL=$((N_K * N_RATIO + 1))
DONE=0
SWEEP_START=$(date +%s)

# log MESSAGE -- prominent, timestamped banner so status stands out from the
# scorer's per-case output. Goes to stderr so it stays visible even if stdout
# is redirected, and is flushed immediately.
log() {
    printf '\n========================================================\n' >&2
    printf '>>> [%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2
    printf '========================================================\n\n' >&2
}

# fmt_dur SECONDS -- humanize a duration as e.g. "1h02m03s".
fmt_dur() {
    s=$1
    printf '%dh%02dm%02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# step_done LABEL STEP_START -- bump the counter and report progress + ETA.
step_done() {
    DONE=$((DONE + 1))
    now=$(date +%s)
    step_elapsed=$((now - $2))
    total_elapsed=$((now - SWEEP_START))
    remaining=$((TOTAL - DONE))
    eta=""
    if [ "$DONE" -gt 0 ] && [ "$remaining" -gt 0 ]; then
        eta=" eta~$(fmt_dur $(((total_elapsed / DONE) * remaining)))"
    fi
    log "done $1 in $(fmt_dur "$step_elapsed") [$DONE/$TOTAL elapsed $(fmt_dur "$total_elapsed")$eta]"
}
# ---------------------------------------------------------------------------

# The Metal backend needs the repo-root metal/ sources on the cwd.
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"
BASELINE="$OUT_DIR/baseline.tsv"

log "sweep start: $TOTAL cells (1 baseline + $N_K k-values x $N_RATIO min-ratio values), out=$OUT_DIR"

log "[1/$TOTAL] baseline (expert-swap OFF, cache=$CACHE)"
step_start=$(date +%s)
"$SCORER" "$MODEL" "$MANIFEST" "$BASELINE" "$CTX" \
    --ssd-streaming --ssd-streaming-cache-experts "$CACHE"
step_done "baseline" "$step_start"

for k in $K_VALUES; do
    for min_ratio in $MIN_RATIO_VALUES; do
        cell="$OUT_DIR/k${k}-minratio${min_ratio}.tsv"
        log "[$((DONE + 1))/$TOTAL] scoring k=$k min_prob_ratio=$min_ratio (cache=$CACHE)"
        step_start=$(date +%s)
        DS4_EXPERT_SWAP_MIN_PROB_RATIO="$min_ratio" \
            "$SCORER" "$MODEL" "$MANIFEST" "$cell" "$CTX" \
            --ssd-streaming --ssd-streaming-cache-experts "$CACHE" \
            --expert-swap "$k"
        step_done "k=$k min_prob_ratio=$min_ratio" "$step_start"

        log "compare k=$k min_prob_ratio=$min_ratio vs baseline"
        python3 "$SCRIPT_DIR/compare_scores.py" "$BASELINE" "$cell"
    done
done

log "sweep complete in $(fmt_dur $(($(date +%s) - SWEEP_START))): results in $OUT_DIR"
