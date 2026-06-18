# SSD-streaming expert features

This document specifies a feature that improves generation speed under the
SSD-streaming regime. It is additive: with the feature disabled the engine
must behave exactly as it does today.

Implementation branch: `expert_swap`.

Target environment: an M-series "Max" MacBook with 32 GB of unified memory,
where the full routed-expert set does not fit in RAM and is streamed from the
GGUF on demand.

## Background and shared terminology

Under `--ssd-streaming`, the non-routed weights stay resident in RAM. Routed
MoE experts are not all loaded at once: the experts that fit in the bounded
**in-memory expert cache** are resident, while the rest remain only in the GGUF
file on SSD until needed. When routing selects a non-resident expert, the engine
loads that expert from the GGUF into the cache, possibly evicting another
expert. This is tolerable for prefill but costly for generation, because every
new token re-routes through the experts.

Relevant model facts (used by the examples below):

| Model | Routed experts (`n_expert`) | Experts used per token (`n_expert_used`) | Shared experts |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 256 (ids `0..255`) | 6 | 1 |
| DeepSeek-V4-Pro | 384 (ids `0..383`) | 6 | 1 |

Routing recap (per layer, per token):

1. The router produces a per-expert score `s(e) = sqrt(softplus(logit(e)))`. This
   score is **unnormalized** and independent per expert (it is *not* a softmax
   over the experts).
2. The top `n_expert_used` (6) experts are selected by the *biased* score
   `s(e) + bias(e)`. This ranked score is what we call the **router score** in
   the rest of this document.
3. The selected experts are combined using their unbiased `s(e)`, normalized to
   sum to 1 and multiplied by `expert_weight_scale`.

### Streaming activation and default expert-cache budget

This feature requires the SSD-streaming regime, but the user should not have to
spell that out. **Passing `--expert-swap` implies `--ssd-streaming`** (it is
turned on automatically if not already requested).

When streaming is enabled this way and `--ssd-streaming-cache-experts` is **not**
given, the expert cache is sized automatically, keeping a 20% safety margin
(consistent with the existing auto-budget behavior):

```
usable_memory      = 0.80 * available_memory        # 20% safety margin
expert_cache_bytes = usable_memory - fixed_resident_bytes
```

`available_memory` is the Metal recommended working set (or free RAM/VRAM on
other backends). `fixed_resident_bytes` is everything that must stay resident
regardless of the routed experts:

- non-routed / fixed-layer weights (attention, router/gate matrices, shared
  experts, embeddings, output head),
- the KV cache for the configured context,
- graph scratch and activations.

Example: a machine reporting **20 GB available** keeps 80% = **16 GB usable**;
after subtracting **4 GB** for the fixed network plus KV cache, the default
expert cache is **12 GB**. The byte budget is then converted to a whole number
of resident experts per layer for the current GGUF (reusing the existing
`ds4_ssd_cache_plan` path).

An explicit `--ssd-streaming-cache-experts <N|sizeGB>` always overrides this
default. The engine **reports the resulting expert cache at startup**, e.g.:

```
expert cache: 12.0 GB  (auto: 80% of 20.0 GB = 16.0 GB usable - 4.0 GB fixed)  ~NN experts/layer
```

(or, when overridden, it states the explicit budget instead of the `auto:`
breakdown).

---

## Expert swap

### Goal

When the router selects an expert that is **not resident** in the expert cache,
avoid the SSD fetch by substituting an expert that **is** resident, provided that
expert is also a plausible choice for this token (it appears within the top-`k`
router window) and its router score is close enough to the missed expert's score
(within the adaptive effective threshold derived from `min_prob_ratio`). If no such
substitute exists, fall back to the normal SSD fetch.

### CLI

```
./ds4 -m ./ds4flash.gguf --expert-swap [k]
DS4_EXPERT_SWAP_MIN_PROB_RATIO=0.3 ./ds4 -m ./ds4flash.gguf --expert-swap 12
```

The only positional argument is the optional router window `k`. The minimum
probability ratio is **not** a positional argument; it is set with the
`DS4_EXPERT_SWAP_MIN_PROB_RATIO` environment variable (default `0.5`). This is
the single, uniform interface across every executable (`ds4`, `ds4-server`,
`ds4-agent`, `ds4-eval`, `ds4-bench`, and `score_official`).

`--expert-swap` implies `--ssd-streaming`; the expert cache is auto-sized (see
"Streaming activation and default expert-cache budget" above) unless
`--ssd-streaming-cache-experts` is also given.

- `k` — **router candidate window**. The engine looks at the top-`k` router-ranked
  experts for the layer. The model still *activates* its fixed `n_expert_used`
  (6) experts; ranks `7..k` form the pool of swap-in candidates. `k = 12` means we
  consider the 6 used experts plus 6 backups beneath them. `k` must be
  `>= n_expert_used`; values above `n_expert` are clamped. The implementation
  must produce a ranked top-`k` expert id/score window, not only the existing
  top-`n_expert_used` activation mask.
- `min_prob_ratio` (`DS4_EXPERT_SWAP_MIN_PROB_RATIO`, default `0.5`) — **minimum probability
  ratio** a candidate must retain relative to the missed expert it replaces. A
  candidate may replace a missed expert only if its router probability is at
  least `min_prob_ratio` times the missed expert's probability (subject to the
  adaptive scaling below). `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0` disables
  substitution and falls through to baseline behavior. The ratio is supplied via
  the `DS4_EXPERT_SWAP_MIN_PROB_RATIO` environment
  variable, never as a positional CLI argument. Quality sweeps must record
  score-gap distributions so the chosen values are interpretable across layers
  and models.

### Definitions (per layer, per token)

- **Activated set** `A` = the top-`n_expert_used` experts (the experts that would
  run with the feature off).
- **Misses** `M` = experts in `A` that are not resident in the cache.
- **Candidate pool** `C` = experts ranked `n_expert_used+1 .. k` (the backups
  inside the window) that **are** resident in the cache and are not already in
  `A`. Because they rank below the activated set, every candidate's router score
  is `<=` every miss's router score.

### Matching algorithm: greedy best-fit

Process misses in **descending router score**. For each miss `m`, compute
`effective_min_prob_ratio(layer, m)` from the adaptive rule below, then choose
the **highest-scoring** unused candidate `c` in `C` such that
`prob(c) >= prob(m) * effective_min_prob_ratio(layer, m)`. If one exists, swap
`m -> c`, remove `c` from `C`, and `c` runs from cache. If none exists, `m` is
fetched from SSD as usual.

This maximizes the number of avoided fetches while keeping each substitution as
close as possible (smallest score gap) to the expert it replaces. If the
candidate pool empties before all misses are handled, the remaining misses are
fetched.

### V1 adaptive threshold from imatrix

V1 uses an imatrix-informed effective threshold by default:

```
effective_min_prob_ratio(layer, missed) =
    min_prob_ratio * layer_scale[layer] * missed_tolerance[layer][missed]
```

`min_prob_ratio` is the `DS4_EXPERT_SWAP_MIN_PROB_RATIO` value. If it is zero,
substitution is disabled before any scaling is applied.

The adaptive scales are static for a loaded model and are derived from routed-MoE
imatrix statistics when available. DS4's imatrix stores per-layer, per-expert
activation-importance vectors for routed gate/up and down tensors. Collapse
those vectors to one scalar importance per `(layer, expert)`:

```
importance[layer][expert] =
    0.5 * mean(log1p(gate_up_imatrix[layer][expert])) +
    0.5 * mean(log1p(down_imatrix[layer][expert]))
```

If one side is unavailable, use the available side. If no imatrix information is
available for a layer/expert, use that layer's median importance so the scale is
neutral.

Compute layer and expert scaling as:

```
layer_importance[layer] = median_expert(importance[layer][expert])
global_layer_importance = median_layer(layer_importance[layer])

layer_scale[layer] = clamp(
    global_layer_importance / layer_importance[layer],
    0.5,
    1.5)

missed_tolerance[layer][missed] = clamp(
    sqrt(layer_importance[layer] / importance[layer][missed]),
    0.5,
    1.5)
```

This makes swaps stricter in layers and for missed experts that the imatrix
marks as more sensitive, while allowing slightly larger gaps for lower-risk
misses. The rule intentionally does not use pairwise expert similarity in v1;
the only candidate-dependent term remains the router score gap.

The runtime should consume compact precomputed scale tables when the GGUF or an
associated sidecar provides them. It should not load full imatrix vectors on the
per-token hot path. If no imatrix-derived scale table is available for the
model, all scales are `1.0`, making the effective ratio equal to
`min_prob_ratio`; the startup report must state whether adaptive imatrix scaling
is active or using this neutral fallback.

### Execution model: top-k planning and I/O overlap

The router score computation and ranked top-`k` selection should be produced by
backend-native GPU work. For the first implementation, the small top-`k` expert
id/score window may be read back to the host, matching the existing
SSD-streaming selected-id readback architecture. The CPU may snapshot cache
residency, classify hits and misses, filter resident candidates, and run the
greedy best-fit loop over that small window.

This CPU planning step must stay narrow: it should operate only on the current
layer/token top-`k` ids and scores plus the cache residency snapshot. It must not
read back full router-score tensors or introduce a synchronization point larger
than the existing selected-expert readback path. A later GPU-only planner is an
optional optimization, not a v1 requirement.

Misses that cannot be substituted still use the normal SSD-streaming fetch path,
but those reads should be issued as early as possible. Once the final activated
set is known, the runtime should preserve the existing resident/missing split
pipeline: issue or continue asynchronous `pread` requests for unavoidable
misses, submit computation for already-resident experts, then wait only before
the layer needs all expert outputs for the final mixture.

The intended pipeline for a layer/token is:

1. Compute router scores and the ranked top-`k` window on the GPU.
2. Read back only the top-`k` ids/scores needed for planning.
3. Snapshot current cache residency for the layer, decide swaps on the CPU, and
   produce the final six experts plus a missing mask.
4. Immediately issue or continue asynchronous `pread` requests for misses that
   remain.
5. Launch resident expert computation from cache while the missing experts are
   loaded.
6. Once the missing experts are resident and safe to bind, launch their
   computation and join with the resident outputs before finalizing the mixture.

This requirement applies even when no swap is possible: the implementation
should preserve the existing resident/missing overlap path to hide as much SSD
latency as possible. Synchronization should occur only where scratch-buffer
reuse or final mixture semantics require it.

### Combination weights

After swaps, the layer still combines exactly `n_expert_used` (6) experts: the
cache-hit members of `A`, the substitutes, and any fetched misses. Each
participating expert contributes with **its own** unbiased router score `s(e)`;
the missed expert it replaced is dropped from the mixture. The 6 weights are then
normalized and scaled exactly as today. (A substitute computes a real expert, so
it carries its own weight rather than the dropped expert's weight.)

### Cache hotness and eviction policy

Expert swap changes the set of experts that run, so it must define how cache
hotness is updated:

- The routed activated set `A` is credited with route hotness, even when an
  activated miss is substituted and not fetched. This preserves the signal that
  the router wanted that expert.
- Each substitute that actually runs is also credited with execution hotness.
  This prevents resident substitutes from looking cold while they are carrying
  work.
- Fetched misses are credited by both rules because they were routed and
  executed.

Eviction remains free to use the existing combined hotness/use-count heuristic,
but metrics must report substituted misses separately from fetched misses. This
is needed to detect a long-running case where an important routed expert is
always substituted and therefore never promoted into the cache.

### Worked example

Configuration: Flash (`n_expert = 256`, `n_expert_used = 6`), `--expert-swap 12`
with `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0.3`.
For this worked example, assume the adaptive imatrix scales are neutral for
these misses, so the effective minimum probability ratio is `0.3`.

Cache residents for layer `l` (subset shown):
`17, 131, 205, 240, 118, 196, 12, 84, 153, 221, ...`

Top-12 router ranking for the current token:

| rank | expert id | router score | resident? | role |
|---:|---:|---:|:--|:--|
| 1 | 131 | 0.91 | yes | activated (hit) |
| 2 | 17  | 0.85 | yes | activated (hit) |
| 3 | 88  | 0.74 | no  | activated (**miss**) |
| 4 | 205 | 0.66 | yes | activated (hit) |
| 5 | 42  | 0.61 | no  | activated (**miss**) |
| 6 | 9   | 0.55 | no  | activated (**miss**) |
| 7 | 240 | 0.50 | yes | candidate |
| 8 | 53  | 0.39 | no  | (resident-miss, not a candidate) |
| 9 | 118 | 0.34 | yes | candidate |
| 10 | 7  | 0.28 | no  | — |
| 11 | 196 | 0.22 | yes | candidate |
| 12 | 61 | 0.10 | no  | — |

- Activated set `A = {131, 17, 88, 205, 42, 9}`. Cache hits: `131, 17, 205`.
- Misses `M = {88 (0.74), 42 (0.61), 9 (0.55)}`.
- Candidate pool `C = {240 (0.50), 118 (0.34), 196 (0.22)}` (resident experts in
  ranks 7..12; `53` and `7` and `61` are not resident, so they are not candidates).

Greedy best-fit with `min_prob_ratio = 0.3`, misses processed high → low:

1. **Miss 88 (0.74).** Eligible candidates need score `>= 0.44`. Available: `240`
   (0.50, gap `0.24 <= 0.3`). Pick the highest-scoring → **swap 88 → 240**.
   Consume `240`.
2. **Miss 42 (0.61).** Eligible need score `>= 0.31`. Available: `118` (0.34, gap
   `0.27 <= 0.3`); `196` (0.22) fails. → **swap 42 → 118**. Consume `118`.
3. **Miss 9 (0.55).** Eligible need score `>= 0.25`. Only `196` (0.22) remains,
   gap `0.55 - 0.22 = 0.33 > 0.3` → no eligible candidate → **fetch 9 from SSD**.

Result: the 6 experts that run for this token are
`{131, 17, 240, 205, 118, 9}`. Two SSD fetches were avoided (`88`, `42`), one
fetch was unavoidable (`9`).

### Edge cases / requirements

- `--expert-swap` enables `--ssd-streaming` automatically (see "Streaming
  activation and default expert-cache budget" above). If `--ssd-streaming` was
  already passed, behavior is unchanged. If the model fully fits in RAM the swap
  logic is inert (there are no misses), which is the correct outcome.
- This feature is strictly additive. The default path, and any run where
  `--expert-swap` is not active, must preserve the current routing, loading,
  cache, logits, and generation behavior.
- The shared expert(s) and non-routed weights are unaffected.
- Degenerate baseline cases must fall through to the existing behavior:
  `k = n_expert_used` (6 for Flash/Pro) has an empty backup candidate pool, and
  `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0` disables substitution. In practice,
  `--expert-swap 6`, and `--expert-swap <any k>` with
  `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0`, should match baseline except for
  startup/reporting noise. Implement `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0` as an
  explicit substitution-disabled fast path so exact score ties cannot
  accidentally swap.
- With a very large `min_prob_ratio`, any resident backup inside the top-`k` window is
  accepted after adaptive scaling.
- Determinism: identical inputs, cache contents, in-flight load state, and
  eviction state must produce identical swaps. Ties on candidate score are
  broken by lower expert id. Quality runs must use a defined cache warm-up and
  seeding protocol so the cache state is comparable across rows.

---

## Quality / loss degradation testing

Expert swap is a lossy inference tradeoff, so quality should be measured with
the same official-continuation method used for GGUF quantization changes in
`gguf-tools/quality-testing`. The primary metric is target-token negative log
likelihood (NLL): collect deterministic official DeepSeek continuations, then
measure how much probability the local run assigns to those exact continuation
tokens.

### Metrics

Use `gguf-tools/quality-testing/compare_scores.py` and report the same fields as
quantization work:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: swap-enabled NLL minus baseline NLL. Positive values
  are loss degradation.
- `case_wins_new_old_ties`: per-prompt NLL wins/ties/losses.
- `first_token_matches`: count of prompts where the greedy first token matches
  the official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation.

For this feature, "old" is the same GGUF with expert swap disabled and "new" is
the same GGUF with SSD streaming enabled and expert swap enabled. This isolates
the runtime substitution loss from quantization or model-file differences.

On the target 32 GB machine, do **not** use a non-streaming or fully resident
run as the baseline. The full routed-expert model does not fit. The baseline is
therefore `--ssd-streaming` with the same expert-cache budget and no active
substitution.

Do not use the auto-sized expert cache for A/B quality rows. The Metal
recommended working set and free-memory estimates can vary between runs, which
can change the resident expert count and invalidate comparisons. Pass an
explicit `--ssd-streaming-cache-experts <N|sizeGB>` for every baseline,
degenerate, and sweep row, and record the resolved startup budget in the result
notes.

### Scorer support

`gguf-tools/quality-testing/score_official` currently scores a model path,
manifest, output path, and context size. Add a narrow way to pass DS4 engine
runtime options to the scorer, specifically:

```
--ssd-streaming
--ssd-streaming-cache-experts <N|sizeGB>
--expert-swap [k]
```

(`min_prob_ratio` is supplied via `DS4_EXPERT_SWAP_MIN_PROB_RATIO`, identical to
every other executable.)

The scorer must still default to the current behavior when no runtime options
are provided, for compatibility with existing quantization evaluation. The
expert-swap quality plan below must always pass `--ssd-streaming` and a fitting
cache budget, because the target 32 GB machine cannot run the full model
resident.

### Baseline and degenerate checks

Build the scorer:

```sh
make -C gguf-tools quality-score
```

Score the streaming baseline and degenerate configurations against the same
manifest. Use the same explicit expert-cache budget for all rows; pick a budget
that fits the 32 GB target machine without paging. Before scoring each row, use
the same cache warm-up/seeding procedure and deterministic prompt order:

```sh
gguf-tools/quality-testing/score_official \
  ./ds4flash.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/expert-swap-baseline.tsv \
  4096 \
  --ssd-streaming --ssd-streaming-cache-experts 12GB

DS4_EXPERT_SWAP_MIN_PROB_RATIO=0 \
gguf-tools/quality-testing/score_official \
  ./ds4flash.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/expert-swap-k6-minratio0.tsv \
  4096 \
  --ssd-streaming --ssd-streaming-cache-experts 12GB --expert-swap 6

python3 gguf-tools/quality-testing/compare_scores.py \
  /tmp/expert-swap-baseline.tsv \
  /tmp/expert-swap-k6-minratio0.tsv
```

(The `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0` prefix sets the env var for that single
invocation.)

Acceptance criterion: `--expert-swap 6` with `DS4_EXPERT_SWAP_MIN_PROB_RATIO=0`
must match baseline within normal floating-point backend noise. `avg_nll`,
first-token matches, and greedy LCP should be identical or explainably bit-level
close. Run an additional `--expert-swap <k>` with
`DS4_EXPERT_SWAP_MIN_PROB_RATIO=0` case to verify that the zero-ratio setting
also falls through to baseline in realistic score distributions.

### Swap quality sweep

Run a small grid over the intended operating region, using the same machine,
same GGUF, same manifest, same context size, and same expert-cache budget:

```sh
DS4_EXPERT_SWAP_MIN_PROB_RATIO=0.3 \
gguf-tools/quality-testing/score_official \
  ./ds4flash.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/expert-swap-k12-minratio03.tsv \
  4096 \
  --ssd-streaming --ssd-streaming-cache-experts 12GB --expert-swap 12

python3 gguf-tools/quality-testing/compare_scores.py \
  /tmp/expert-swap-baseline.tsv \
  /tmp/expert-swap-k12-minratio03.tsv
```

Suggested first sweep:

| k | min_prob_ratio |
|---:|---:|
| 6 | 0 |
| 8 | 0.05 |
| 8 | 0.10 |
| 12 | 0.10 |
| 12 | 0.20 |
| 12 | 0.30 |
| 18 | 0.20 |
| 18 | 0.30 |

For each row, pass `k` as the `--expert-swap` positional and `min_prob_ratio` via
`DS4_EXPERT_SWAP_MIN_PROB_RATIO`, e.g. row `(12, 0.20)` is
`DS4_EXPERT_SWAP_MIN_PROB_RATIO=0.20 ... --expert-swap 12`.

For every row, record:

- the comparator metrics above,
- the expert-cache budget and measured resident experts per layer,
- swap attempts, successful swaps, avoided SSD fetches, and fallback fetches,
- substituted misses that were not fetched, per-layer score-gap histograms, and
  the distribution of accepted and rejected candidate gaps,
- adaptive imatrix scaling status, `layer_scale` summary, `missed_tolerance`
  summary, and the resulting effective minimum-ratio distribution,
- generation throughput for the same configuration, so the NLL degradation can
  be evaluated against the speed gain.
