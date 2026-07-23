# Distributed Expert Offload — Implementation Plan

Two-machine inference for **DeepSeek-V4-Flash** where the model is held
**entirely in combined RAM** and the second machine acts as a **remote expert
compute server**, not an SSD replacement. A coordinator holds all backbone
weights plus the hottest experts in an LRU cache; a worker holds the remaining
(colder) experts. On a coordinator cache miss the coordinator ships an 8 KB
activation to the worker, which computes the expert and ships the result back —
weights never cross the wire, and the SSD is never on the token critical path.

Status: design. Target hardware: **mone** (M1 Max, 64 GB, coordinator) +
**mtwo** (M2 Max, 32 GB, worker) over **Thunderbolt 4 / `bridge0`**.

> **This is NOT tensor parallelism.** The existing `ds4_tp` module (tensor
> parallelism, RDMA, intended for M5-class machines) is **not used**. M1/M2 do
> not support RDMA, so the only transport is **plain TCP over Thunderbolt**
> (`bridge0`, `TCP_NODELAY`, jumbo frames). This mode is an asymmetric
> coordinator + expert-compute-server; the worker is not a parallel rank and
> never runs the model graph. The reusable core is the **LRU expert cache
> *discipline*** proven by the SSD-streaming cache — promote-on-touch, evict the
> coldest — applied to a **separate, RAM-resident (mlock'd) offload cache**, not
> the SSD-streaming cache instance itself. Its "missing expert" data source is
> the peer, not the local SSD.

> **Architecture note (LRU, clarified).** Earlier drafts implied *reusing the
> SSD-streaming cache instance* (`g_stream_expert_cache_*`) and merely redirecting
> its miss branch. That is **not** the intent. The offload feature has its **own**
> expert cache (`g_offload_expert_cache_*` in `ds4_metal.m`), separate from the
> SSD-streaming cache, populated from the GGUF into mlock'd slabs at startup
> (§5.1). What is reused is the **LRU policy**, not the data structure: the
> coldest experts are evicted and touched experts promoted, exactly as the
> streaming cache does. Startup fills it statically (warm-started from the
> hotlist); Phase 2 (§6) makes it a live LRU with promote-on-hit + background
> eviction.

---

## 1. Verified model facts (read from the GGUF, not memory)

`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` — full GGUF 86.7 GB.

| Parameter | Value |
|---|---|
| Layers (`block_count`) | 43 (all MoE) |
| Routed experts / layer | 256 |
| Experts used / layer (`expert_used_count`) | 6 |
| Shared experts | 1 (always on) |
| Expert FFN length | 2048 |
| Hidden size (`embedding_length`) | 4096 → **activation vector = 8 KB @ fp16** |
| Attention heads / KV heads | 64 / 1 (MLA — tiny KV cache) |

| Tensor group | Size | Notes |
|---|---|---|
| Routed experts | **77.91 GB** | IQ2_XXS gate/up + Q2_K down; **7.08 MB/expert**, 11 008 experts total |
| Attention | 5.80 GB | Q8_0, resident on coordinator |
| Shared expert | 1.15 GB | Q8_0, always active, coordinator |
| Embed / output | 1.62 GB | `token_embd` streamed from SSD (buys back ~1 GB) |
| dense_ffn + other | 0.23 GB | coordinator |
| **Backbone (resident on coordinator)** | **≈ 7.2 GB** | after streaming `token_embd` |

---

## 2. The memory budget — everything fits in RAM

Total RAM 96 GB. Reserve ~10 GB for the two OSes → **86 GB for the model**.
Resident model = 86.7 − ~1 GB (streamed `token_embd`) ≈ **85.7 GB**. It fits with
~0.3 GB margin. **No SSD streaming on the critical path — ever.**

| | RAM | wired ceiling (raised) | holds | expert share |
|---|---|---|---|---|
| mone (coordinator) | 64 GB | ~57 GB | 7.2 GB backbone + **~50 GB hot experts** | ~64% |
| mtwo (worker) | 32 GB | ~28 GB | **~27 GB experts** (no backbone, no KV) | ~35% |
| **combined expert cache** | | | **~77.9 GB of 77.9** | **100%** |

**Invariant:** the union of the two RAM caches holds *every* expert. A
coordinator miss is therefore *always* a worker hit — never an SSD read. The SSD
is used only as the backing store for background LRU swaps (§6).

### Prerequisite: raise the Metal wired limit

`iogpu.wired_limit_mb` is `0` (default ≈ 2/3 RAM) on both machines today, which
caps mlock at ~42 GB (mone) / ~21 GB (mtwo) — too small. Set on each:

```sh
# mone
sudo sysctl iogpu.wired_limit_mb=58368   # ~57 GB, leaves ~7 GB for the OS
# mtwo
sudo sysctl iogpu.wired_limit_mb=28672   # ~28 GB, leaves ~4 GB for the OS
```

Keep the current memory strategy: **mlock the expert cache, mmap the backbone**
(the mmap'd backbone stays resident because it is read every token). Stream only
`token_embd`.

---

## 3. Architecture

```
                mone  (coordinator)                       mtwo (worker)
   ┌───────────────────────────────────────┐     ┌──────────────────────────┐
   │ embeddings (SSD-streamed)              │     │  expert compute server   │
   │ attention (all 43 layers)  ── Q8       │     │  - NO attention / KV     │
   │ router + shared expert                 │     │  - NO backbone           │
   │ residual stream + KV cache             │     │  - mlock'd expert cache  │
   │ HOT expert LRU cache (~50 GB, mlock)   │     │    (~27 GB, cold experts)│
   │ residency table (which experts local)  │     │  - full model on SSD     │
   │ full model on SSD (swap backing)       │     │    (swap backing)        │
   └───────────────────────────────────────┘     └──────────────────────────┘
             │            ▲                                  ▲          │
             │  8 KB hidden vec + expert ids/weights (miss)  │          │
             └───────────────────────────────────────────────┘          │
                        Thunderbolt 4 / bridge0  ◄── 8 KB result ────────┘
                        plain TCP (TCP_NODELAY, jumbo frames) — no RDMA
```

Per decode token, for each of the 43 layers the coordinator:

1. Computes attention + router locally → 6 chosen `(expert_id, weight)` pairs.
2. **Residency-table lookup** splits the 6 into *local* (in LRU) and *remote*.
3. For remote experts (≈ 0.15 × 6 ≈ ~1 per layer): fires an async request to the
   worker carrying the post-attention **normalized hidden vector (8 KB)** plus
   the remote `expert_id`s and their routing weights.
4. Computes its *local* experts on-GPU meanwhile (latency hiding, §7).
5. Receives the worker's combined partial (8 KB), adds local + remote expert
   outputs into the residual, proceeds to layer *l+1*.

The worker runs a tight loop: receive `(layer, hidden_vec, expert_ids, weights)`
→ compute those experts → return the weighted sum. It holds no attention, no KV,
no residual state. This is a **new, lighter worker role** than today's lockstep
TP worker (§4).

---

## 4. Reuse map — build on SSD-streaming, not TP

The reusable core is the **LRU expert-cache discipline** proven by the
SSD-streaming cache — not the streaming cache *instance*. The offload feature
owns a **separate** cache (`g_offload_expert_cache_*`), mlock'd and RAM-resident,
that follows the same LRU policy (evict coldest, promote touched). The `ds4_tp`
(tensor-parallel / RDMA / M5) module is deliberately **not** used — its lockstep
full-graph model and RDMA slab are the wrong shape for M1/M2. The transport is a
**new, small, plain-TCP module** (`ds4_offload.c`).

| Need | Existing code | Reuse / change |
|---|---|---|
| Dynamic LRU expert cache | `g_stream_expert_cache_*` clock-LRU in `ds4_metal.m`; `split_resident` / `split_missing` per-layer split; hotness decay | **Reuse the LRU *policy*, in a separate cache.** New `g_offload_expert_cache_*` (mlock'd slabs, §5.1); the `split_missing` branch requests the expert from the **peer**, not SSD |
| Per-expert skip in shader | the per-expert ownership predicate used by the MoE kernels in `metal/moe.metal` | Reuse the *mechanism* as a **residency check** ("is this expert resident locally?") — a generic bitmap test, not TP ownership |
| Cache sizing / mlock | `ds4_ssd.c` (`ds4_ssd_auto_cache_plan`, `ds4_ssd_memory_lock_acquire`) | Extend to the asymmetric two-node budget in §2 |
| Warm-start hot set | `ds4_streaming_hotlist.inc` (+ profiler in `ds4.c:1304`) | Seed the initial mone/mtwo partition from the precomputed hotlist |
| Coordinator/worker CLI plumbing | `ds4_dist_*` (horizontal-split distributed mode, TCP) | Borrow the option-parsing / role bring-up shape; **not** its layer-shard semantics |
| Transport | — (new) | **New `ds4_offload.c`**: one persistent `TCP_NODELAY` socket over `bridge0`, length-prefixed frames (§8). Socket/`getaddrinfo` boilerplate can be lifted from `ds4_dist`/`ds4_tp`, but the protocol is new |

**Why this is not TP:** tensor parallelism runs both ranks in lockstep over the
*full* graph (attention replicated, experts split 50/50, every layer exchanged,
RDMA slab). Here the worker runs **no graph at all** — it is a stateless
expert-compute server that answers per-layer requests only when the coordinator
misses (~27 of 43 layers), holds no backbone/KV, and speaks plain TCP. It is a
**new engine mode** whose only shared DNA with TP is "two Macs over Thunderbolt."

---

## 5. Component design

### 5.1 Residency table & dynamic ownership

- Authoritative state on the coordinator: a bitmap `resident[layer][expert]`
  (43 × 256 = 11 008 bits = 1.4 KB) marking experts held in mone's LRU. This is
  the same information the streaming cache's resident-slot table already tracks
  per layer.
- The MoE kernels already skip experts they do not hold; feed them the residency
  bitmap (a per-layer residency buffer in the kernel args) instead of the static
  ownership test. The coordinator computes resident experts; the missing slice is
  handled by §5.2.
- Lookup is O(6) per layer, branch-free; no measurable cost.

### 5.2 Per-token decode dataflow (miss path)

The existing `split_resident` / `split_missing` machinery already partitions each
layer's active experts. The change is purely in the *missing* branch:

```
today:   missing experts → pick LRU slot → pread 7 MB from SSD → GPU compute
new:     missing experts → ds4_offload_request(layer, hidden, ids, weights)
                         → worker computes → recv 8 KB partial → accumulate
```

The coordinator still runs the *resident* experts on its own GPU exactly as
today. Only the data source for the missing slice changes.

### 5.3 Worker (expert compute server) loop

New standalone loop in `ds4_offload_worker_run()` (no graph, no session, no KV):

```
loop:
  frame = recv()                       # EXPERT_REQ: layer, seq, ids[k], weights[k], hidden[8KB], evict_ids
  y = 0
  for (id, w) in (ids, weights):
     y += w * expert_ffn(layer, id, hidden)   # id resident in mtwo mlock cache
  send(EXPERT_RESP: seq, y[8KB])       # critical path — reply first
  # background, after the response (§6):
  for (promoted_id, evict_id) in zip(ids, evict_ids):
     load evict_id from mtwo's SSD into promoted_id's just-vacated slot
```

Worker holds no residual — it is stateless between requests except for its
mlock'd expert cache and pending swap work.

### 5.4 Bandwidth & latency budget (per token)

- Miss-incurring layers: `P(≥1 miss) = 1 − 0.85^6 ≈ 0.62` → ~27 of 43 layers.
- Wire traffic: 27 × (8 KB out + 8 KB in) ≈ **0.42 MB/token** — trivial.
- Critical path is **round-trip latency**, not bandwidth (§7).

---

## 6. LRU swap protocol (background, off critical path)

The LRU behaves **exactly as the SSD-streaming path does today: promote on the
first hit.** The only change is what a promotion/eviction *moves*.

**The swap is folded into the miss request — no separate control frame, no ACK.**
When expert **X** (owned by mtwo) is routed, it both needs computing *now* and
becomes an LRU promotion; mone's LRU victim **Y** must be demoted to mtwo. The
coordinator already knows Y at routing time (it is the current LRU tail), so it
piggybacks the eviction on the request it is sending anyway:

```
coordinator, at layer l:
  1. Pick X (missing) and the LRU victim Y it will evict for X.
  2. Send EXPERT_REQ{ compute: X, evict_hint: Y }  (Y costs +4 bytes on a frame
     already in flight — no extra round trip).
  3. IMMEDIATELY start background SSD load of X into a free slot, concurrent with
     the layer's local expert compute + the round trip.
  4. Compute local experts; receive EXPERT_RESP; accumulate; go to layer l+1.

worker, on EXPERT_REQ:
  5. Compute X (resident), send EXPERT_RESP  (critical path — nothing else first).
  6. THEN background-load Y from mtwo's OWN SSD copy into X's slot (X is being
     promoted to the coordinator, so its worker slot is exactly the room for Y).
```

Both SSD loads (coordinator←X, worker←Y) start early and run under the layer's
compute + RTT, so the swap is **covered by work already happening** — no added
latency. **Weights never cross Thunderbolt**: each side pages its newly-owned
expert from its local SSD backing copy (both SSDs hold the full model); the wire
carries only the 4-byte `evict_hint` already riding the request.

The 1-for-1 nature keeps the global invariant automatically: X leaves the
worker exactly as Y arrives (same slot), and Y leaves the coordinator exactly as
X arrives — cache occupancy is constant on both sides, and every expert stays
resident on exactly one machine.

### The accepted race, and how to shrink it for free

With no ACK there is one window: after the coordinator frees Y but before the
worker finishes loading Y, a request for Y hits neither RAM → one SSD-latency
token. This is low-probability *by construction* — Y is the LRU tail (coldest),
so it is the least likely expert to be re-requested in the ~1–2 ms it takes the
worker to page it in. **Accepted risk.**

Optional near-zero-cost mitigation (if the race ever bites): **lazy eviction**.
Keep a few spare cache slots; load X into a spare and mark Y "evict-pending"
rather than freeing it immediately. Y stays resident and *servable* until the
spare pool runs low (several layers later), by which point the worker has long
since loaded Y. If Y is requested during the grace window it is still a coordinator
hit. Cost: a handful of 7 MB slots. No ACK, no extra traffic.

**SSD-bandwidth note:** unlike today's single-machine streaming, each promotion
now drives a 7 MB read on *both* SSDs (coordinator loads X, worker loads Y). At
the target ~20 t/s with ~27 promotions/token this is a few GB/s of background
read *per machine* — within a single NVMe's budget, but worth measuring: if a
machine's SSD can't keep up, promotions lag and hit rate droops. Both SSDs are
otherwise idle during inference (point 1), so the full device bandwidth is
available for it.

Swap rate is naturally self-limiting: at >85% steady-state hit the hot set is
stable, so swaps are rare after warm-up. Seed the initial partition from
`ds4_streaming_hotlist.inc` so warm-up is short.

---

## 7. Latency hiding

The MoE residual chain is sequential, so each miss-layer adds a round trip on the
critical path. **Overlap** hides it: fire the worker request *before* computing
mone's local experts. mone's ~5 local experts/layer (~130 µs of GPU work) run
concurrently with the worker RTT. Fully hidden when `local_compute ≥ RTT`;
otherwise the uncovered remainder is the tax.

### Projected throughput

Coordinator is backbone-bound: ~7 GB backbone + ~1.55 GB hot experts ≈ 8.5 GB/token
÷ ~280 GB/s ≈ **30 ms → ~33 t/s ceiling**. Worker tax = ~27 miss-layers ×
(RTT − ~130 µs hidden):

| bridge0 RTT (real TCP `TCP_NODELAY`) | worker tax | decode | vs today's 10 t/s |
|---|---|---|---|
| ~0.4 ms (tuned: jumbo MTU + `TCP_NODELAY`) | ~7 ms | **~37 ms → ~26 t/s** | ~2.6× |
| ~1 ms (untuned) | ~23 ms | ~53 ms → ~19 t/s | ~1.9× |

Baselines to beat: mone-solo SSD-streaming **10 t/s**; horizontal split **7.5 t/s**.
The win comes from eliminating all critical-path SSD reads (whole model in RAM)
and serving the ~15% misses over Thunderbolt (~0.4 ms) instead of SSD (~ms).

Measured so far (ICMP, `bridge0`): min **0.47 ms** RTT — a real `TCP_NODELAY`
socket should beat it. **This single number picks the row above and is the first
thing to measure (Phase 0).** Do *not* use `en4` — it enumerated as `100baseTX`
(100 Mbit). `bridge0` only.

---

## 8. Wire protocol (new `ds4_offload` module, plain TCP)

One persistent `TCP_NODELAY` connection over `bridge0`. Length-prefixed frames,
little-endian, no RDMA, no slab:

```
frame = [ u32 len ][ u8 type ][ payload ]
```

| Frame `type` | Dir | Payload |
|---|---|---|
| `EXPERT_REQ` | coord → worker | `layer u16`, `seq u64`, `k u8`, `expert_ids[k] u16`, `weights[k] f32`, `hidden[n_embd] f16`, `evict_k u8`, `evict_ids[evict_k] u16` |
| `EXPERT_RESP` | worker → coord | `seq u64`, `y[n_embd] f16` (weighted sum) |
| `HELLO` | both | model id, `n_layer`, `n_embd`, quant, expert-partition hash — abort on mismatch |

The swap is carried entirely by the `evict_ids` tail of `EXPERT_REQ` (§6) — there
are **no** `OWNERSHIP` / `ACK` frames. The worker computes `expert_ids`, replies,
then loads each `evict_ids[i]` into the slot just vacated by the corresponding
promoted expert. Normally `evict_k == k` (1-for-1 swap); it may be 0 when the
coordinator's cache still has free slots (warm-up) and nothing needs evicting.

`n_embd = 4096` → hidden/result payloads are **8 KB f16** each. Send partials as
**f16** (not f32): halves wire bytes, and the accumulation domain is already IQ2.
Because everything is one small TCP stream, set a large `SO_SNDBUF`/`SO_RCVBUF`,
`TCP_NODELAY`, and (once jumbo is enabled) rely on `mtu 9000` so an 8 KB payload
is ~1 segment instead of ~6. The coordinator issues `EXPERT_REQ` non-blocking and
collects `EXPERT_RESP` by `seq` so per-layer requests can overlap local compute.

---

## 9. Implementation roadmap

> **Implementation status (branch `expert-offload`).** Transport, worker loop,
> residency table, offload-cache compute (Unit #2, cos=1.0), **and the
> coordinator decode splice (Unit C)** are landed and validated on `mone`.
> The splice is a pass-through wrapper `metal_graph_routed_moe_or_offload`
> (ds4.c) around the two single-node streaming routed calls: offload inactive =
> byte-for-byte the original (all other regimes untouched); active = split the
> layer's experts local/remote, ship the 8 KB hidden + remote ids/weights to the
> worker, compute the resident experts locally with the remote weights zeroed,
> and add the worker's f16 partial back with an explicit queued GPU add (the
> routed kernel's own `add_in` addend is inert on this path — a subtle bug found
> and fixed during validation). Correctness confirmed single-process via a
> loopback mode (`DS4_OFFLOAD_LOOPBACK`, remote experts computed through the
> local offload cache): under heavy offload (~64% remote-eligible, ~600 splice
> firings over 12 tokens) greedy decode is **token-identical** to a solo run and
> logits match within f16 wire noise (cos 0.99992, max |Δlogit| ~0.5). The only
> unproven piece is the TCP hop itself (pure network, no Metal) + the worker's
> cache compute (already cos=1.0) — i.e. the two-machine run. New files:
> `ds4_offload.{c,h}`, `ds4_offload_ping.c` (`./ds4-offload-ping`). Wired into
> every backend's `CORE_OBJS`.

**Phase 0 — Measure & de-risk (0.5 day).**
- [x] Real TCP `TCP_NODELAY` ping-pong bouncing an 8 KB buffer over `bridge0`
      (coordinator on mone, listener on mtwo); record p50/p99 RTT.
      → `./ds4-offload-ping --serve` (mtwo) / `--host <ip>` (mone); reports
      min/p50/p99/max and maps p50 onto the §7 go/no-go row. Loopback smoke test
      passes (p50 0.024 ms, 8 KB payload).
- [ ] Raise `iogpu.wired_limit_mb` on both; confirm ~50 GB / ~27 GB mlock without
      paging (watch `vm_stat` compressor + swapins). *(operational; §2 has the values)*
- [ ] Instrument the current router to log per-layer expert IDs; replay an LRU of
      mone's real capacity to confirm >85% hit and quantify swap rate.
- **Go/no-go gate:** p50 RTT and confirmed fit.

**Phase 1 — Static asymmetric split, miss-driven (2–3 days).**
- [x] New `ds4_offload.c` transport (persistent `TCP_NODELAY` socket, frames §8).
- [x] Worker expert-server loop (no attention/KV/graph): `ds4_offload_worker_run`,
      one coordinator at a time, replies before background evict-load (§5.3, §6).
- [x] Worker expert compute (Unit #2): `ds4_gpu_offload_cache_run_layer` reads
      the mlock'd offload cache via the slots6 kernels (cos=1.0 vs mmap). The
      worker serves *only* from the cache (the mmap path is invalid under
      streaming, H2); a miss is an ERROR status, never silent garbage (H1).
- [x] Residency table + O(k) local/remote split (`ds4_offload_residency_*`, §5.1).
- [x] New engine mode: `--expert-offload` (coord) / `--expert-server` (worker) —
      CLI wired, HELLO carries the partition id, mutually exclusive, in `--help`.
- [x] Partition experts, RAM-first (hotlist-ranked): every (layer,expert) is
      ranked by the flash hotlist (hottest first; unranked experts appended
      deterministically so all 11008 get a rank). The coordinator caches the
      hottest `coord_cap`, the worker the next `worker_cap`, and only the cold
      tail beyond both caches streams from SSD. **Both caches auto-size to the
      machine's wired budget by default** (`offload_compute_capacities`); the
      peer's capacity is read from env (`DS4_OFFLOAD_COORD_EXPERTS` /
      `DS4_OFFLOAD_CACHE_EXPERTS`) and HELLO's partition id aborts on a mismatch.
      Automatic capacity negotiation at HELLO (dropping the peer-env) is the
      remaining nicety.
- [x] Feed the residency to the live decode: `metal_graph_routed_moe_or_offload`
      splits the routed experts and computes only the resident ones locally
      (remote weights zeroed).
- [x] Redirect the missing slice from SSD to the worker: the wrapper issues
      `EXPERT_REQ` for the remote experts and adds `EXPERT_RESP` back into the
      routed output (explicit GPU add — the kernel `add_in` addend is inert here).
- [~] Overlap: request is issued then collected around the *local* compute, but
      the current wrapper collects before the routed kernel runs (add-back needs
      the partial), so RTT is not yet hidden. Overlap is the next perf step.
- **Exit:** correct logits — **DONE** (loopback: token-identical to solo, cos
      0.99992). Two-machine `≥ ~18 t/s` is the remaining hardware measurement.

**Phase 2 — Dynamic LRU swaps (2 days).**
- [ ] Promote-on-first-hit wired to background SSD paging; carry the LRU victim as
      `evict_hint` on `EXPERT_REQ` (no separate frame, no ACK — §6).
- [ ] Coordinator loads X on request-issue; worker loads `evict_ids` after replying.
- [ ] Swap-rate + hit-rate telemetry (extend `g_stream_expert_cache_*`); confirm
      per-machine background SSD read stays within device budget.
- [ ] (If the Y-race bites) add lazy eviction with a small spare-slot pool (§6).
- **Exit:** hit rate holds >85% across topic shifts; no critical-path SSD reads.

**Phase 3 — Robustness.**
- [ ] Worker-drop fallback to solo SSD-streaming (graceful degrade).
- [ ] Jumbo-frame (`mtu 9000`) setup doc + auto-detect on `bridge0`.
- [ ] Socket tuning (`SO_SNDBUF`/`SO_RCVBUF`, `TCP_NODELAY`) verified under load.

---

## 10. Validation

- **Correctness:** per-token hidden-state hash vs single-machine reference (a
  small offload-side check; do not pull in the TP lockstep machinery); eval suite
  (`ds4-eval`) parity within IQ2 noise.
- **Perf:** `ds4-bench` decode t/s at RTT {measured, tuned}; report worker tax,
  hit rate, swap rate, wire MB/token.
- **Memory:** confirm zero decode-time SSD reads (`g_stream_expert_cache_pread_*`
  must stay flat during steady-state decode; all pread activity attributable to
  background swaps only).

---

## 11. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Real RTT ≫ 0.47 ms | Worker tax dominates, < 15 t/s | Phase 0 gate; jumbo MTU; `TCP_NODELAY`; per-token overlap (§7) |
| Wired-limit pressure / OS swap | Thrash, stalls | Conservative ceilings (§2); stream `token_embd`; monitor compressor |
| Swap thrash on topic shift | Wire/SSD churn | Warm-start from hotlist; swaps are background; hit rate self-stabilizes |
| Y-race after ACK-less swap | Rare slow token (Y = LRU tail, unlikely re-request) | Accepted; optional lazy eviction w/ spare slots (§6) |
| Both SSDs page per promotion | Background read pressure | Both SSDs idle during inference; measure vs device budget (§6) |
| TCP latency variance / Nagle | Jittery tax | `TCP_NODELAY`, one persistent socket, no per-request connect |
| Worker failure mid-run | Hang | Phase 3 fallback to solo streaming |

---

## 12. Open questions

1. Exact `token_embd` streamable size vs `output.weight` (need ~1 GB buyback to
   fit — confirm the split of the 1.62 GB embed/output group).
2. Are attention + shared expert (Q8, ~7 GB) read in full every decode token, or
   is there existing sparsity to exploit? (Sets the ~33 t/s ceiling.)
3. Real `TCP_NODELAY` p50/p99 RTT over `bridge0` at MTU 1500 vs 9000 (Phase 0).
4. Best f16 vs f32 choice for wire partials given IQ2 accumulation error.
</content>
</invoke>
