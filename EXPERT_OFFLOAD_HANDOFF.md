# Expert Offload — Model-Loop Handoff

Companion to `DISTRIBUTED_EXPERT_OFFLOAD_PLAN.md`. This is the concrete
implementation guide for the parts that must be built **against the loaded model
with a connected worker** — where correctness needs a hidden-state-hash check
and the latency-hiding needs a live round trip to measure. Everything below
references real symbols/line numbers in this tree (branch `expert-offload`).

## What is already in place (built, OOM-safe)

- **Transport** — `ds4_offload.{c,h}`: `HELLO`/`EXPERT_REQ`/`EXPERT_RESP`,
  `ds4_offload_client_{connect,request,collect}`, `ds4_offload_worker_run`,
  residency bitmap (`ds4_offload_residency_*`). Unit-tested off-hardware.
- **Phase-0 gate** — `ds4-offload-ping` (`--serve` / `--host`).
- **Worker CLI** — `ds4 --expert-server` (+ `--expert-offload-bind/-port`) loads
  the model and runs the compute server.
- **Worker compute (Unit B, unverified)** — `ds4_engine_offload_compute_experts`
  (ds4.c) → `ds4_gpu_offload_run_layer` (ds4_metal.m), reusing
  `ds4_gpu_routed_moe_one_tensor`.

## Step 0 — OOM-safe run recipe (single machine, two processes)

Both processes `mmap` the model; **never mlock** the production cache on one
box. Use SSD-streaming with a small cache.

```sh
# terminal A — worker (expert-compute server)
./ds4 --expert-server --expert-offload-port 47300 \
      -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 512

# terminal B — coordinator (once Unit C lands)
./ds4 --expert-offload 127.0.0.1 --expert-offload-port 47300 \
      -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 512 \
      -p "..." -n 32
```

Watch `vm_stat` (compressor/swapins) the whole time. If pressure rises, drop
`--ssd-streaming-cache-experts`.

## Step 1 — Verify Unit B before building on it

Unit B has two unknowns baked in as assumptions; resolve both with a
**per-token hidden-state hash** (plan §10) comparing offloaded vs. solo:

1. **`x` dtype.** `ds4_gpu_offload_run_layer` writes the hidden as **f32**
   (matching the routed scratch tensors and the decode `x = ffn_norm`). If the
   forward actually expects f16 `x`, results are garbage — switch the write to
   f16 and re-test.
2. **Command-buffer lifecycle.** We rely on `ds4_gpu_routed_moe_one_tensor`
   committing+waiting on its `had_batch==false` path, then read the shared `out`.
   Confirm the read observes completed work (add an explicit
   `ds4_gpu_signal_batch_and_wait_event` if not).

**Test harness idea:** add a hidden CLI/self-test that, for a fixed hidden
vector + expert set, computes the weighted expert sum two ways — (a) normal
decode's routed forward, (b) `ds4_engine_offload_compute_experts` — and asserts
they match within IQ2 noise. This needs the model but is single-process (lower
OOM risk than two processes).

## Step 2 — Unit C: coordinator decode splice (flag-gated)

**Splice point:** the routed-MoE call in the decode path,
`ds4.c:~23824` (`ds4_gpu_routed_moe_one_tensor(...)` inside the per-layer FFN),
and its siblings at 23009/23090/23271/23477/23569. Gate everything on a new
`engine->offload_active` (set when `--expert-offload <host>` parsed).

**Coordinator client bring-up.** In `ds4_cli.c main()`, when
`cfg.offload.host` is set: build HELLO via `cli_offload_hello(engine)`, call
`ds4_offload_client_connect(host, port, &hello, timeout, ...)`, stash the client
on the engine (new field). Close it on shutdown. Fall back to solo streaming if
connect fails (Phase 3 graceful degrade).

**Residency table.** Build `ds4_offload_residency` from the partition: seed the
hottest ~64% of experts/layer (from `ds4_streaming_hotlist.inc`) as resident on
the coordinator, the rest remote. This is the *same* set the coordinator's LRU
holds; keep the bitmap in sync with the streaming cache's resident-slot table
(`g_stream_expert_cache_*`). The `partition_hash` in HELLO must cover this
partition so both sides agree.

**Per-layer dataflow (matches plan §3, §5.2):**

1. Router picks 6 `(expert_id, weight)`.
2. `ds4_offload_residency_split` → `local_idx` / `remote_idx`.
3. **Issue remote request first** (overlap): grab the normalized hidden
   (`metal_graph_ffn_norm(g)` → read to f16), then
   `ds4_offload_client_request(client, il, remote_ids, remote_weights,
   n_remote, hidden_f16, evict_ids, evict_k, ...)`. `evict_*` stays empty until
   Phase 2.
4. **Compute local experts on-GPU meanwhile.** Reuse `ds4_gpu_routed_moe_one_tensor`
   but zero the weights of the remote slots (same padding trick as Unit B) so
   only the resident experts contribute. (All experts are readable from the
   coordinator's mmap, so this is *correct* even for a cold expert — offload is
   the perf win, not a correctness requirement.)
5. `ds4_offload_client_collect(client, seq, remote_partial_f16, ...)`; add the
   f16 partial into the routed output before the shared-expert/HC combine.

**Correctness invariant:** with the worker returning the exact experts the
coordinator would otherwise compute locally, `offloaded_result ==
solo_result` per token (within IQ2 noise). This is the acceptance test — a
per-token hidden-state hash vs. a solo `-m ... ` run, no offload.

**Overlap measurement:** with correctness locked, compare decode t/s at the
measured `bridge0` RTT (Step via `ds4-offload-ping`) against the plan's §7 rows.
Report worker tax, hit rate, wire MB/token (plan §10 Perf).

## Step 3 — Partition / hotlist (Phase 1 tail)

Seed the mone/mtwo split from `ds4_streaming_hotlist.inc` (+ the profiler at
`ds4.c:~1304`). Hottest ~64% → coordinator resident, rest → worker. Feed the
same set into both the residency bitmap and the streaming cache warm-start
(`ds4_gpu_stream_expert_cache_seed_*` in ds4_gpu.h).

## Step 4 — Phase 2 dynamic LRU swaps

Piggyback the LRU victim on `EXPERT_REQ` (`evict_ids`, already in the wire
format). Coordinator: on a miss for X, pick LRU victim Y, send
`evict_ids=[Y]`, start background SSD load of X into Y's freed slot. Worker:
after replying, `opt.evict(...)` pages each `evict_id` from its own SSD copy
into the just-vacated promoted slot (wire the `ds4_offload_expert_evict_fn` —
currently NULL in `run_offload_worker`). No ACK; accept the rare Y-race, or add
lazy eviction with a small spare-slot pool (plan §6). Extend
`g_stream_expert_cache_*` telemetry for swap/hit rates.

## Step 5 — Phase 3 (robustness)

- Worker-drop → solo streaming fallback; jumbo-frame (`mtu 9000`) setup +
  autodetect on `bridge0`; verify `SO_SNDBUF`/`SO_RCVBUF`/`TCP_NODELAY` under
  load (already set in `off_socket_tune`).

## Key symbols / locations

| Thing | Where |
|---|---|
| Per-layer routed forward (reuse) | `ds4_gpu_routed_moe_one_tensor` — ds4_metal.m:34622, decl ds4_gpu.h:2309 |
| Decode splice site | ds4.c:23824 (+ 23009/23090/23271/23477/23569) |
| Per-layer expert offsets/types | `e->weights.layer[il].ffn_{gate,up,down}_exps` (`abs_offset`,`type`,`dim[]`) |
| Row/expert bytes | `routed_expert_row_bytes()`, `dim[1]*row_bytes` (ds4.c:4408) |
| Routed scratch sizes | ds4.c:17036 (`N_EXPERT_USED*mid_dim`, down `N_EXPERT_USED*N_EMBD`) |
| Normalized hidden to send | `metal_graph_ffn_norm(g)` |
| Selected-expert override | `ds4_gpu_routed_moe_set_selected_override` (ds4_metal.m:34612) |
| Streaming cache state | `g_stream_expert_cache_*` (ds4_metal.m) |
| Worker evict hook | `ds4_offload_expert_evict_fn` (ds4_offload.h), set in `run_offload_worker` |
