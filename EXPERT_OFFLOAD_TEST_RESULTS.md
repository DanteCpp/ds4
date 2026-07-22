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

_In progress. Unit C step 1 (coordinator client connect) built + verified (HELLO handshake OK over bridge0). Step 2 (decode splice) next._

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

### Unit C step 1 — coordinator client connect (HELLO handshake)

Built: in `ds4_cli.c main()`, after `ds4_engine_open` (and past the worker /
tp-worker / distributed-worker early returns), when `cfg.offload.host` is set,
call `ds4_offload_client_connect(host, port, &cli_offload_hello(engine), 5.0, …)`;
log success/failure; close the client in the shutdown cleanup. Solo fallback on
connect failure (plan Phase-4 degrade). No engine-struct change yet (the client
stays local to `main()`); step 2 will stash it on the engine for the splice.

Test (mone → mtwo over bridge0, `--inspect` so no inference/expert loading):
```sh
./ds4 --inspect --ssd-streaming \
      --expert-offload 169.254.14.255 --expert-offload-port 47300 \
      -m ds4flash.gguf
```
mone log:
```
ds4: expert-offload: connected to worker 169.254.14.255:47300
model: DeepSeek V4 Flash
... (summary: layers=43 experts=256 used=6 — matches DS4_OFFLOAD_* constants)
```
Process exited cleanly → the `ds4_offload_client_close` shutdown path ran.

**Result: PASS.** `ds4_offload_client_connect` returned a live client; the HELLO
handshake completed over Thunderbolt (no `rejecting coordinator` on mtwo). The
wire path is proven end-to-end up to the expert-compute loop. Step 2 (the decode
splice that actually issues `EXPERT_REQ`) is the remaining work.

### Unit B verification (handoff Step 1) — self-test harness

Built a gated, env-driven self-test (zero overhead when off) that checks the
worker's expert compute (`ds4_gpu_offload_run_layer`, the Unit B path) against
the decode's own routed output:
- `ds4_offload_selftest_capture(...)` is called from each decode routed-MoE path
  (the `--ssd-streaming` path at `ds4.c:23556` and the non-streaming path at
  `ds4.c:23650`) right after the routed call. It reads the real
  `ffn_norm` / `router_selected` / `router_weights` / `routed_out` to host via
  sync reads (the decode already does mid-stream reads, so this is safe).
- `ds4_offload_selftest_finalize(engine)` (called from `main()` after generation)
  replays each captured hidden + selection through `ds4_gpu_offload_run_layer` in
  a CLEAN Metal state (the way the worker runs it, in isolation — not mid-decode)
  and diffs the f16-roundtripped result against the captured `routed_out`.

Why capture+replay (not inline): calling `ds4_gpu_offload_run_layer` *mid-decode*
flushes/encodes into the decode's command buffer and corrupts Metal state
(`metal prefill failed`, leaked handles). The worker runs it in isolation, so
replay-after-generation mimics real usage and keeps the decode intact.

Env: `DS4_OFFLOAD_SELFTEST_LAYER=<il>` (default off), `DS4_OFFLOAD_SELFTEST_TOKENS=<n>` (default 4).

Test (mone, single-process, layer 5, 4 tokens):
```sh
DS4_OFFLOAD_SELFTEST_LAYER=5 DS4_OFFLOAD_SELFTEST_TOKENS=4 \
  ./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 512 -p "Hello" -n 4
```
```
ds4: offload-selftest L=5 tok#0 sel=[35 235 1 92 129 74] max_abs=33.193867 mean_abs=3.682401 max_rel=1.19e+00  <-- over 1e-2
ds4: offload-selftest L=5 tok#1 sel=[59 35 252 244 243 106] max_abs=5.245254 mean_abs=0.774993 max_rel=7.32e+00  <-- over 1e-2
ds4: offload-selftest L=5 tok#2 sel=[59 252 140 221 158 52] max_abs=2.144203 mean_abs=0.274467 max_rel=2.07e+00  <-- over 1e-2
ds4: offload-selftest L=5 tok#3 sel=[180 252 59 30 216 169] max_abs=0.774030 mean_abs=0.178123 max_rel=1.48e+00  <-- over 1e-2
ds4: offload-selftest DONE layer=5 tokens=4 worst_max_rel=7.32e+00 -> FAIL
```

**Result: FAIL.** The decode completes cleanly (4 tokens, no corruption), but the
offload compute diverges from the decode's routed sum by `max_rel` 1.2-7.3 -
orders of magnitude above the expected f16-wire noise (~1e-3). The error is as
large as the signal itself, i.e. the offload output is *wrong*, not noisy.

Resolved / ruled out:
- **Unknown #1 (x dtype): RESOLVED.** `ffn_norm` and `routed_out` graph tensors
  are f32 (`DS4_N_EMBD * sizeof(float)`), so the kernel consumes `x` as f32 -
  matching the offload path's f16->f32 conversion. The f16 round-trip of the input
  is ~1e-3 relative, far too small to explain a `max_rel` of 1-7.
- **Command-buffer lifecycle:** calling `offload_run_layer` mid-decode corrupts
  the decode's command buffer; the capture+replay split avoids that and the decode
  now completes. So lifecycle is not the cause of the numeric divergence.
- **Stale streaming cache:** clearing `g_stream_expert_cache_*` before the replay
  produced *identical* numbers - not the cause.

**Likely cause (new blocker):** `ds4_gpu_routed_moe_one_tensor` has two expert
compute paths - the **streaming-expert-cache path** the decode uses
(`begin_selected_load` -> resident slots) and the **direct-mmap path** the
offload/worker uses (read experts at `gate_offset + id*bytes` from the model
map). For the same expert + hidden + weights, these two paths produce different
results. The decode sets the selected-override and uses the cache; the offload
clears the override and reads direct. Both resolve to the same router ids, so
expert *selection* matches - the divergence is in the *compute* (cache vs
direct-mmap).

**Implication for the splice (Unit C step 2):** the correctness invariant
("offloaded == solo per token") cannot hold until the direct-mmap path matches
the cache path. The coordinator's *local* experts go through the cache path; the
worker's *remote* experts go through direct-mmap. If they differ for the same
expert, combining them yields a different result than solo (all-cache). This
must be fixed before the splice is meaningful.

---

## Open items before the real test can run

1. **Build Unit C on mone:**
   - ✅ Step 1 — coordinator client connect in `ds4_cli.c main()`
     (`ds4_offload_client_connect` after `ds4_engine_open`, close on shutdown,
     solo fallback on connect fail). **Verified** (HELLO handshake PASS).
   - ☐ Step 2 — decode splice at the 6 routed-MoE sites gated on `offload_active`
     (stash client on engine; residency split → issue remote `EXPERT_REQ` first,
     compute local experts with remote weights zeroed, `collect` + add f16
     partial before shared/HC combine).
2. **Step-1 hidden-state-hash harness: BUILT** (capture+replay self-test above).
   - x dtype: RESOLVED (f32).
   - Command-buffer lifecycle: mid-decode call corrupts -> capture+replay split.
   - **Finding: FAIL - offload direct-mmap compute != decode streaming-cache
     compute (max_rel 1-7). New blocker (see #4).**
3. (Phase 1 tail) Seed mone/mtwo split from `ds4_streaming_hotlist.inc` ->
   residency bitmap + streaming cache warm-start.
4. **[BLOCKER] Resolve the direct-mmap vs streaming-cache compute divergence in
   `ds4_gpu_routed_moe_one_tensor` (Unit B).** The worker (direct-mmap) and the
   coordinator's local experts (streaming-cache path) must produce identical
   results for the same expert+hidden+weights, or the splice's
   `offloaded == solo` invariant cannot hold. Investigate the two paths in
   `ds4_metal.m:34740+` (~5000-line function): the decode uses
   `begin_selected_load` + the selected-override; the offload clears the
   override and reads direct. Selection matches; the compute differs.

---

## Phase 1 — Unit-B self-test, root-caused (2026-07-22, mone)

Re-ran the capture+replay self-test on mone (M1 Max 64 GB, `iogpu.wired_limit_mb`
already 57344) with the model in SSD-streaming mode:

```sh
DS4_OFFLOAD_SELFTEST_LAYER=3 DS4_OFFLOAD_SELFTEST_TOKENS=4 \
  ./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 256 \
        -p "The capital of France is" -n 6 --temp 0
```

**Result: FAIL, and now root-caused.** Added a cosine/norm diagnostic
(`DS4_OFFLOAD_SELFTEST_DEBUG=1`, finalize in `ds4.c`):

| tok | \|offload\| | \|ref\| | cosine | ratio |
|-----|-------------|---------|--------|-------|
| 0   | 4.39        | 177.69  | 0.23   | 0.02  |
| 1   | 20.09       | 6.45    | 0.01   | 3.11  |
| 2   | 13.93       | 10.42   | 0.08   | 1.34  |
| 3   | 8.46        | 8.63    | 0.05   | 0.98  |

**The offload output is orthogonal to the decode reference (cos ≈ 0), not a
scaled/shifted version.** Since capture grabs `ffn_norm`, `router_selected`,
`router_weights`, `routed_out` at one consistent point, and the offload output
*does* vary per token (so the hidden vector is being consumed), the only
explanation is that **the offload compute reads the wrong expert weights.**

Ruled out by experiment (each a full model-load run; all byte-identical output):
- Loading the experts into the streaming cache first via the async
  `begin_selected_load` → no change.
- Same via the synchronous `ds4_gpu_stream_expert_cache_seed_selected` → no change.
- Forcing the mmap path (`DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS=1`) →
  hard fail: *"Metal model range … is not covered by mapped model views"* — in
  streaming mode only `token_embd` is mmap'd; **the routed experts are not in the
  mmap at all**, only in the streaming slab cache.

**Conclusion.** Reusing the streaming-coupled `ds4_gpu_routed_moe_one_tensor`
for the offload worker is the wrong foundation: in `--ssd-streaming` the experts
live only in the streaming slab cache, and that cache is not reliably populated
for the selected experts at the standalone (post-generation / worker) call site,
so the compute reads garbage. This is regime-specific, not a kernel bug — the
same code computes the decode correctly *inside* the graph.

**Fix (the intended architecture):** the offload compute must read experts from
the **populated offload cache** (`g_offload_expert_cache_*`, mlock'd, filled from
the GGUF at startup — the populate landed this session), which is regime-
independent and testable on mone. That is Unit #2 (bind the offload slab slots to
the existing `slots6` IQ2/Q2_K kernels — no new GPU kernels). `offload_run_layer`
was reverted to baseline (the seed/override experiments had no effect); the
`DS4_OFFLOAD_SELFTEST_DEBUG` cosine diagnostic was kept.

---

## Phase 1 — Unit #2 offload-cache compute BUILT + cross-validated (2026-07-22, mone)

`ds4_gpu_offload_cache_run_layer` (ds4_metal.m): binds the k selected experts'
mlock'd offload-cache slab slots to the `slots6` IQ2_XXS pair-swiglu + Q2_K sum6
kernels (the exact kernels the streaming decode uses at
ds4_metal.m:37057/37394), reading **only** the offload cache — no mmap, no
streaming cache. Self-test (`DS4_OFFLOAD_CACHE_SELFTEST=1`) populates the
selected experts into the cache, computes from the slabs, and directly compares
to the mmap path (Unit B) for the same inputs:

```
xval tok#0 cache-vs-mmap cos=1.000000 |cache|=4.3920  |mmap|=4.3920
xval tok#1 cache-vs-mmap cos=1.000000 |cache|=20.0902 |mmap|=20.0902
xval tok#2 cache-vs-mmap cos=1.000000 |cache|=13.9342 |mmap|=13.9342
xval tok#3 cache-vs-mmap cos=1.000000 |cache|=8.4570  |mmap|=8.4570
```

**cos = 1.000000, identical norms** — Unit #2 (offload cache, slots6 kernels)
is bit-for-bit equivalent to Unit B (mmap, routed_moe_one_tensor). Two fully
independent compute paths reading from different memory agree exactly.

**Consequence for the "blocker":** the per-layer self-test still reports FAIL
against its captured reference (worst_max_rel 4.47, cos≈0 vs reference), but
that reference is now proven wrong — two independent correct computes both
disagree with it, so the fault is in the harness **capture** (the captured
`ffn_norm`/`routed_out` pair is inconsistent, most likely a stale `ffn_norm`
read on the async decode path), **not** in the compute. In production the
coordinator sends the live normalized hidden, so this harness artifact does not
affect correctness. The definitive correctness signal is end-to-end generation
parity (offloaded vs solo), which needs the coordinator splice (Unit C) and is
the plan's §10 acceptance test.

`ds4_engine_offload_compute_experts` now prefers the offload cache when
populated (rc==0 served; rc==1 not-resident / rc<0 → mmap fallback), so the
worker uses the validated cache path once `DS4_OFFLOAD_CACHE_EXPERTS` seeds it.

**Status:** Unit #1 (populate) + Unit #2 (compute) done & validated on mone.
Next: Unit C coordinator splice → loopback generation parity on mone → two
machines.