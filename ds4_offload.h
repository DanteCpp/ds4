#ifndef DS4_OFFLOAD_H
#define DS4_OFFLOAD_H

/* Distributed expert offload transport for DeepSeek-V4-Flash (Metal only).
 *
 * This is the "new, small, plain-TCP module" from DISTRIBUTED_EXPERT_OFFLOAD_PLAN.md
 * (§4, §8). It is deliberately NOT tensor parallelism: there is no RDMA, no slab,
 * no lockstep full graph. A coordinator (mone) holds the backbone plus the hottest
 * experts in its LRU cache; a worker (mtwo) is a stateless expert-compute server
 * holding the colder experts. On a coordinator cache miss the coordinator ships an
 * 8 KB activation to the worker over Thunderbolt (bridge0), the worker computes the
 * routed experts and ships back the 8 KB weighted sum. Weights never cross the wire.
 *
 * The module owns:
 *   - the wire protocol (HELLO / PLAN / PLAN_ACK / EXPERT_REQ / EXPERT_RESP), §8;
 *   - the v2 startup orchestration: the worker offers its available expert-cache
 *     memory in HELLO, the coordinator decides the RAM-first split (its own hot
 *     tier, the worker's tier, and — only when combined RAM cannot hold the
 *     model — the coldest tail streamed from the coordinator's SSD as the last
 *     resort) and ships the worker the exact expert ids to load in PLAN;
 *   - the coordinator client (connect, orchestrate, request, seq-keyed response
 *     collection);
 *   - the worker server loop (§5.3) driven by registered plan/compute callbacks;
 *   - the residency bitmap (§5.1) that splits routed experts into local vs remote;
 *   - the Phase-0 TCP TCP_NODELAY ping-pong benchmark (§9, the go/no-go RTT gate).
 *
 * The two remaining integration seams into the Metal engine (feeding the residency
 * bitmap to the live MoE kernel, and having the worker run the expert FFN on the
 * GPU) are reached through ds4_offload_expert_compute_fn and the residency API; see
 * the notes on those declarations.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Model constants (§1, verified from the GGUF). These are fixed for
 * DeepSeek-V4-Flash; the HELLO handshake aborts on any mismatch so a wrong
 * model or partition can never silently corrupt results.
 * --------------------------------------------------------------------- */
#define DS4_OFFLOAD_N_LAYER        43     /* block_count, all MoE            */
#define DS4_OFFLOAD_N_ROUTED       256    /* routed experts per layer        */
#define DS4_OFFLOAD_N_USED         6      /* expert_used_count               */
#define DS4_OFFLOAD_N_EMBD         4096   /* embedding_length                */
/* Hidden / result payloads travel as f16: 4096 * 2 = 8192 bytes (§8). */
#define DS4_OFFLOAD_HIDDEN_F16_BYTES ((size_t)DS4_OFFLOAD_N_EMBD * 2)

#define DS4_OFFLOAD_DEFAULT_PORT   47300
#define DS4_OFFLOAD_MAGIC          0x4F464C44u /* "OFLD" */

/* Protocol version. v2 adds the startup orchestration phase: the worker's
 * HELLO offers its available expert-cache memory, the coordinator's reply
 * HELLO carries the decided capacity split, and a PLAN frame then hands the
 * worker the exact expert ids to load. v1 peers abort on the version check. */
#define DS4_OFFLOAD_PROTOCOL_VERSION 2

/* EXPERT_RESP status byte (§8, H1). A worker that cannot compute the requested
 * experts (GPU error, or — the important case — the expert is not resident in
 * its cache in the streaming regime where the mmap path is invalid) replies
 * with DS4_OFFLOAD_STATUS_ERROR instead of a valid-looking zero vector, so the
 * coordinator never folds silent garbage into the residual. */
#define DS4_OFFLOAD_STATUS_OK      0u
#define DS4_OFFLOAD_STATUS_ERROR   1u

/* Wire frame = [ u32 len ][ u8 type ][ payload ] (§8). `len` counts type+payload. */
typedef enum {
    DS4_OFFLOAD_FRAME_HELLO       = 1,  /* both directions, once, at connect  */
    DS4_OFFLOAD_FRAME_EXPERT_REQ  = 2,  /* coordinator -> worker              */
    DS4_OFFLOAD_FRAME_EXPERT_RESP = 3,  /* worker -> coordinator              */
    DS4_OFFLOAD_FRAME_BYE         = 4,  /* graceful shutdown                  */
    DS4_OFFLOAD_FRAME_PING        = 5,  /* Phase-0 benchmark only             */
    DS4_OFFLOAD_FRAME_PONG        = 6,  /* Phase-0 benchmark only             */
    DS4_OFFLOAD_FRAME_PLAN        = 7,  /* coordinator -> worker, v2          */
    DS4_OFFLOAD_FRAME_PLAN_ACK    = 8,  /* worker -> coordinator, v2          */
} ds4_offload_frame_type;

/* One expert the coordinator assigns to the worker in the PLAN frame. */
typedef struct {
    uint16_t layer;
    uint16_t expert;
} ds4_offload_expert_ref;

/* Max experts a PLAN can assign: the whole routed-expert space. */
#define DS4_OFFLOAD_PLAN_MAX (DS4_OFFLOAD_N_LAYER * DS4_OFFLOAD_N_ROUTED)

/* HELLO.flags bit: the coordinator streams the coldest tail (the experts that
 * fit in neither RAM cache) from its own SSD — the last-resort regime. */
#define DS4_OFFLOAD_HELLO_FLAG_SSD_TAIL 0x1u

/* HELLO payload, v2. The exchange is asymmetric (the orchestration phase):
 *   1. worker -> coordinator: identity + `mem_avail_bytes` offer (decision
 *      fields zero; partition_hash zero — the split is not decided yet).
 *   2. coordinator -> worker: same identity, plus the DECISION: `coord_cap`
 *      hottest experts cached by the coordinator, `worker_cap` next-hotter
 *      experts assigned to the worker, `plan_count` of them enumerated in the
 *      PLAN frame that follows, flags bit SSD_TAIL when the cold tail spills
 *      to the coordinator's SSD. `partition_hash` identifies the split.
 * Identity fields (magic..model_id) must match; a mismatch aborts. */
typedef struct {
    uint32_t magic;             /* DS4_OFFLOAD_MAGIC                          */
    uint32_t version;           /* DS4_OFFLOAD_PROTOCOL_VERSION               */
    uint16_t n_layer;           /* must equal DS4_OFFLOAD_N_LAYER            */
    uint16_t n_used;            /* must equal DS4_OFFLOAD_N_USED             */
    uint32_t n_embd;            /* must equal DS4_OFFLOAD_N_EMBD             */
    uint32_t n_routed;          /* must equal DS4_OFFLOAD_N_ROUTED          */
    uint64_t partition_hash;    /* decision id (coordinator -> worker)        */
    uint64_t model_id;          /* GGUF identity hash                        */
    uint64_t mem_avail_bytes;   /* worker offer: bytes it can wire for experts*/
    uint32_t coord_cap;         /* decision: experts cached on the coordinator*/
    uint32_t worker_cap;        /* decision: experts assigned to the worker   */
    uint32_t plan_count;        /* decision: entries in the PLAN frame        */
    uint32_t flags;             /* DS4_OFFLOAD_HELLO_FLAG_*                   */
} ds4_offload_hello;

/* ------------------------------------------------------------------------
 * Session debug log (diagnostic). Same "ds4: ..." line style as the rest of
 * the codebase, written to a file with a monotonic millisecond timestamp so a
 * distributed run can be analyzed post-hoc and correlated across the two
 * machines: per layer, per token, which experts were served from local RAM,
 * from the worker over the wire (and its latency), or streamed from SSD.
 * --------------------------------------------------------------------- */
typedef struct ds4_offload_log ds4_offload_log;

/* Open the log in append mode. Returns NULL on failure (a diagnostic never
 * aborts the run). */
ds4_offload_log *ds4_offload_log_open(const char *path);
void ds4_offload_log_close(ds4_offload_log *l);
/* "ds4: [t=+<ms>] <fmt>" to the file (flushed every line) AND stderr — for
 * low-volume lifecycle lines (orchestration, token summaries, totals). */
void ds4_offload_logf(ds4_offload_log *l, const char *fmt, ...);
/* File-only variant for the high-volume per-layer / per-request stream. */
void ds4_offload_logf_file(ds4_offload_log *l, const char *fmt, ...);

/* ------------------------------------------------------------------------
 * Residency table (§5.1). Authoritative on the coordinator: which experts of
 * each layer are held locally (in mone's LRU). 43 x 256 bits = 1.4 KB.
 * --------------------------------------------------------------------- */
typedef struct {
    uint64_t bits[DS4_OFFLOAD_N_LAYER][(DS4_OFFLOAD_N_ROUTED + 63) / 64];
} ds4_offload_residency;

void ds4_offload_residency_clear(ds4_offload_residency *r);
void ds4_offload_residency_set(ds4_offload_residency *r, int layer, int expert, bool resident);
bool ds4_offload_residency_get(const ds4_offload_residency *r, int layer, int expert);

/* Split a layer's k routed (expert_id, weight) pairs into a "local" slice the
 * coordinator computes on its own GPU and a "remote" slice to send to the
 * worker. Local and remote index arrays receive positions into the input
 * arrays; returns via *n_local / *n_remote. O(k), branch-light (§5.1). */
void ds4_offload_residency_split(const ds4_offload_residency *r,
                                 int layer,
                                 const uint16_t *expert_ids, int k,
                                 uint8_t *local_idx, int *n_local,
                                 uint8_t *remote_idx, int *n_remote);

/* ------------------------------------------------------------------------
 * Coordinator client. One persistent TCP_NODELAY socket over bridge0.
 * --------------------------------------------------------------------- */
typedef struct ds4_offload_client ds4_offload_client;

/* Dial the worker and complete step 1 of the orchestration handshake: the
 * worker's HELLO (identity + `mem_avail_bytes` offer) is returned via
 * `worker_hello` (may be NULL). `hello` carries this side's identity; the
 * decision fields are ignored at this stage. Fails if the peer's identity
 * disagrees. Follow with ds4_offload_client_orchestrate before any request. */
ds4_offload_client *ds4_offload_client_connect(const char *host, int port,
                                               const ds4_offload_hello *hello,
                                               double timeout_sec,
                                               ds4_offload_hello *worker_hello,
                                               char *err, size_t errlen);
void ds4_offload_client_close(ds4_offload_client *c);

/* Step 2 of the orchestration phase: send the coordinator's decision HELLO
 * (`decision` carries coord_cap / worker_cap / plan_count / flags /
 * partition_hash — fill them from the engine's split decision), then the PLAN
 * frame listing the `count` experts the worker must load, and block until the
 * worker finishes installing them and replies PLAN_ACK (this can take minutes:
 * the worker preads + mlocks gigabytes). Returns 0 when the worker accepted
 * the plan (*installed / *wired_bytes from the ACK), -1 on transport error or
 * worker rejection. Must be called exactly once, right after connect. */
int ds4_offload_client_orchestrate(ds4_offload_client *c,
                                   const ds4_offload_hello *decision,
                                   const ds4_offload_expert_ref *plan,
                                   uint32_t count,
                                   uint32_t *installed,
                                   uint64_t *wired_bytes,
                                   char *err, size_t errlen);

/* Issue one EXPERT_REQ (§8): compute `expert_ids[k]` weighted by `weights[k]`
 * on `hidden` (n_embd f16), and carry `swap_k` coordinator-authoritative cache
 * swaps — for each, the worker evicts `swap_evict[i]` and loads `swap_load[i]`
 * (both in `layer`) into its slot (§6). The coordinator names both and mirrors
 * the change, so the caches never desynchronize. Blocking send (frames are
 * small and the socket has a multi-MB send buffer, so this rarely blocks);
 * returns the seq to collect the response by, or 0 on error. */
uint64_t ds4_offload_client_request(ds4_offload_client *c,
                                    int layer,
                                    const uint16_t *expert_ids,
                                    const float *weights, int k,
                                    const uint16_t *hidden_f16,
                                    const uint16_t *swap_evict,
                                    const uint16_t *swap_load, int swap_k,
                                    char *err, size_t errlen);

/* Collect the EXPERT_RESP for `seq` (blocks until it arrives), writing the
 * n_embd f16 weighted sum into `out_f16`. Responses are collected strictly
 * in-order on the single request stream; overlap still works because the
 * coordinator issues several requests before collecting the first. Returns 0 on
 * success, -1 on transport error, +1 if the worker replied ERROR (H1). The
 * latter two are worker-drop signals: `out_f16` is untouched, degrade to solo. */
int ds4_offload_client_collect(ds4_offload_client *c, uint64_t seq,
                               uint16_t *out_f16, char *err, size_t errlen);

/* ------------------------------------------------------------------------
 * Worker (expert compute server), §5.3. Holds no attention / KV / residual.
 * --------------------------------------------------------------------- */

/* Compute the weighted sum of `k` experts of `layer` applied to `hidden`
 * (n_embd f16) into `out` (n_embd f16). Registered by the Metal engine; the
 * worker loop calls it on the critical path, then applies any coordinator swaps
 * (evict/load pairs, §6) after replying. Returns 0 on success.
 *
 * `user` is the opaque pointer passed to ds4_offload_worker_run. When no Metal
 * backend is registered (e.g. the Phase-0 ping tool), pass the built-in
 * ds4_offload_expert_compute_zero which returns a zero vector. */
typedef int (*ds4_offload_expert_compute_fn)(void *user,
                                             int layer,
                                             const uint16_t *expert_ids,
                                             const float *weights, int k,
                                             const uint16_t *hidden_f16,
                                             uint16_t *out_f16);

/* Hook run after the response is sent (and before the next request is read):
 * apply the coordinator's `swap_k` cache swaps — for each, evict
 * `swap_evict[i]` and load `swap_load[i]` (both in `layer`) from the worker's
 * own GGUF into the evicted slot (§6). The coordinator is authoritative over
 * the worker cache and mirrors the identical change, so the two never
 * desynchronize. May be NULL (Phase 1: static split, no swaps). */
typedef void (*ds4_offload_expert_evict_fn)(void *user,
                                            int layer,
                                            const uint16_t *swap_evict,
                                            const uint16_t *swap_load,
                                            int swap_k);

/* Orchestration hook (v2): the coordinator's PLAN has arrived — install
 * exactly `plan[count]` into this worker's offload cache and adopt the
 * decided split (coord_cap is informational; worker_cap == count). Fill
 * *installed / *wired_bytes for the ACK. Returns 0 on success; on non-zero
 * the worker ACKs ERROR with `err` and the connection closes (a partial cache
 * would silently serve wrong experts, H2). May be NULL: any plan is rejected. */
typedef int (*ds4_offload_plan_fn)(void *user,
                                   uint32_t coord_cap,
                                   const ds4_offload_expert_ref *plan,
                                   uint32_t count,
                                   uint32_t *installed,
                                   uint64_t *wired_bytes,
                                   char *err, size_t errlen);

/* Optional diagnostics hook: the worker calls this each time it flushes a
 * per-token aggregate log line, so the engine can append a short status
 * string (offload-cache residency, hit/miss counters, mlock failures, ...)
 * that verifies the cache is (still) populated and serving. Write a
 * NUL-terminated string into `buf`. May be NULL (no diagnostics appended). */
typedef void (*ds4_offload_worker_diag_fn)(void *user,
                                           char *buf, size_t buflen);

int ds4_offload_expert_compute_zero(void *user, int layer,
                                    const uint16_t *expert_ids,
                                    const float *weights, int k,
                                    const uint16_t *hidden_f16,
                                    uint16_t *out_f16);

typedef struct {
    const char *bind_host;      /* NULL / "" = all interfaces                */
    int port;                   /* 0 = DS4_OFFLOAD_DEFAULT_PORT              */
    ds4_offload_hello hello;    /* worker identity for the handshake         */
    ds4_offload_expert_compute_fn compute;
    ds4_offload_expert_evict_fn evict; /* may be NULL                        */
    ds4_offload_plan_fn plan;          /* may be NULL (reject orchestration) */
    ds4_offload_worker_diag_fn diag;   /* may be NULL (per-token log suffix) */
    ds4_offload_log *log;              /* may be NULL (session debug log)     */
    void *user;
    volatile int *stop;         /* set non-zero from another thread to exit  */
} ds4_offload_worker_options;

/* Accept one coordinator, run the expert-compute loop until BYE / stop / peer
 * close, then return. Serves a single coordinator at a time by design (§3). */
int ds4_offload_worker_run(const ds4_offload_worker_options *opt,
                           char *err, size_t errlen);

/* ------------------------------------------------------------------------
 * Phase-0 RTT gate (§7, §9). Bounce an 8 KB buffer over a real TCP_NODELAY
 * socket and report p50/p99 RTT — the single number that picks the throughput
 * row and gates go/no-go. Runnable on one host via loopback for a smoke test.
 * --------------------------------------------------------------------- */
typedef struct {
    double min_ms, p50_ms, p99_ms, max_ms, mean_ms;
    uint64_t samples;
    size_t payload_bytes;
} ds4_offload_pingpong_stats;

/* Server side: accept one client and echo PING->PONG until it disconnects. */
int ds4_offload_pingpong_serve(const char *bind_host, int port,
                               size_t payload_bytes,
                               volatile int *stop,
                               char *err, size_t errlen);

/* Client side: dial, then time `iterations` round trips of `payload_bytes`. */
int ds4_offload_pingpong_run(const char *host, int port,
                             size_t payload_bytes, int iterations,
                             ds4_offload_pingpong_stats *out,
                             char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* DS4_OFFLOAD_H */
