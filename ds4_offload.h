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
 *   - the wire protocol (HELLO / EXPERT_REQ / EXPERT_RESP), §8;
 *   - the coordinator client (connect, request, seq-keyed response collection);
 *   - the worker server loop (§5.3) driven by a registered expert-compute callback;
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

/* Wire frame = [ u32 len ][ u8 type ][ payload ] (§8). `len` counts type+payload. */
typedef enum {
    DS4_OFFLOAD_FRAME_HELLO       = 1,  /* both directions, once, at connect  */
    DS4_OFFLOAD_FRAME_EXPERT_REQ  = 2,  /* coordinator -> worker              */
    DS4_OFFLOAD_FRAME_EXPERT_RESP = 3,  /* worker -> coordinator              */
    DS4_OFFLOAD_FRAME_BYE         = 4,  /* graceful shutdown                  */
    DS4_OFFLOAD_FRAME_PING        = 5,  /* Phase-0 benchmark only             */
    DS4_OFFLOAD_FRAME_PONG        = 6,  /* Phase-0 benchmark only             */
} ds4_offload_frame_type;

/* HELLO payload: identity both sides must agree on before any EXPERT_REQ. */
typedef struct {
    uint32_t magic;             /* DS4_OFFLOAD_MAGIC                          */
    uint32_t version;           /* protocol version                          */
    uint16_t n_layer;           /* must equal DS4_OFFLOAD_N_LAYER            */
    uint16_t n_used;            /* must equal DS4_OFFLOAD_N_USED             */
    uint32_t n_embd;            /* must equal DS4_OFFLOAD_N_EMBD             */
    uint32_t n_routed;          /* must equal DS4_OFFLOAD_N_ROUTED          */
    uint64_t partition_hash;    /* hash of the expert partition (§8)         */
    uint64_t model_id;          /* GGUF identity hash                        */
} ds4_offload_hello;

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

/* Dial the worker and complete the HELLO handshake. `hello` carries this side's
 * identity; the call fails if the peer's HELLO disagrees. */
ds4_offload_client *ds4_offload_client_connect(const char *host, int port,
                                               const ds4_offload_hello *hello,
                                               double timeout_sec,
                                               char *err, size_t errlen);
void ds4_offload_client_close(ds4_offload_client *c);

/* Issue one EXPERT_REQ (§8): compute `expert_ids[k]` weighted by `weights[k]`
 * on `hidden` (n_embd f16), demoting `evict_ids[evict_k]` from the coordinator
 * (piggybacked LRU eviction hint, §6). Non-blocking send; returns the seq to
 * collect the response by, or 0 on error. */
uint64_t ds4_offload_client_request(ds4_offload_client *c,
                                    int layer,
                                    const uint16_t *expert_ids,
                                    const float *weights, int k,
                                    const uint16_t *hidden_f16,
                                    const uint16_t *evict_ids, int evict_k,
                                    char *err, size_t errlen);

/* Collect the EXPERT_RESP for `seq` (blocks until it arrives), writing the
 * n_embd f16 weighted sum into `out_f16`. Responses may arrive out of order;
 * the client buffers by seq so per-layer requests can overlap local compute. */
int ds4_offload_client_collect(ds4_offload_client *c, uint64_t seq,
                               uint16_t *out_f16, char *err, size_t errlen);

/* ------------------------------------------------------------------------
 * Worker (expert compute server), §5.3. Holds no attention / KV / residual.
 * --------------------------------------------------------------------- */

/* Compute the weighted sum of `k` experts of `layer` applied to `hidden`
 * (n_embd f16) into `out` (n_embd f16). Registered by the Metal engine; the
 * worker loop calls it on the critical path, then performs the background
 * evict-load of `evict_ids` (§6) after replying. Returns 0 on success.
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

/* Background hook run after the response is sent: page `evict_ids[evict_k]`
 * into the slots just vacated by the promoted experts, from the worker's own
 * SSD copy (§6). May be NULL (Phase 1: static split, no swaps). */
typedef void (*ds4_offload_expert_evict_fn)(void *user,
                                            int layer,
                                            const uint16_t *evict_ids,
                                            int evict_k);

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
