/* Distributed expert offload transport — plain TCP over Thunderbolt (bridge0).
 * See ds4_offload.h and DISTRIBUTED_EXPERT_OFFLOAD_PLAN.md. Metal only.
 *
 * No RDMA, no slab, no lockstep: one persistent TCP_NODELAY socket, length-
 * prefixed little-endian frames. The coordinator is the client; the worker is a
 * stateless expert-compute server. */

#include "ds4_offload.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------------
 * Small helpers.
 * --------------------------------------------------------------------- */

static void off_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static double off_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Full read/write that survive short transfers and EINTR. Return 1 on success,
 * 0 on peer close or error. */
static int off_write_full(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static int off_read_full(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += r;
        len -= (size_t)r;
    }
    return 1;
}

static void off_socket_tune(int fd) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    /* TCP_NODELAY is mandatory: the critical path is a single small round trip,
     * so Nagle coalescing would add up to a full RTT of latency (§8, §11). */
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Room for a few 8 KB frames in flight so overlapped per-layer requests do
     * not stall on the socket buffer (§8). */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

/* Bound a blocking read so a dead/powered-off worker surfaces as a transport
 * error (worker-drop) in seconds instead of the kernel's multi-minute TCP
 * timeout (M7). Only set on the coordinator client fd — the worker's server
 * connection is intentionally allowed to idle between tokens. */
static void off_set_rcvtimeo(int fd, double sec) {
    struct timeval tv;
    tv.tv_sec = (time_t)sec;
    tv.tv_usec = (suseconds_t)((sec - (double)tv.tv_sec) * 1.0e6);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static int off_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        off_set_err(err, errlen, "offload listen resolve %s:%d: %s",
                    host ? host : "*", port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 1) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0)
        off_set_err(err, errlen, "offload listen %s:%d: %s",
                    host ? host : "*", port, strerror(errno));
    return fd;
}

static int off_dial(const char *host, int port, double timeout_sec,
                    char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = off_now_ms() + timeout_sec * 1000.0;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {0}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the worker loads its expert cache; still say
         * why every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "ds4-offload: connecting to %s:%d ... (%s)\n",
                    host, port,
                    gai != 0 ? gai_strerror(gai)
                             : last_errno ? strerror(last_errno)
                                          : "no address worked");
        }
        usleep(200 * 1000);
    } while (off_now_ms() < deadline);
    off_set_err(err, errlen, "offload connect %s:%d: %s", host, port,
                last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

/* Frame I/O. Header on the wire: [ u32 len ][ u8 type ], then `len-1` payload
 * bytes. `len` counts the type byte plus the payload (§8). */
static int off_send_frame(int fd, uint8_t type, const void *payload, uint32_t bytes) {
    uint32_t len = bytes + 1;
    unsigned char head[5];
    memcpy(head, &len, 4);
    head[4] = type;
    if (!off_write_full(fd, head, 5)) return 0;
    if (bytes && !off_write_full(fd, payload, bytes)) return 0;
    return 1;
}

/* Read a frame into `buf` (capacity `cap`). Returns payload length via *bytes
 * and type via *type. Returns 1 on success, 0 on close/error, -1 if the frame
 * is larger than `cap`. */
static int off_recv_frame(int fd, uint8_t *type, void *buf, uint32_t cap,
                          uint32_t *bytes) {
    unsigned char head[5];
    if (!off_read_full(fd, head, 5)) return 0;
    uint32_t len;
    memcpy(&len, head, 4);
    if (len == 0) return 0;             /* malformed: len always counts type */
    *type = head[4];
    uint32_t payload = len - 1;
    if (payload > cap) return -1;
    if (payload && !off_read_full(fd, buf, payload)) return 0;
    *bytes = payload;
    return 1;
}

/* ------------------------------------------------------------------------
 * Residency table (§5.1).
 * --------------------------------------------------------------------- */

void ds4_offload_residency_clear(ds4_offload_residency *r) {
    memset(r, 0, sizeof(*r));
}

void ds4_offload_residency_set(ds4_offload_residency *r, int layer, int expert,
                               bool resident) {
    if (layer < 0 || layer >= DS4_OFFLOAD_N_LAYER) return;
    if (expert < 0 || expert >= DS4_OFFLOAD_N_ROUTED) return;
    uint64_t mask = (uint64_t)1 << (expert & 63);
    if (resident) r->bits[layer][expert >> 6] |= mask;
    else          r->bits[layer][expert >> 6] &= ~mask;
}

bool ds4_offload_residency_get(const ds4_offload_residency *r, int layer,
                               int expert) {
    if (layer < 0 || layer >= DS4_OFFLOAD_N_LAYER) return false;
    if (expert < 0 || expert >= DS4_OFFLOAD_N_ROUTED) return false;
    return (r->bits[layer][expert >> 6] >> (expert & 63)) & 1u;
}

void ds4_offload_residency_split(const ds4_offload_residency *r, int layer,
                                 const uint16_t *expert_ids, int k,
                                 uint8_t *local_idx, int *n_local,
                                 uint8_t *remote_idx, int *n_remote) {
    int nl = 0, nr = 0;
    for (int i = 0; i < k; i++) {
        if (ds4_offload_residency_get(r, layer, expert_ids[i]))
            local_idx[nl++] = (uint8_t)i;
        else
            remote_idx[nr++] = (uint8_t)i;
    }
    *n_local = nl;
    *n_remote = nr;
}

/* ------------------------------------------------------------------------
 * HELLO handshake.
 * --------------------------------------------------------------------- */

static int off_hello_mismatch(const ds4_offload_hello *a,
                              const ds4_offload_hello *b) {
    return a->magic != b->magic || a->version != b->version ||
           a->n_layer != b->n_layer || a->n_used != b->n_used ||
           a->n_embd != b->n_embd || a->n_routed != b->n_routed ||
           a->partition_hash != b->partition_hash || a->model_id != b->model_id;
}

/* Exchange HELLO and abort on mismatch. Both sides send their own then read the
 * peer's; symmetric so there is no ordering deadlock on a small frame. */
static int off_handshake(int fd, const ds4_offload_hello *mine,
                         char *err, size_t errlen) {
    if (!off_send_frame(fd, DS4_OFFLOAD_FRAME_HELLO, mine, sizeof(*mine))) {
        off_set_err(err, errlen, "offload HELLO send failed");
        return -1;
    }
    ds4_offload_hello peer;
    uint8_t type = 0;
    uint32_t bytes = 0;
    int rc = off_recv_frame(fd, &type, &peer, sizeof(peer), &bytes);
    if (rc <= 0 || type != DS4_OFFLOAD_FRAME_HELLO || bytes != sizeof(peer)) {
        off_set_err(err, errlen, "offload HELLO recv failed (rc=%d type=%u)",
                    rc, type);
        return -1;
    }
    if (off_hello_mismatch(mine, &peer)) {
        off_set_err(err, errlen,
                    "offload HELLO mismatch: peer model/partition differs "
                    "(model %llx/%llx part %llx/%llx layers %u/%u)",
                    (unsigned long long)mine->model_id,
                    (unsigned long long)peer.model_id,
                    (unsigned long long)mine->partition_hash,
                    (unsigned long long)peer.partition_hash,
                    mine->n_layer, peer.n_layer);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * EXPERT_REQ / EXPERT_RESP payload (de)serialization (§8).
 *   REQ:  layer u16, seq u64, k u8, ids[k] u16, weights[k] f32,
 *         hidden[n_embd] f16, evict_k u8, evict_ids[evict_k] u16
 *   RESP: seq u64, y[n_embd] f16
 * --------------------------------------------------------------------- */

/* Max REQ payload: 2 + 8 + 1 + k*2 + k*4 + 8192 + 1 + k*2 with k<=N_USED. */
#define OFF_REQ_MAX (2 + 8 + 1 + DS4_OFFLOAD_N_USED * 8 + \
                     (int)DS4_OFFLOAD_HIDDEN_F16_BYTES + 1)
/* RESP: seq u64, status u8, y[n_embd] f16 (§8, H1). */
#define OFF_RESP_MAX (8 + 1 + (int)DS4_OFFLOAD_HIDDEN_F16_BYTES)

static uint32_t off_pack_req(unsigned char *out, int layer, uint64_t seq,
                             const uint16_t *ids, const float *weights, int k,
                             const uint16_t *hidden_f16,
                             const uint16_t *evict_ids, int evict_k) {
    unsigned char *p = out;
    uint16_t l16 = (uint16_t)layer;
    memcpy(p, &l16, 2); p += 2;
    memcpy(p, &seq, 8); p += 8;
    *p++ = (uint8_t)k;
    memcpy(p, ids, (size_t)k * 2); p += (size_t)k * 2;
    memcpy(p, weights, (size_t)k * 4); p += (size_t)k * 4;
    memcpy(p, hidden_f16, DS4_OFFLOAD_HIDDEN_F16_BYTES);
    p += DS4_OFFLOAD_HIDDEN_F16_BYTES;
    *p++ = (uint8_t)evict_k;
    memcpy(p, evict_ids, (size_t)evict_k * 2); p += (size_t)evict_k * 2;
    return (uint32_t)(p - out);
}

/* Parse a REQ payload; returns 0 on success, -1 if malformed / truncated. */
static int off_unpack_req(const unsigned char *in, uint32_t bytes,
                          int *layer, uint64_t *seq,
                          uint16_t *ids, float *weights, int *k,
                          uint16_t *hidden_f16,
                          uint16_t *evict_ids, int *evict_k) {
    const unsigned char *p = in;
    const unsigned char *end = in + bytes;
    if (end - p < 2 + 8 + 1) return -1;
    uint16_t l16; memcpy(&l16, p, 2); p += 2;
    memcpy(seq, p, 8); p += 8;
    uint8_t kk = *p++;
    if (kk > DS4_OFFLOAD_N_USED) return -1;
    size_t need = (size_t)kk * 2 + (size_t)kk * 4 +
                  DS4_OFFLOAD_HIDDEN_F16_BYTES + 1;
    if ((size_t)(end - p) < need) return -1;
    memcpy(ids, p, (size_t)kk * 2); p += (size_t)kk * 2;
    memcpy(weights, p, (size_t)kk * 4); p += (size_t)kk * 4;
    memcpy(hidden_f16, p, DS4_OFFLOAD_HIDDEN_F16_BYTES);
    p += DS4_OFFLOAD_HIDDEN_F16_BYTES;
    uint8_t ek = *p++;
    if (ek > DS4_OFFLOAD_N_USED) return -1;
    if ((size_t)(end - p) < (size_t)ek * 2) return -1;
    memcpy(evict_ids, p, (size_t)ek * 2);
    *layer = l16;
    *k = kk;
    *evict_k = ek;
    return 0;
}

/* ------------------------------------------------------------------------
 * Coordinator client.
 * --------------------------------------------------------------------- */

/* Persistent coordinator->worker connection. Requests are issued under `lock`
 * (monotonic `next_seq`) and responses collected in-order on the same stream
 * (§7); the lock keeps a future multi-threaded issuer from interleaving frames. */
struct ds4_offload_client {
    int fd;
    uint64_t next_seq;
    pthread_mutex_t lock;
};

ds4_offload_client *ds4_offload_client_connect(const char *host, int port,
                                               const ds4_offload_hello *hello,
                                               double timeout_sec,
                                               char *err, size_t errlen) {
    if (port <= 0) port = DS4_OFFLOAD_DEFAULT_PORT;
    int fd = off_dial(host, port, timeout_sec, err, errlen);
    if (fd < 0) return NULL;
    off_socket_tune(fd);
    if (off_handshake(fd, hello, err, errlen) != 0) {
        close(fd);
        return NULL;
    }
    /* Generous vs one round trip (sub-ms) but far below the kernel TCP timeout,
     * so a worker that dies mid-run fails the next collect rather than hanging. */
    off_set_rcvtimeo(fd, 15.0);
    ds4_offload_client *c = calloc(1, sizeof(*c));
    if (!c) {
        off_set_err(err, errlen, "offload client: out of memory");
        close(fd);
        return NULL;
    }
    c->fd = fd;
    c->next_seq = 1;
    pthread_mutex_init(&c->lock, NULL);
    return c;
}

void ds4_offload_client_close(ds4_offload_client *c) {
    if (!c) return;
    off_send_frame(c->fd, DS4_OFFLOAD_FRAME_BYE, NULL, 0);
    close(c->fd);
    pthread_mutex_destroy(&c->lock);
    free(c);
}

uint64_t ds4_offload_client_request(ds4_offload_client *c, int layer,
                                    const uint16_t *expert_ids,
                                    const float *weights, int k,
                                    const uint16_t *hidden_f16,
                                    const uint16_t *evict_ids, int evict_k,
                                    char *err, size_t errlen) {
    if (!c || k < 0 || k > DS4_OFFLOAD_N_USED || evict_k < 0 ||
        evict_k > DS4_OFFLOAD_N_USED) {
        off_set_err(err, errlen, "offload request: bad arguments");
        return 0;
    }
    unsigned char req[OFF_REQ_MAX];
    pthread_mutex_lock(&c->lock);
    uint64_t seq = c->next_seq++;
    uint32_t n = off_pack_req(req, layer, seq, expert_ids, weights, k,
                              hidden_f16, evict_ids, evict_k);
    int ok = off_send_frame(c->fd, DS4_OFFLOAD_FRAME_EXPERT_REQ, req, n);
    pthread_mutex_unlock(&c->lock);
    if (!ok) {
        off_set_err(err, errlen, "offload request: send failed (worker gone?)");
        return 0;
    }
    return seq;
}

/* Phase-1 collection is strictly in-order per the single request stream: the
 * worker replies in receive order, so collecting the next EXPERT_RESP and
 * checking its seq is sufficient and keeps the client lock-simple. Overlap
 * across layers still works because the coordinator issues several requests
 * before collecting the first (the sends are queued on the socket).
 *
 * Returns 0 on success, -1 on transport error (bad/timed-out/closed socket —
 * treat as worker-drop), +1 when the worker replied with an ERROR status (H1;
 * likewise a worker-drop signal — the coordinator must not use `out_f16`). */
int ds4_offload_client_collect(ds4_offload_client *c, uint64_t seq,
                               uint16_t *out_f16, char *err, size_t errlen) {
    if (!c) {
        off_set_err(err, errlen, "offload collect: null client");
        return -1;
    }
    unsigned char resp[OFF_RESP_MAX];
    uint8_t type = 0;
    uint32_t bytes = 0;
    pthread_mutex_lock(&c->lock);
    int rc = off_recv_frame(c->fd, &type, resp, sizeof(resp), &bytes);
    pthread_mutex_unlock(&c->lock);
    if (rc <= 0 || type != DS4_OFFLOAD_FRAME_EXPERT_RESP ||
        bytes != OFF_RESP_MAX) {
        off_set_err(err, errlen, "offload collect: bad response (rc=%d type=%u)",
                    rc, type);
        return -1;
    }
    uint64_t got;
    memcpy(&got, resp, 8);
    if (got != seq) {
        off_set_err(err, errlen,
                    "offload collect: seq mismatch (want %llu got %llu)",
                    (unsigned long long)seq, (unsigned long long)got);
        return -1;
    }
    if (resp[8] != DS4_OFFLOAD_STATUS_OK) {
        off_set_err(err, errlen,
                    "offload collect: worker reported compute error (seq %llu)",
                    (unsigned long long)seq);
        return 1;
    }
    memcpy(out_f16, resp + 9, DS4_OFFLOAD_HIDDEN_F16_BYTES);
    return 0;
}

/* ------------------------------------------------------------------------
 * Worker (expert compute server), §5.3.
 * --------------------------------------------------------------------- */

int ds4_offload_expert_compute_zero(void *user, int layer,
                                    const uint16_t *expert_ids,
                                    const float *weights, int k,
                                    const uint16_t *hidden_f16,
                                    uint16_t *out_f16) {
    (void)user; (void)layer; (void)expert_ids; (void)weights; (void)k;
    (void)hidden_f16;
    memset(out_f16, 0, DS4_OFFLOAD_HIDDEN_F16_BYTES);
    return 0;
}

int ds4_offload_worker_run(const ds4_offload_worker_options *opt,
                           char *err, size_t errlen) {
    if (!opt || !opt->compute) {
        off_set_err(err, errlen, "offload worker: no compute callback");
        return -1;
    }
    int port = opt->port > 0 ? opt->port : DS4_OFFLOAD_DEFAULT_PORT;
    int lfd = off_listen(opt->bind_host, port, err, errlen);
    if (lfd < 0) return -1;

    fprintf(stderr, "ds4-offload: expert-server listening on %s:%d\n",
            opt->bind_host && opt->bind_host[0] ? opt->bind_host : "*", port);

    int rc = 0;
    for (;;) {
        if (opt->stop && *opt->stop) break;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            off_set_err(err, errlen, "offload worker: accept: %s",
                        strerror(errno));
            rc = -1;
            break;
        }
        off_socket_tune(cfd);
        if (off_handshake(cfd, &opt->hello, err, errlen) != 0) {
            fprintf(stderr, "ds4-offload: rejecting coordinator: %s\n", err);
            close(cfd);
            continue;
        }
        fprintf(stderr, "ds4-offload: coordinator connected\n");

        /* Per-connection scratch (kept off the stack: 8 KB payloads). */
        unsigned char *reqbuf = malloc(OFF_REQ_MAX);
        unsigned char *respbuf = malloc(OFF_RESP_MAX);
        uint16_t *hidden = malloc(DS4_OFFLOAD_HIDDEN_F16_BYTES);
        uint16_t *out = malloc(DS4_OFFLOAD_HIDDEN_F16_BYTES);
        uint16_t ids[DS4_OFFLOAD_N_USED];
        float weights[DS4_OFFLOAD_N_USED];
        uint16_t evict_ids[DS4_OFFLOAD_N_USED];
        if (!reqbuf || !respbuf || !hidden || !out) {
            off_set_err(err, errlen, "offload worker: out of memory");
            free(reqbuf); free(respbuf); free(hidden); free(out);
            close(cfd);
            rc = -1;
            break;
        }

        for (;;) {
            if (opt->stop && *opt->stop) break;
            uint8_t type = 0;
            uint32_t bytes = 0;
            int fr = off_recv_frame(cfd, &type, reqbuf, OFF_REQ_MAX, &bytes);
            if (fr <= 0) break;                 /* peer closed / error */
            if (type == DS4_OFFLOAD_FRAME_BYE) break;
            if (type != DS4_OFFLOAD_FRAME_EXPERT_REQ) continue;

            int layer = 0, k = 0, evict_k = 0;
            uint64_t seq = 0;
            if (off_unpack_req(reqbuf, bytes, &layer, &seq, ids, weights, &k,
                               hidden, evict_ids, &evict_k) != 0) {
                fprintf(stderr, "ds4-offload: malformed EXPERT_REQ, dropping\n");
                continue;
            }

            /* Critical path: compute the routed experts and reply FIRST (§5.3).
             * A compute failure (GPU error, or the requested expert not resident
             * in this worker's cache — the invalid-mmap case, H1/H2) is reported
             * as an ERROR status, never as a valid-looking zero vector, so the
             * coordinator never folds silent garbage into the residual. */
            uint8_t status = DS4_OFFLOAD_STATUS_OK;
            if (opt->compute(opt->user, layer, ids, weights, k, hidden, out) != 0) {
                status = DS4_OFFLOAD_STATUS_ERROR;
                memset(out, 0, DS4_OFFLOAD_HIDDEN_F16_BYTES);
                fprintf(stderr,
                        "ds4-offload: expert compute failed (layer %d k=%d); "
                        "replying ERROR\n", layer, k);
            }
            memcpy(respbuf, &seq, 8);
            respbuf[8] = status;
            memcpy(respbuf + 9, out, DS4_OFFLOAD_HIDDEN_F16_BYTES);
            if (!off_send_frame(cfd, DS4_OFFLOAD_FRAME_EXPERT_RESP, respbuf,
                                OFF_RESP_MAX))
                break;

            /* Background, after the response: page the evicted experts in from
             * this worker's own SSD copy (§6). Weights never cross the wire. */
            if (evict_k > 0 && opt->evict)
                opt->evict(opt->user, layer, evict_ids, evict_k);
        }

        free(reqbuf); free(respbuf); free(hidden); free(out);
        close(cfd);
        fprintf(stderr, "ds4-offload: coordinator disconnected\n");
    }
    close(lfd);
    return rc;
}

/* ------------------------------------------------------------------------
 * Phase-0 ping-pong RTT gate (§7, §9).
 * --------------------------------------------------------------------- */

int ds4_offload_pingpong_serve(const char *bind_host, int port,
                               size_t payload_bytes, volatile int *stop,
                               char *err, size_t errlen) {
    if (port <= 0) port = DS4_OFFLOAD_DEFAULT_PORT;
    int lfd = off_listen(bind_host, port, err, errlen);
    if (lfd < 0) return -1;
    fprintf(stderr, "ds4-offload-ping: server listening on %s:%d (%zu B)\n",
            bind_host && bind_host[0] ? bind_host : "*", port, payload_bytes);
    unsigned char *buf = malloc(payload_bytes ? payload_bytes : 1);
    if (!buf) {
        off_set_err(err, errlen, "pingpong serve: out of memory");
        close(lfd);
        return -1;
    }
    int rc = 0;
    for (;;) {
        if (stop && *stop) break;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            off_set_err(err, errlen, "pingpong serve: accept: %s",
                        strerror(errno));
            rc = -1;
            break;
        }
        off_socket_tune(cfd);
        for (;;) {
            if (stop && *stop) break;
            uint8_t type = 0;
            uint32_t bytes = 0;
            int fr = off_recv_frame(cfd, &type, buf, (uint32_t)payload_bytes,
                                    &bytes);
            if (fr <= 0) break;
            if (type != DS4_OFFLOAD_FRAME_PING) break;
            if (!off_send_frame(cfd, DS4_OFFLOAD_FRAME_PONG, buf, bytes)) break;
        }
        close(cfd);
        if (stop && *stop) break;
    }
    free(buf);
    close(lfd);
    return rc;
}

static int off_cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int ds4_offload_pingpong_run(const char *host, int port, size_t payload_bytes,
                             int iterations, ds4_offload_pingpong_stats *out,
                             char *err, size_t errlen) {
    if (port <= 0) port = DS4_OFFLOAD_DEFAULT_PORT;
    if (iterations <= 0) iterations = 1000;
    int fd = off_dial(host, port, 30.0, err, errlen);
    if (fd < 0) return -1;
    off_socket_tune(fd);

    unsigned char *snd = malloc(payload_bytes ? payload_bytes : 1);
    unsigned char *rcv = malloc(payload_bytes ? payload_bytes : 1);
    double *samples = malloc((size_t)iterations * sizeof(double));
    if (!snd || !rcv || !samples) {
        off_set_err(err, errlen, "pingpong run: out of memory");
        free(snd); free(rcv); free(samples); close(fd);
        return -1;
    }
    for (size_t i = 0; i < payload_bytes; i++) snd[i] = (unsigned char)i;

    /* Warm the path (TCP window, page-ins) before timing. */
    for (int w = 0; w < 16; w++) {
        uint8_t t; uint32_t b;
        if (!off_send_frame(fd, DS4_OFFLOAD_FRAME_PING, snd,
                            (uint32_t)payload_bytes) ||
            off_recv_frame(fd, &t, rcv, (uint32_t)payload_bytes, &b) <= 0) {
            off_set_err(err, errlen, "pingpong run: warmup failed");
            free(snd); free(rcv); free(samples); close(fd);
            return -1;
        }
    }

    double sum = 0.0, mn = 1e30, mx = 0.0;
    for (int i = 0; i < iterations; i++) {
        double t0 = off_now_ms();
        uint8_t t; uint32_t b;
        if (!off_send_frame(fd, DS4_OFFLOAD_FRAME_PING, snd,
                            (uint32_t)payload_bytes) ||
            off_recv_frame(fd, &t, rcv, (uint32_t)payload_bytes, &b) <= 0 ||
            t != DS4_OFFLOAD_FRAME_PONG) {
            off_set_err(err, errlen, "pingpong run: exchange failed at %d", i);
            free(snd); free(rcv); free(samples); close(fd);
            return -1;
        }
        double dt = off_now_ms() - t0;
        samples[i] = dt;
        sum += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
    }
    off_send_frame(fd, DS4_OFFLOAD_FRAME_BYE, NULL, 0);
    close(fd);

    qsort(samples, (size_t)iterations, sizeof(double), off_cmp_double);
    if (out) {
        out->samples = (uint64_t)iterations;
        out->payload_bytes = payload_bytes;
        out->min_ms = mn;
        out->max_ms = mx;
        out->mean_ms = sum / iterations;
        out->p50_ms = samples[iterations / 2];
        out->p99_ms = samples[(int)((double)iterations * 0.99)];
    }
    free(snd); free(rcv); free(samples);
    return 0;
}
