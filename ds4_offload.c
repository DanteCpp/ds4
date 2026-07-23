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

/* ------------------------------------------------------------------------
 * Session debug log.
 * --------------------------------------------------------------------- */

struct ds4_offload_log {
    FILE *f;
    double t0_ms;
};

ds4_offload_log *ds4_offload_log_open(const char *path) {
    if (!path || !path[0]) return NULL;
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "ds4: offload log: cannot open %s: %s\n",
                path, strerror(errno));
        return NULL;
    }
    ds4_offload_log *l = calloc(1, sizeof(*l));
    if (!l) { fclose(f); return NULL; }
    l->f = f;
    l->t0_ms = off_now_ms();
    return l;
}

void ds4_offload_log_close(ds4_offload_log *l) {
    if (!l) return;
    fclose(l->f);
    free(l);
}

static void off_logv(ds4_offload_log *l, int also_stderr,
                     const char *fmt, va_list ap) {
    if (!l || !l->f) return;
    char body[1500];
    vsnprintf(body, sizeof(body), fmt, ap);
    fprintf(l->f, "ds4: [t=+%.1fms] %s\n", off_now_ms() - l->t0_ms, body);
    fflush(l->f);
    if (also_stderr)
        fprintf(stderr, "ds4: [t=+%.1fms] %s\n", off_now_ms() - l->t0_ms, body);
}

void ds4_offload_logf(ds4_offload_log *l, const char *fmt, ...) {
    /* File-only: when a session log is enabled the per-token / lifecycle lines
     * belong in the file, not the interactive CLI (the user reads the file
     * post-hoc). Direct fprintf(stderr) lifecycle notices elsewhere are
     * unaffected; this only silences the timestamped log stream. */
    va_list ap; va_start(ap, fmt);
    off_logv(l, 0, fmt, ap);
    va_end(ap);
}

void ds4_offload_logf_file(ds4_offload_log *l, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    off_logv(l, 0, fmt, ap);
    va_end(ap);
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

/* Identity fields both sides must agree on before any orchestration. The
 * partition_hash is NOT compared: in v2 the split is decided by the
 * coordinator during the handshake, so there is nothing to disagree about. */
static int off_hello_mismatch(const ds4_offload_hello *a,
                              const ds4_offload_hello *b) {
    return a->magic != b->magic || a->version != b->version ||
           a->n_layer != b->n_layer || a->n_used != b->n_used ||
           a->n_embd != b->n_embd || a->n_routed != b->n_routed ||
           a->model_id != b->model_id;
}

static int off_hello_check(const ds4_offload_hello *mine,
                           const ds4_offload_hello *peer,
                           char *err, size_t errlen) {
    if (off_hello_mismatch(mine, peer)) {
        off_set_err(err, errlen,
                    "offload HELLO mismatch: peer model/protocol differs "
                    "(model %llx/%llx proto %u/%u layers %u/%u)",
                    (unsigned long long)mine->model_id,
                    (unsigned long long)peer->model_id,
                    mine->version, peer->version,
                    mine->n_layer, peer->n_layer);
        return -1;
    }
    return 0;
}

static int off_hello_recv(int fd, ds4_offload_hello *peer,
                          char *err, size_t errlen) {
    uint8_t type = 0;
    uint32_t bytes = 0;
    int rc = off_recv_frame(fd, &type, peer, sizeof(*peer), &bytes);
    if (rc <= 0 || type != DS4_OFFLOAD_FRAME_HELLO || bytes != sizeof(*peer)) {
        off_set_err(err, errlen, "offload HELLO recv failed (rc=%d type=%u)",
                    rc, type);
        return -1;
    }
    return 0;
}

/* PLAN wire format: u32 count, then count x (u16 layer, u16 expert) LE. */
#define OFF_PLAN_MAX_BYTES (4 + (uint32_t)DS4_OFFLOAD_PLAN_MAX * 4)

static uint32_t off_pack_plan(unsigned char *out,
                              const ds4_offload_expert_ref *plan,
                              uint32_t count) {
    unsigned char *p = out;
    memcpy(p, &count, 4); p += 4;
    for (uint32_t i = 0; i < count; i++) {
        memcpy(p, &plan[i].layer, 2); p += 2;
        memcpy(p, &plan[i].expert, 2); p += 2;
    }
    return (uint32_t)(p - out);
}

static int off_unpack_plan(const unsigned char *in, uint32_t bytes,
                           ds4_offload_expert_ref *plan, uint32_t max,
                           uint32_t *count) {
    if (bytes < 4) return -1;
    uint32_t n;
    memcpy(&n, in, 4);
    if (n > max || bytes != 4 + n * 4) return -1;
    const unsigned char *p = in + 4;
    for (uint32_t i = 0; i < n; i++) {
        memcpy(&plan[i].layer, p, 2); p += 2;
        memcpy(&plan[i].expert, p, 2); p += 2;
    }
    *count = n;
    return 0;
}

/* PLAN_ACK wire format: u32 status (0 ok / 1 error), u32 installed,
 * u64 wired_bytes, then a NUL-free short message (<= 120 B, may be empty). */
#define OFF_PLAN_ACK_MSG 120
#define OFF_PLAN_ACK_MAX (4 + 4 + 8 + OFF_PLAN_ACK_MSG)

static uint32_t off_pack_plan_ack(unsigned char *out, uint32_t status,
                                  uint32_t installed, uint64_t wired_bytes,
                                  const char *msg) {
    unsigned char *p = out;
    memcpy(p, &status, 4); p += 4;
    memcpy(p, &installed, 4); p += 4;
    memcpy(p, &wired_bytes, 8); p += 8;
    if (msg) {
        size_t m = strlen(msg);
        if (m > OFF_PLAN_ACK_MSG) m = OFF_PLAN_ACK_MSG;
        memcpy(p, msg, m); p += m;
    }
    return (uint32_t)(p - out);
}

/* ------------------------------------------------------------------------
 * EXPERT_REQ / EXPERT_RESP payload (de)serialization (§8).
 *   REQ:  layer u16, seq u64, k u8, ids[k] u16, weights[k] f32,
 *         hidden[n_embd] f16, swap_k u8, (evict u16, load u16)[swap_k]
 *   RESP: seq u64, status u8, y[n_embd] f16
 *
 * Phase-2 swap (plan §6): each swap entry names BOTH the expert to evict from
 * the worker cache and the expert to load into its slot — the coordinator is
 * authoritative over the worker's contents, so it dictates the exact victim
 * rather than letting the worker choose (which would desynchronize the two
 * sides). Both experts belong to `layer`.
 * --------------------------------------------------------------------- */

/* Max REQ payload: 2+8+1 + k*(2 ids + 4 w) + hidden + 1 + k*(2+2 swap pair). */
#define OFF_REQ_MAX (2 + 8 + 1 + DS4_OFFLOAD_N_USED * 10 + \
                     (int)DS4_OFFLOAD_HIDDEN_F16_BYTES + 1)
/* RESP: seq u64, status u8, y[n_embd] f16 (§8, H1). */
#define OFF_RESP_MAX (8 + 1 + (int)DS4_OFFLOAD_HIDDEN_F16_BYTES)

static uint32_t off_pack_req(unsigned char *out, int layer, uint64_t seq,
                             const uint16_t *ids, const float *weights, int k,
                             const uint16_t *hidden_f16,
                             const uint16_t *swap_evict,
                             const uint16_t *swap_load, int swap_k) {
    unsigned char *p = out;
    uint16_t l16 = (uint16_t)layer;
    memcpy(p, &l16, 2); p += 2;
    memcpy(p, &seq, 8); p += 8;
    *p++ = (uint8_t)k;
    memcpy(p, ids, (size_t)k * 2); p += (size_t)k * 2;
    memcpy(p, weights, (size_t)k * 4); p += (size_t)k * 4;
    memcpy(p, hidden_f16, DS4_OFFLOAD_HIDDEN_F16_BYTES);
    p += DS4_OFFLOAD_HIDDEN_F16_BYTES;
    *p++ = (uint8_t)swap_k;
    for (int i = 0; i < swap_k; i++) {
        memcpy(p, &swap_evict[i], 2); p += 2;
        memcpy(p, &swap_load[i], 2);  p += 2;
    }
    return (uint32_t)(p - out);
}

/* Parse a REQ payload; returns 0 on success, -1 if malformed / truncated. */
static int off_unpack_req(const unsigned char *in, uint32_t bytes,
                          int *layer, uint64_t *seq,
                          uint16_t *ids, float *weights, int *k,
                          uint16_t *hidden_f16,
                          uint16_t *swap_evict, uint16_t *swap_load, int *swap_k) {
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
    uint8_t sk = *p++;
    if (sk > DS4_OFFLOAD_N_USED) return -1;
    if ((size_t)(end - p) < (size_t)sk * 4) return -1;
    for (int i = 0; i < sk; i++) {
        memcpy(&swap_evict[i], p, 2); p += 2;
        memcpy(&swap_load[i],  p, 2); p += 2;
    }
    *layer = l16;
    *k = kk;
    *swap_k = sk;
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
                                               ds4_offload_hello *worker_hello,
                                               char *err, size_t errlen) {
    if (port <= 0) port = DS4_OFFLOAD_DEFAULT_PORT;
    int fd = off_dial(host, port, timeout_sec, err, errlen);
    if (fd < 0) return NULL;
    off_socket_tune(fd);
    /* Orchestration step 1: the worker speaks first, offering its identity and
     * available expert-cache memory. No receive timeout yet — the worker may
     * still be loading its model. The coordinator's own HELLO goes out only
     * after the split decision (ds4_offload_client_orchestrate). */
    ds4_offload_hello peer;
    if (off_hello_recv(fd, &peer, err, errlen) != 0 ||
        off_hello_check(hello, &peer, err, errlen) != 0) {
        close(fd);
        return NULL;
    }
    ds4_offload_client *c = calloc(1, sizeof(*c));
    if (!c) {
        off_set_err(err, errlen, "offload client: out of memory");
        close(fd);
        return NULL;
    }
    c->fd = fd;
    c->next_seq = 1;
    pthread_mutex_init(&c->lock, NULL);
    if (worker_hello) *worker_hello = peer;
    return c;
}

int ds4_offload_client_orchestrate(ds4_offload_client *c,
                                   const ds4_offload_hello *decision,
                                   const ds4_offload_expert_ref *plan,
                                   uint32_t count,
                                   uint32_t *installed,
                                   uint64_t *wired_bytes,
                                   char *err, size_t errlen) {
    if (!c || !decision || (count > 0 && !plan) ||
        count > DS4_OFFLOAD_PLAN_MAX) {
        off_set_err(err, errlen, "offload orchestrate: bad arguments");
        return -1;
    }
    pthread_mutex_lock(&c->lock);
    int rc = -1;
    unsigned char *planbuf = malloc(count ? 4 + count * 4 : 4);
    unsigned char ack[OFF_PLAN_ACK_MAX];
    if (!planbuf) {
        off_set_err(err, errlen, "offload orchestrate: out of memory");
        goto done;
    }
    /* Orchestration step 2: decision HELLO + PLAN, then block while the worker
     * installs its assigned experts (pread + mlock; can take minutes). */
    if (!off_send_frame(c->fd, DS4_OFFLOAD_FRAME_HELLO, decision,
                        sizeof(*decision))) {
        off_set_err(err, errlen, "offload orchestrate: HELLO send failed");
        goto done;
    }
    uint32_t pn = off_pack_plan(planbuf, plan, count);
    if (!off_send_frame(c->fd, DS4_OFFLOAD_FRAME_PLAN, planbuf, pn)) {
        off_set_err(err, errlen, "offload orchestrate: PLAN send failed");
        goto done;
    }
    uint8_t type = 0;
    uint32_t bytes = 0;
    int rr = off_recv_frame(c->fd, &type, ack, sizeof(ack), &bytes);
    if (rr <= 0 || type != DS4_OFFLOAD_FRAME_PLAN_ACK || bytes < 16) {
        off_set_err(err, errlen,
                    "offload orchestrate: PLAN_ACK recv failed (rc=%d type=%u)",
                    rr, type);
        goto done;
    }
    uint32_t status, ack_installed;
    uint64_t ack_wired;
    memcpy(&status, ack, 4);
    memcpy(&ack_installed, ack + 4, 4);
    memcpy(&ack_wired, ack + 8, 8);
    if (status != 0) {
        char msg[OFF_PLAN_ACK_MSG + 1] = "";
        if (bytes > 16) memcpy(msg, ack + 16, bytes - 16);
        off_set_err(err, errlen, "offload orchestrate: worker rejected plan: %s",
                    msg[0] ? msg : "no detail");
        goto done;
    }
    if (installed) *installed = ack_installed;
    if (wired_bytes) *wired_bytes = ack_wired;
    /* Orchestration done; requests now flow. Bound reads so a worker that dies
     * mid-run fails the next collect rather than hanging (M7). */
    off_set_rcvtimeo(c->fd, 15.0);
    rc = 0;
done:
    free(planbuf);
    pthread_mutex_unlock(&c->lock);
    return rc;
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
                                    const uint16_t *swap_evict,
                                    const uint16_t *swap_load, int swap_k,
                                    char *err, size_t errlen) {
    if (!c || k < 0 || k > DS4_OFFLOAD_N_USED || swap_k < 0 ||
        swap_k > DS4_OFFLOAD_N_USED) {
        off_set_err(err, errlen, "offload request: bad arguments");
        return 0;
    }
    unsigned char req[OFF_REQ_MAX];
    pthread_mutex_lock(&c->lock);
    uint64_t seq = c->next_seq++;
    uint32_t n = off_pack_req(req, layer, seq, expert_ids, weights, k,
                              hidden_f16, swap_evict, swap_load, swap_k);
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

/* Verbose per-request worker log (every EXPERT_REQ with ids/weights/timing).
 * The per-token aggregate line is always printed; set
 * DS4_OFFLOAD_WORKER_VERBOSE=1 for the full request stream. */
static int off_worker_verbose(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("DS4_OFFLOAD_WORKER_VERBOSE");
        v = (s && s[0] && s[0] != '0') ? 1 : 0;
    }
    return v;
}

/* Per-token aggregation for the worker's live compute log. One token = one
 * increasing run of layer requests (the coordinator sends at most one
 * EXPERT_REQ per layer per token, in layer order); a non-increasing layer id
 * closes the previous token and flushes its aggregate line. The aggregate
 * answers "what did the worker execute for this token, how long did it take,
 * and is the cache still serving": layer-req count, total experts executed,
 * summed GPU compute time, errors (cache misses surface here), eviction
 * hints, and the engine's diag suffix (residency / hits / misses). */
typedef struct {
    uint64_t tok_no;
    int open;
    int layers;                                /* EXPERT_REQs in this token  */
    int experts;                               /* sum of k over the requests */
    int errors;                                /* compute failures (= cache misses, H1) */
    int evict_hints;                           /* evict_k sum                */
    double compute_ms;                         /* summed compute-callback time */
    double wall0_ms;                           /* first request of the token   */
    int prev_layer;
    uint8_t k[DS4_OFFLOAD_N_LAYER];            /* per-layer k (0 = not seen) */
    uint16_t ids[DS4_OFFLOAD_N_LAYER][DS4_OFFLOAD_N_USED];
} off_tok_agg;

static void off_tok_agg_init(off_tok_agg *a) {
    memset(a, 0, sizeof(*a));
    a->prev_layer = -1;
}

static void off_tok_agg_flush(off_tok_agg *a,
                              const ds4_offload_worker_options *opt) {
    if (!a->open) return;
    const double wall_ms = off_now_ms() - a->wall0_ms;
    char diag[192] = "";
    if (opt->diag) opt->diag(opt->user, diag, sizeof(diag));
    fprintf(stderr,
            "ds4-offload: tok#%llu done: %d layer-reqs, %d experts, "
            "compute %.2f ms (%.2f ms/layer), wall %.2f ms, "
            "errors=%d, evict-hints=%d%s%s\n",
            (unsigned long long)a->tok_no, a->layers, a->experts,
            a->compute_ms,
            a->layers > 0 ? a->compute_ms / a->layers : 0.0,
            wall_ms, a->errors, a->evict_hints,
            diag[0] ? " | " : "", diag);
    if (opt->log) {
        ds4_offload_logf(opt->log,
            "tok#%llu done: %d layer-reqs, %d experts, compute %.2f ms "
            "(%.2f ms/layer), wall %.2f ms, errors=%d, evict-hints=%d%s%s",
            (unsigned long long)a->tok_no, a->layers, a->experts,
            a->compute_ms,
            a->layers > 0 ? a->compute_ms / a->layers : 0.0,
            wall_ms, a->errors, a->evict_hints,
            diag[0] ? " | " : "", diag);
        /* The per-token expert map always goes to the session log (it is the
         * point of the file); stderr gets it only in verbose mode. */
        char map[1200];
        size_t mo = (size_t)snprintf(map, sizeof(map), "tok#%llu experts:",
                                     (unsigned long long)a->tok_no);
        for (int l = 0; l < DS4_OFFLOAD_N_LAYER && mo < sizeof(map) - 40; l++) {
            if (!a->k[l]) continue;
            mo += (size_t)snprintf(map + mo, sizeof(map) - mo, " L%d=[", l);
            for (int i = 0; i < a->k[l]; i++)
                mo += (size_t)snprintf(map + mo, sizeof(map) - mo, "%s%u",
                                       i ? "," : "", (unsigned)a->ids[l][i]);
            mo += (size_t)snprintf(map + mo, sizeof(map) - mo, "]");
        }
        ds4_offload_logf_file(opt->log, "%s", map);
    }
    if (off_worker_verbose()) {
        fprintf(stderr, "ds4-offload:   tok#%llu experts:",
                (unsigned long long)a->tok_no);
        for (int l = 0; l < DS4_OFFLOAD_N_LAYER; l++) {
            if (!a->k[l]) continue;
            fprintf(stderr, " L%d=[", l);
            for (int i = 0; i < a->k[l]; i++)
                fprintf(stderr, "%s%u", i ? "," : "", (unsigned)a->ids[l][i]);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }
    const uint64_t next = a->tok_no + 1;
    off_tok_agg_init(a);
    a->tok_no = next;
}

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

        /* Orchestration phase (v2). Step 1: offer identity + available memory.
         * Step 2: read the coordinator's decision HELLO, then the PLAN frame
         * with the exact expert ids to hold, install them (the plan callback
         * preads + mlocks; this is the slow part of startup), and ACK. Only
         * then does the request loop below start. */
        if (!off_send_frame(cfd, DS4_OFFLOAD_FRAME_HELLO, &opt->hello,
                            sizeof(opt->hello))) {
            fprintf(stderr, "ds4-offload: HELLO send failed\n");
            close(cfd);
            continue;
        }
        ds4_offload_hello decision;
        if (off_hello_recv(cfd, &decision, err, errlen) != 0 ||
            off_hello_check(&opt->hello, &decision, err, errlen) != 0) {
            fprintf(stderr, "ds4-offload: rejecting coordinator: %s\n", err);
            close(cfd);
            continue;
        }
        fprintf(stderr,
                "ds4-offload: coordinator connected; plan: %u experts for this "
                "worker (coordinator keeps %u hottest, SSD tail %s)\n",
                decision.worker_cap, decision.coord_cap,
                (decision.flags & DS4_OFFLOAD_HELLO_FLAG_SSD_TAIL)
                    ? "active on coordinator" : "empty (combined RAM fits all)");
        if (opt->log)
            ds4_offload_logf(opt->log,
                "coordinator connected; plan: %u experts for this worker "
                "(coordinator keeps %u hottest, SSD tail %s)",
                decision.worker_cap, decision.coord_cap,
                (decision.flags & DS4_OFFLOAD_HELLO_FLAG_SSD_TAIL)
                    ? "active on coordinator" : "empty (combined RAM fits all)");
        if (decision.plan_count != decision.worker_cap ||
            decision.plan_count > DS4_OFFLOAD_PLAN_MAX) {
            fprintf(stderr, "ds4-offload: bad plan_count %u (worker_cap %u), "
                            "dropping\n", decision.plan_count, decision.worker_cap);
            close(cfd);
            continue;
        }
        {
            unsigned char *planbuf = malloc(OFF_PLAN_MAX_BYTES);
            ds4_offload_expert_ref *planrefs =
                malloc((size_t)DS4_OFFLOAD_PLAN_MAX * sizeof(*planrefs));
            unsigned char ackbuf[OFF_PLAN_ACK_MAX];
            uint32_t installed = 0;
            uint64_t wired = 0;
            uint32_t status = DS4_OFFLOAD_STATUS_OK;
            char plan_err[OFF_PLAN_ACK_MSG + 1] = "";
            uint8_t type = 0;
            uint32_t bytes = 0;
            int fr = planbuf && planrefs
                ? off_recv_frame(cfd, &type, planbuf, OFF_PLAN_MAX_BYTES, &bytes)
                : -1;
            if (fr <= 0 || type != DS4_OFFLOAD_FRAME_PLAN ||
                off_unpack_plan(planbuf, bytes, planrefs,
                                DS4_OFFLOAD_PLAN_MAX, &installed) != 0 ||
                installed != decision.plan_count) {
                fprintf(stderr, "ds4-offload: malformed PLAN frame, dropping\n");
                free(planbuf); free(planrefs);
                close(cfd);
                continue;
            }
            uint32_t count = installed;
            installed = 0;
            if (!opt->plan) {
                status = DS4_OFFLOAD_STATUS_ERROR;
                snprintf(plan_err, sizeof(plan_err),
                         "worker has no plan callback");
            } else if (opt->plan(opt->user, decision.coord_cap, planrefs, count,
                                 &installed, &wired,
                                 plan_err, sizeof(plan_err) - 1) != 0) {
                status = DS4_OFFLOAD_STATUS_ERROR;
            }
            uint32_t an = off_pack_plan_ack(ackbuf, status, installed, wired,
                                            plan_err);
            off_send_frame(cfd, DS4_OFFLOAD_FRAME_PLAN_ACK, ackbuf, an);
            free(planbuf); free(planrefs);
            if (status != DS4_OFFLOAD_STATUS_OK) {
                fprintf(stderr,
                        "ds4-offload: plan install failed: %s; closing\n",
                        plan_err[0] ? plan_err : "unknown error");
                close(cfd);
                continue;
            }
            fprintf(stderr,
                    "ds4-offload: plan installed: %u/%u experts wired "
                    "(%.2f GiB); serving\n",
                    installed, count,
                    (double)wired / (1024.0 * 1024.0 * 1024.0));
            if (opt->log)
                ds4_offload_logf(opt->log,
                    "plan installed: %u/%u experts wired (%.2f GiB); serving",
                    installed, count,
                    (double)wired / (1024.0 * 1024.0 * 1024.0));
        }

        /* Per-token aggregation state + session totals (reset per coordinator
         * connection so the token numbering matches this session). */
        off_tok_agg agg;
        off_tok_agg_init(&agg);
        uint64_t sess_reqs = 0, sess_experts = 0;
        double sess_compute_ms = 0.0;

        /* Per-connection scratch (kept off the stack: 8 KB payloads). */
        unsigned char *reqbuf = malloc(OFF_REQ_MAX);
        unsigned char *respbuf = malloc(OFF_RESP_MAX);
        uint16_t *hidden = malloc(DS4_OFFLOAD_HIDDEN_F16_BYTES);
        uint16_t *out = malloc(DS4_OFFLOAD_HIDDEN_F16_BYTES);
        uint16_t ids[DS4_OFFLOAD_N_USED];
        float weights[DS4_OFFLOAD_N_USED];
        uint16_t swap_evict[DS4_OFFLOAD_N_USED];
        uint16_t swap_load[DS4_OFFLOAD_N_USED];
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

            int layer = 0, k = 0, swap_k = 0;
            uint64_t seq = 0;
            if (off_unpack_req(reqbuf, bytes, &layer, &seq, ids, weights, &k,
                               hidden, swap_evict, swap_load, &swap_k) != 0) {
                fprintf(stderr, "ds4-offload: malformed EXPERT_REQ, dropping\n");
                continue;
            }

            /* Apply the coordinator's swaps BEFORE computing so a
             * just-loaded expert is present for this request (the
             * deferred-swap piggyback can carry a load=Y that is also
             * in the compute set).  Swap cost is on the critical path
             * but eliminates the race and the extra network message. */
            if (swap_k > 0 && opt->evict) {
                opt->evict(opt->user, layer, swap_evict, swap_load, swap_k);
                agg.evict_hints += swap_k;
            }

            /* Critical path: compute the routed experts and reply.
             * A compute failure (GPU error, or the requested expert not
             * resident in this worker's cache — the invalid-mmap case,
             * H1/H2) is reported as an ERROR status, never as a
             * valid-looking zero vector, so the coordinator never
             * folds silent garbage into the residual. */
            const double compute_t0 = off_now_ms();
            const int compute_rc =
                opt->compute(opt->user, layer, ids, weights, k, hidden, out);
            const double compute_ms = off_now_ms() - compute_t0;
            uint8_t status = DS4_OFFLOAD_STATUS_OK;
            if (compute_rc != 0) {
                status = DS4_OFFLOAD_STATUS_ERROR;
                memset(out, 0, DS4_OFFLOAD_HIDDEN_F16_BYTES);
                fprintf(stderr,
                        "ds4-offload: expert compute failed (layer %d k=%d); "
                        "replying ERROR\n", layer, k);
            }

            if (off_worker_verbose() || opt->log) {
                char ids_buf[64], w_buf[96];
                size_t io = 0, wo = 0;
                for (int i = 0; i < k; i++) {
                    io += (size_t)snprintf(ids_buf + io, sizeof(ids_buf) - io,
                                           "%s%u", i ? "," : "", (unsigned)ids[i]);
                    wo += (size_t)snprintf(w_buf + wo, sizeof(w_buf) - wo,
                                           "%s%.4f", i ? "," : "", (double)weights[i]);
                }
                if (off_worker_verbose())
                    fprintf(stderr,
                            "ds4-offload:   req seq=%llu L%d k=%d ids=[%s] "
                            "w=[%s] compute=%.2f ms %s\n",
                            (unsigned long long)seq, layer, k, ids_buf, w_buf,
                            compute_ms, compute_rc == 0 ? "ok" : "ERROR");
                if (opt->log)
                    ds4_offload_logf_file(opt->log,
                        "req seq=%llu L%d k=%d ids=[%s] w=[%s] compute=%.2f ms %s",
                        (unsigned long long)seq, layer, k, ids_buf, w_buf,
                        compute_ms, compute_rc == 0 ? "ok" : "ERROR");
            }

            /* Aggregate into the current token. A non-increasing layer id
             * means the coordinator started the next token: flush first. */
            if (layer >= 0 && layer < DS4_OFFLOAD_N_LAYER) {
                if (agg.open && layer <= agg.prev_layer)
                    off_tok_agg_flush(&agg, opt);
                if (!agg.open) {
                    agg.open = 1;
                    agg.wall0_ms = compute_t0;
                }
                agg.layers++;
                agg.experts += k;
                agg.errors += (compute_rc != 0);
                agg.compute_ms += compute_ms;
                agg.k[layer] = (uint8_t)(k > DS4_OFFLOAD_N_USED
                                         ? DS4_OFFLOAD_N_USED : k);
                memcpy(agg.ids[layer], ids,
                       (size_t)agg.k[layer] * sizeof(uint16_t));
                agg.prev_layer = layer;
                sess_reqs++;
                sess_experts += (uint64_t)k;
                sess_compute_ms += compute_ms;
            }
            memcpy(respbuf, &seq, 8);
            respbuf[8] = status;
            memcpy(respbuf + 9, out, DS4_OFFLOAD_HIDDEN_F16_BYTES);
            if (!off_send_frame(cfd, DS4_OFFLOAD_FRAME_EXPERT_RESP, respbuf,
                                OFF_RESP_MAX))
                break;
        }

        off_tok_agg_flush(&agg, opt);
        fprintf(stderr,
                "ds4-offload: session totals: %llu tokens, %llu layer-reqs, "
                "%llu experts, compute %.2f ms total%s\n",
                (unsigned long long)agg.tok_no,
                (unsigned long long)sess_reqs,
                (unsigned long long)sess_experts, sess_compute_ms,
                agg.tok_no > 0
                    ? " (see per-token lines above for per-layer detail)"
                    : "");
        if (opt->log)
            ds4_offload_logf(opt->log,
                "session totals: %llu tokens, %llu layer-reqs, %llu experts, "
                "compute %.2f ms total; coordinator disconnected",
                (unsigned long long)agg.tok_no,
                (unsigned long long)sess_reqs,
                (unsigned long long)sess_experts, sess_compute_ms);

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
