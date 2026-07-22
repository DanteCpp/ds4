/* ds4-offload-ping — Phase 0 of DISTRIBUTED_EXPERT_OFFLOAD_PLAN.md.
 *
 * The go/no-go gate: bounce an 8 KB buffer over a real TCP_NODELAY socket on
 * bridge0 and report p50/p99 RTT. This single number picks the throughput row
 * in §7 and is "the first thing to measure". Do NOT use en4 (100 Mbit); bridge0
 * only.
 *
 *   worker (mtwo):        ./ds4-offload-ping --serve [--bind <ip>] [--port N]
 *   coordinator (mone):   ./ds4-offload-ping --host <mtwo-bridge0-ip> [--port N]
 *
 * Smoke-test on one host over loopback:
 *   ./ds4-offload-ping --serve &
 *   ./ds4-offload-ping --host 127.0.0.1
 */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_offload.h"

static volatile int g_stop = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

static void usage(FILE *fp) {
    fprintf(fp,
        "ds4-offload-ping — Phase 0 bridge0 RTT gate\n"
        "  --serve                run the echo server (on the worker, mtwo)\n"
        "  --host <ip>            run the client against <ip> (on mone)\n"
        "  --bind <ip>            server bind address (default: all)\n"
        "  --port <n>            TCP port (default %d)\n"
        "  --bytes <n>           payload size (default 8192 = 8 KB f16 hidden)\n"
        "  --iters <n>           client round trips to time (default 2000)\n",
        DS4_OFFLOAD_DEFAULT_PORT);
}

int main(int argc, char **argv) {
    int serve = 0;
    const char *host = NULL;
    const char *bind_host = NULL;
    int port = DS4_OFFLOAD_DEFAULT_PORT;
    size_t bytes = DS4_OFFLOAD_HIDDEN_F16_BYTES; /* 8 KB */
    int iters = 2000;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(a, "--bind") && i + 1 < argc) bind_host = argv[++i];
        else if (!strcmp(a, "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--bytes") && i + 1 < argc) bytes = (size_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(stderr); return 2; }
    }
    if (serve == (host != NULL)) {
        fprintf(stderr, "specify exactly one of --serve or --host\n");
        usage(stderr);
        return 2;
    }

    signal(SIGINT, on_sigint);
#ifdef SIGPIPE
    signal(SIGPIPE, SIG_IGN);
#endif

    char err[256] = {0};
    if (serve) {
        if (ds4_offload_pingpong_serve(bind_host, port, bytes, &g_stop,
                                       err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-offload-ping serve: %s\n", err);
            return 1;
        }
        return 0;
    }

    ds4_offload_pingpong_stats st;
    if (ds4_offload_pingpong_run(host, port, bytes, iters, &st,
                                 err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-offload-ping run: %s\n", err);
        return 1;
    }
    printf("ds4-offload-ping: %llu round trips of %zu B over %s:%d\n",
           (unsigned long long)st.samples, st.payload_bytes, host, port);
    printf("  RTT ms:  min %.3f  p50 %.3f  p99 %.3f  max %.3f  mean %.3f\n",
           st.min_ms, st.p50_ms, st.p99_ms, st.max_ms, st.mean_ms);

    /* Map the measured p50 onto the §7 throughput rows so the gate is legible. */
    const char *verdict;
    if (st.p50_ms <= 0.5)      verdict = "GO (tuned row: ~26 t/s projected)";
    else if (st.p50_ms <= 1.2) verdict = "GO (untuned row: ~19 t/s projected)";
    else                       verdict = "MARGINAL: RTT high — enable jumbo MTU / check bridge0";
    printf("  gate: p50 %.3f ms -> %s\n", st.p50_ms, verdict);
    return 0;
}
