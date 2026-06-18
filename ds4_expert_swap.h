#ifndef DS4_EXPERT_SWAP_H
#define DS4_EXPERT_SWAP_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Expert swap (see EXPERT_SWAP.md).
 *
 * Under SSD streaming, when the router selects an expert that is not resident
 * in the in-memory expert cache, this module decides whether to substitute a
 * resident backup expert (drawn from the top-k router window) instead of
 * paying for an SSD fetch.  The logic here is intentionally backend-agnostic
 * and free of any GPU or model dependency: a backend reads back the ranked
 * top-k expert id/score/probability window plus a cache-residency snapshot,
 * fills a small candidate array, and calls ds4_expert_swap_plan_layer().  The
 * returned plan names the experts that should actually run and which of them
 * still require a fetch.
 *
 * The feature is strictly additive: with k == n_expert_used the planner
 * reproduces baseline routing exactly.
 */

/* Largest n_expert_used we plan for (DeepSeek V4 uses 6). */
#define DS4_EXPERT_SWAP_MAX_USED 16

/* Defaults applied to a bare `--expert-swap` with no positional arguments. */
#define DS4_EXPERT_SWAP_DEFAULT_K 12u
#define DS4_EXPERT_SWAP_DEFAULT_P 0.3f

/* Parsed CLI configuration. */
typedef struct {
    bool     enabled;        /* --expert-swap was requested */
    uint32_t k;              /* router candidate window (raw; clamp at model load) */
    float    min_prob_ratio; /* candidate must retain this fraction of displaced prob */
    float    max_prob_drop;  /* optional absolute probability drop cap; 0 disables */
} ds4_expert_swap_config;

/* One ranked router-window entry, highest router score first. */
typedef struct {
    int32_t expert_id;
    float   router_score;   /* biased ranked score s(e)+bias(e) */
    float   router_prob;    /* raw router probability used for eligibility */
    bool    activated;      /* within the top-n_expert_used activation set */
    bool    resident;       /* currently resident in the expert cache */
} ds4_expert_swap_candidate;

typedef enum {
    DS4_EXPERT_SWAP_HIT = 0,    /* activated and already resident */
    DS4_EXPERT_SWAP_SUBSTITUTE, /* activated miss replaced by a resident backup */
    DS4_EXPERT_SWAP_FETCH,      /* activated miss with no eligible backup */
} ds4_expert_swap_kind;

typedef struct {
    uint32_t misses;            /* activated experts not resident */
    uint32_t swaps;             /* successful substitutions */
    uint32_t fetches;           /* misses that fell through to an SSD fetch */
    uint32_t substituted_misses;/* == swaps (routed but not fetched) */
} ds4_expert_swap_metrics;

/* Plan for one layer/token.  Slot order matches the activated set order. */
typedef struct {
    uint32_t             n_used;
    int32_t              run_id[DS4_EXPERT_SWAP_MAX_USED];    /* expert that runs */
    int32_t              routed_id[DS4_EXPERT_SWAP_MAX_USED]; /* expert routed to */
    ds4_expert_swap_kind kind[DS4_EXPERT_SWAP_MAX_USED];
    bool                 needs_fetch[DS4_EXPERT_SWAP_MAX_USED];
    ds4_expert_swap_metrics metrics;
} ds4_expert_swap_plan;

/* ---- CLI parsing ---------------------------------------------------------
 * Parse optional positional `k` that follows `--expert-swap`.  argv tokens are
 * consumed only when they are bare unsigned integers (so a following flag or
 * prompt is never swallowed).  Always succeeds: missing tokens take defaults.
 * Probability gates are controlled by DS4_EXPERT_SWAP_MIN_PROB_RATIO and
 * DS4_EXPERT_SWAP_MAX_PROB_DROP, matching the remote top-k-expert-sub math. */
void ds4_expert_swap_parse_args(const char *next1, const char *next2,
                                ds4_expert_swap_config *out, int *consumed);

/* Clamp k for a loaded model: k >= n_expert_used, k <= n_expert. */
uint32_t ds4_expert_swap_clamp_k(uint32_t k, uint32_t n_expert_used,
                                 uint32_t n_expert);

/* ---- Planner -------------------------------------------------------------
 * `window` is ranked by descending router score; the activated entries are the
 * top n_used (flagged with .activated).  Produces the final running set. */
void ds4_expert_swap_plan_layer(ds4_expert_swap_plan            *out,
                                const ds4_expert_swap_candidate *window,
                                uint32_t                         n_window,
                                uint32_t                         n_used,
                                const ds4_expert_swap_config    *config);

#endif
