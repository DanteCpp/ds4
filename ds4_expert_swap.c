#include "ds4_expert_swap.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- small helpers -------------------------------------------------------*/

static bool token_is_uint(const char *s) {
    if (!s || !s[0]) return false;
    for (const char *ch = s; *ch; ch++) {
        if (!isdigit((unsigned char)*ch)) return false;
    }
    return true;
}

static float env_float2(const char *name1, const char *name2,
                        float def, float min, float max) {
    const char *env = getenv(name1);
    if (!env || !env[0]) env = getenv(name2);
    if (!env || !env[0]) return def;
    char *end = NULL;
    float v = strtof(env, &end);
    if (end == env || *end != '\0' || v < min || v > max) return def;
    return v;
}

static bool prob_allowed(float cur_prob, float cand_prob,
                         const ds4_expert_swap_config *config) {
    if (!(cand_prob >= 0.0f) || !(cur_prob >= 0.0f)) return false;
    const float min_prob_ratio = config ? config->min_prob_ratio : 0.50f;
    if (min_prob_ratio <= 0.0f) return false;
    if (cand_prob >= cur_prob) return true;
    if (min_prob_ratio > 0.0f && cand_prob < cur_prob * min_prob_ratio) return false;
    const float max_prob_drop = config ? config->max_prob_drop : 0.0f;
    if (max_prob_drop > 0.0f && cur_prob - cand_prob > max_prob_drop) return false;
    return true;
}

/* ---- CLI parsing ---------------------------------------------------------*/

void ds4_expert_swap_parse_args(const char *next1, const char *next2,
                                ds4_expert_swap_config *out, int *consumed) {
    (void)next2;
    if (consumed) *consumed = 0;
    if (!out) return;
    out->enabled = true;
    out->k = DS4_EXPERT_SWAP_DEFAULT_K;
    out->min_prob_ratio =
        env_float2("DS4_EXPERT_SWAP_MIN_PROB_RATIO",
                   "DS4_METAL_TOP_K_EXPERT_SUB_MIN_PROB_RATIO",
                   0.50f, 0.0f, 1.0f);
    out->max_prob_drop =
        env_float2("DS4_EXPERT_SWAP_MAX_PROB_DROP",
                   "DS4_METAL_TOP_K_EXPERT_SUB_MAX_PROB_DROP",
                   0.0f, 0.0f, 1.0e30f);

    if (token_is_uint(next1)) {
        out->k = (uint32_t)strtoul(next1, NULL, 10);
        if (consumed) *consumed = 1;
    }
}

uint32_t ds4_expert_swap_clamp_k(uint32_t k, uint32_t n_expert_used,
                                 uint32_t n_expert) {
    if (k < n_expert_used) k = n_expert_used;
    if (n_expert != 0 && k > n_expert) k = n_expert;
    return k;
}

/* ---- planner -------------------------------------------------------------*/

void ds4_expert_swap_plan_layer(ds4_expert_swap_plan            *out,
                                const ds4_expert_swap_candidate *window,
                                uint32_t                         n_window,
                                uint32_t                         n_used,
                                const ds4_expert_swap_config    *config) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!window || n_used == 0) return;
    if (n_used > DS4_EXPERT_SWAP_MAX_USED) n_used = DS4_EXPERT_SWAP_MAX_USED;

    float slot_prob[DS4_EXPERT_SWAP_MAX_USED];
    bool  slot_miss[DS4_EXPERT_SWAP_MAX_USED];

    /* Fill activated slots in ranked router order. */
    uint32_t slot = 0;
    for (uint32_t i = 0; i < n_window && slot < n_used; i++) {
        if (!window[i].activated) continue;
        out->routed_id[slot] = window[i].expert_id;
        slot_prob[slot] = window[i].router_prob;
        if (window[i].resident) {
            out->kind[slot] = DS4_EXPERT_SWAP_HIT;
            out->run_id[slot] = window[i].expert_id;
            out->needs_fetch[slot] = false;
            slot_miss[slot] = false;
        } else {
            out->kind[slot] = DS4_EXPERT_SWAP_FETCH;
            out->run_id[slot] = window[i].expert_id;
            out->needs_fetch[slot] = true;
            slot_miss[slot] = true;
            out->metrics.misses++;
        }
        slot++;
    }
    out->n_used = slot;

    uint32_t cand_idx[DS4_EXPERT_SWAP_MAX_USED * 8];
    bool     cand_used[DS4_EXPERT_SWAP_MAX_USED * 8];
    uint32_t n_cand = 0;
    const uint32_t cand_cap = (uint32_t)(sizeof(cand_idx) / sizeof(cand_idx[0]));
    for (uint32_t i = 0; i < n_window && n_cand < cand_cap; i++) {
        if (window[i].activated || !window[i].resident) continue;
        cand_idx[n_cand] = i;
        cand_used[n_cand] = false;
        n_cand++;
    }

    /* Greedy remote top-k-expert-sub behavior: each missed routed slot takes
     * the highest-ranked unused resident candidate whose raw probability drop
     * passes the configured gates. */
    for (uint32_t s = 0; s < out->n_used; s++) {
        if (!slot_miss[s]) continue;
        int best = -1;
        for (uint32_t c = 0; c < n_cand; c++) {
            if (cand_used[c]) continue;
            const ds4_expert_swap_candidate *cand = &window[cand_idx[c]];
            if (!prob_allowed(slot_prob[s], cand->router_prob, config)) continue;
            best = (int)c;
            break;
        }

        if (best >= 0) {
            cand_used[best] = true;
            out->kind[s] = DS4_EXPERT_SWAP_SUBSTITUTE;
            out->run_id[s] = window[cand_idx[best]].expert_id;
            out->needs_fetch[s] = false;
            out->metrics.swaps++;
        } else {
            out->metrics.fetches++;
        }
    }
    out->metrics.substituted_misses = out->metrics.swaps;
}
