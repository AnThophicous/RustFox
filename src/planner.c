#include "fox/fox.h"
#include "fox_internal.h"
#include "planner_internal.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static void lower_name(const char *src, char *dst, size_t cap)
{
    size_t i;
    if (cap == 0) return;
    for (i = 0; i + 1 < cap && src[i]; i++)
        dst[i] = (char)tolower((unsigned char)src[i]);
    dst[i] = '\0';
}

static uint32_t tensor_priority(const char *name, fox_plan_placement *placement,
                                const char **reason)
{
    char lower[256];
    lower_name(name, lower, sizeof(lower));
    if (strstr(lower, "tok_embd") || strstr(lower, "token_emb") ||
        strstr(lower, "embed_tokens")) {
        *placement = FOX_PLAN_MMAP;
        *reason = "sparse token embedding";
        return 0;
    }
    if (strstr(lower, "norm") || strstr(lower, "rope") ||
        strstr(lower, "attn") || strstr(lower, "attention") ||
        strstr(lower, "q_proj") || strstr(lower, "k_proj") ||
        strstr(lower, "v_proj") || strstr(lower, "o_proj") ||
        strstr(lower, "output")) {
        *placement = FOX_PLAN_PIN;
        *reason = "hot per-token tensor";
        return 3;
    }
    if (strstr(lower, "ffn") || strstr(lower, "gate") ||
        strstr(lower, "up_proj") || strstr(lower, "down_proj") ||
        strstr(lower, "expert")) {
        *placement = FOX_PLAN_STREAM;
        *reason = "large sequential projection";
        return 1;
    }
    *placement = FOX_PLAN_STREAM;
    *reason = "default sequential placement";
    return 2;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (b > UINT64_MAX - a) return 0;
    *out = a + b;
    return 1;
}

fox_status fox_plan_build(const fox_gguf *model, const fox_plan_config *config,
                          fox_plan **out)
{
    fox_plan *plan;
    fox_plan_config cfg;
    size_t i, j, best;
    fox_gguf_tensor tensor;
    uint64_t resident = 0;
    double token_seconds;

    if (!model || !config || !out) return FOX_ERR_ARG;
    *out = NULL;
    if (config->seq_read_mbps < 0.0 || config->compute_tok_s < 0.0)
        return FOX_ERR_ARG;
    cfg = *config;
    plan = (fox_plan *)calloc(1, sizeof(*plan));
    if (!plan) return FOX_ERR_NOMEM;
    plan->count = fox_gguf_tensor_count(model);
    plan->items = (fox_plan_item *)calloc(plan->count, sizeof(*plan->items));
    if (plan->count && !plan->items) { free(plan); return FOX_ERR_NOMEM; }

    for (i = 0; i < plan->count; i++) {
        const char *reason;
        fox_plan_placement initial;
        if (fox_gguf_tensor_at(model, i, &tensor) != FOX_OK) {
            fox_plan_destroy(plan);
            return FOX_ERR_FORMAT;
        }
        plan->items[i].tensor_index = i;
        plan->items[i].bytes = tensor.size_bytes;
        plan->items[i].priority = tensor_priority(tensor.name, &initial, &reason);
        plan->items[i].placement = initial == FOX_PLAN_MMAP ? FOX_PLAN_MMAP : FOX_PLAN_STREAM;
        plan->items[i].reason = initial == FOX_PLAN_MMAP ? reason : "awaits residency selection";
        if (!add_u64(plan->summary.model_bytes, tensor.size_bytes,
                     &plan->summary.model_bytes)) {
            fox_plan_destroy(plan);
            return FOX_ERR_FORMAT;
        }
        if (initial == FOX_PLAN_MMAP &&
            !add_u64(plan->summary.mmap_bytes, tensor.size_bytes,
                     &plan->summary.mmap_bytes)) {
            fox_plan_destroy(plan);
            return FOX_ERR_FORMAT;
        }
    }

    for (;;) {
        best = SIZE_MAX;
        for (j = 0; j < plan->count; j++) {
            if (plan->items[j].placement != FOX_PLAN_STREAM ||
                plan->items[j].priority < 3) continue;
            if (best == SIZE_MAX || plan->items[j].priority > plan->items[best].priority)
                best = j;
        }
        if (best == SIZE_MAX) break;
        if (plan->items[best].bytes <= cfg.residency_budget_bytes - resident) {
            plan->items[best].placement = FOX_PLAN_PIN;
            plan->items[best].reason = "hot tensor fits residency budget";
            resident += plan->items[best].bytes;
            plan->summary.pinned_bytes = resident;
        } else {
            plan->items[best].priority = 0;
        }
    }

    for (i = 0; i < plan->count; i++) {
        if (plan->items[i].placement == FOX_PLAN_STREAM &&
            !add_u64(plan->summary.streamed_bytes, plan->items[i].bytes,
                     &plan->summary.streamed_bytes)) {
            fox_plan_destroy(plan);
            return FOX_ERR_FORMAT;
        }
    }
    plan->summary.compute_seconds = cfg.compute_tok_s > 0.0 ? 1.0 / cfg.compute_tok_s : 0.0;
    if (cfg.seq_read_mbps > 0.0)
        plan->summary.io_seconds = (double)plan->summary.streamed_bytes /
            (cfg.seq_read_mbps * 1024.0 * 1024.0);
    token_seconds = plan->summary.compute_seconds > plan->summary.io_seconds ?
        plan->summary.compute_seconds : plan->summary.io_seconds;
    plan->summary.predicted_tok_s = token_seconds > 0.0 ? 1.0 / token_seconds : 0.0;
    *out = plan;
    return FOX_OK;
}

void fox_plan_destroy(fox_plan *plan)
{
    if (!plan) return;
    free(plan->items);
    free(plan);
}

size_t fox_plan_count(const fox_plan *plan)
{
    return plan ? plan->count : 0;
}

fox_status fox_plan_item_at(const fox_plan *plan, size_t index, fox_plan_item *out)
{
    if (!plan || !out) return FOX_ERR_ARG;
    if (index >= plan->count) return FOX_ERR_NOTFOUND;
    *out = plan->items[index];
    return FOX_OK;
}

fox_status fox_plan_get_summary(const fox_plan *plan, fox_plan_summary *out)
{
    if (!plan || !out) return FOX_ERR_ARG;
    *out = plan->summary;
    return FOX_OK;
}
