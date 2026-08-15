#include "fox_internal.h"
#include "model_internal.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static fox_status meta_uint(const fox_gguf *g, const char *arch,
                            const char *suffix, uint64_t *out,
                            uint64_t fallback, int required)
{
    char key[160];
    size_t idx;

    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    idx = fox_gguf_find(g, key);
    if (idx == (size_t)-1) {
        if (required)
            return fox_fail(FOX_ERR_NOTFOUND, "model: '%s' is missing", key);
        *out = fallback;
        return FOX_OK;
    }
    if (fox_gguf_get_uint(g, idx, out) != FOX_OK)
        return fox_fail(FOX_ERR_FORMAT, "model: '%s' is not an integer", key);
    return FOX_OK;
}

static void meta_f32(const fox_gguf *g, const char *arch, const char *suffix,
                     float *out, float fallback)
{
    char key[160];
    size_t idx;

    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    idx = fox_gguf_find(g, key);
    if (idx == (size_t)-1 || fox_gguf_get_f32(g, idx, out) != FOX_OK)
        *out = fallback;
}

static size_t need_tensor(const fox_gguf *g, const char *name, fox_status *st)
{
    size_t idx = fox_gguf_tensor_find(g, name);
    if (idx == (size_t)-1) {
        *st = fox_fail(FOX_ERR_NOTFOUND, "model: tensor '%s' is missing", name);
        return FOX_TENSOR_ABSENT;
    }
    return idx;
}

static fox_status resolve_layers(fox_model *m, const fox_gguf *g)
{
    fox_status st = FOX_OK;
    char name[FOX_NAME_MAX];
    uint32_t l;

    m->layers = (fox_layer_tensors *)calloc(m->info.n_layer, sizeof(*m->layers));
    if (!m->layers) return fox_fail(FOX_ERR_NOMEM, "model: layer table");

    for (l = 0; l < m->info.n_layer && st == FOX_OK; l++) {
        fox_layer_tensors *t = &m->layers[l];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        t->attn_norm = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
        t->attn_q = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
        t->attn_k = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        t->attn_v = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        t->attn_out = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        t->ffn_norm = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
        t->ffn_gate = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        t->ffn_up = need_tensor(g, name, &st);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        t->ffn_down = need_tensor(g, name, &st);
    }
    return st;
}

static fox_status check_kernels(const fox_model *m, const fox_gguf *g)
{
    size_t i, n = fox_gguf_tensor_count(g);

    for (i = 0; i < n; i++) {
        fox_gguf_tensor t;
        if (fox_gguf_tensor_at(g, i, &t) != FOX_OK) continue;
        if (!fox_gemv_supports(t.type))
            return fox_fail(FOX_ERR_UNSUPPORTED,
                            "model: tensor '%s' is %s and this build has no "
                            "kernel for it", t.name, fox_ggml_type_name(t.type));
    }
    (void)m;
    return FOX_OK;
}

fox_status fox_model_open(const char *gguf_path, fox_weights_mode mode,
                          uint64_t budget_bytes, fox_model **out)
{
    fox_model *m;
    const fox_gguf *g;
    fox_status st;
    uint64_t v;
    size_t idx;
    const char *arch = "llama";
    size_t i, n_tensors;

    if (!gguf_path || !out) return fox_fail(FOX_ERR_ARG, "model: null argument");
    *out = NULL;

    m = (fox_model *)calloc(1, sizeof(*m));
    if (!m) return fox_fail(FOX_ERR_NOMEM, "model: handle");

    m->token_embd  = FOX_TENSOR_ABSENT;
    m->output_norm = FOX_TENSOR_ABSENT;
    m->output      = FOX_TENSOR_ABSENT;

    st = fox_weights_open(gguf_path, mode, budget_bytes, &m->weights);
    if (st != FOX_OK) { free(m); return st; }

    g = fox_weights_model(m->weights);
    m->gguf = g;

    idx = fox_gguf_find(g, "general.architecture");
    if (idx != (size_t)-1) {
        const char *s = NULL;
        if (fox_gguf_get_string(g, idx, &s) == FOX_OK && s) arch = s;
    }
    fox_strlcpy(m->info.arch, arch, sizeof(m->info.arch));

    idx = fox_gguf_find(g, "general.name");
    if (idx != (size_t)-1) {
        const char *s = NULL;
        if (fox_gguf_get_string(g, idx, &s) == FOX_OK && s)
            fox_strlcpy(m->info.name, s, sizeof(m->info.name));
    }
    if (!m->info.name[0]) fox_strlcpy(m->info.name, "unnamed", sizeof(m->info.name));

    st = meta_uint(g, arch, "block_count", &v, 0, 1);
    if (st != FOX_OK) goto fail;
    m->info.n_layer = (uint32_t)v;

    st = meta_uint(g, arch, "embedding_length", &v, 0, 1);
    if (st != FOX_OK) goto fail;
    m->info.n_embd = (uint32_t)v;

    st = meta_uint(g, arch, "attention.head_count", &v, 0, 1);
    if (st != FOX_OK) goto fail;
    m->info.n_head = (uint32_t)v;

    st = meta_uint(g, arch, "attention.head_count_kv", &v, m->info.n_head, 0);
    if (st != FOX_OK) goto fail;
    m->info.n_head_kv = (uint32_t)v;

    st = meta_uint(g, arch, "feed_forward_length", &v, 0, 1);
    if (st != FOX_OK) goto fail;
    m->info.n_ff = (uint32_t)v;

    st = meta_uint(g, arch, "context_length", &v, 2048, 0);
    if (st != FOX_OK) goto fail;
    m->info.n_ctx_train = (uint32_t)v;

    st = meta_uint(g, arch, "attention.key_length", &v, 0, 0);
    if (st != FOX_OK) goto fail;
    m->info.head_dim = (uint32_t)v;

    if (m->info.n_layer == 0 || m->info.n_embd == 0 || m->info.n_head == 0 ||
        m->info.n_ff == 0 || m->info.n_head_kv == 0) {
        st = fox_fail(FOX_ERR_FORMAT, "model: architecture metadata is incomplete");
        goto fail;
    }
    if (m->info.head_dim == 0) m->info.head_dim = m->info.n_embd / m->info.n_head;
    if (m->info.head_dim == 0 || m->info.head_dim % 2 != 0) {
        st = fox_fail(FOX_ERR_UNSUPPORTED,
                      "model: head dimension %u is not usable by the rope "
                      "implementation, which rotates adjacent pairs",
                      m->info.head_dim);
        goto fail;
    }
    if (m->info.n_head % m->info.n_head_kv != 0) {
        st = fox_fail(FOX_ERR_FORMAT,
                      "model: %u query heads do not divide evenly into %u kv "
                      "heads", m->info.n_head, m->info.n_head_kv);
        goto fail;
    }

    meta_f32(g, arch, "attention.layer_norm_rms_epsilon", &m->info.rms_eps, 1e-5f);
    meta_f32(g, arch, "rope.freq_base", &m->info.rope_freq_base, 10000.0f);

    st = check_kernels(m, g);
    if (st != FOX_OK) goto fail;

    st = FOX_OK;
    m->token_embd = need_tensor(g, "token_embd.weight", &st);
    if (st != FOX_OK) goto fail;
    m->output_norm = need_tensor(g, "output_norm.weight", &st);
    if (st != FOX_OK) goto fail;

    m->output = fox_gguf_tensor_find(g, "output.weight");
    if (m->output == (size_t)-1) {
        m->output = m->token_embd;
        m->info.tied_output = 1;
    }

    {
        fox_gguf_tensor t;
        if (fox_gguf_tensor_at(g, m->token_embd, &t) != FOX_OK || t.n_dims < 2) {
            st = fox_fail(FOX_ERR_FORMAT, "model: token_embd.weight is malformed");
            goto fail;
        }
        m->type_embd = t.type;
        m->info.n_vocab = (uint32_t)t.ne[1];
        if (t.ne[0] != m->info.n_embd) {
            st = fox_fail(FOX_ERR_FORMAT,
                          "model: token_embd rows are %llu wide but the model "
                          "says the embedding is %u",
                          (unsigned long long)t.ne[0], m->info.n_embd);
            goto fail;
        }
        if (fox_gguf_tensor_at(g, m->output, &t) == FOX_OK) m->type_output = t.type;
    }

    st = resolve_layers(m, g);
    if (st != FOX_OK) goto fail;

    n_tensors = fox_gguf_tensor_count(g);
    for (i = 0; i < n_tensors; i++) {
        fox_gguf_tensor t;
        if (fox_gguf_tensor_at(g, i, &t) == FOX_OK)
            m->info.weight_bytes += t.size_bytes;
    }

    if (fox_tokenizer_open(g, &m->tokenizer) != FOX_OK) {
        FOX_WARN("model: no usable tokenizer metadata (%s); "
                 "text in and out will not work", fox_last_error());
        m->tokenizer = NULL;
    }

    *out = m;
    return FOX_OK;

fail:
    fox_model_close(m);
    return st;
}

void fox_model_close(fox_model *m)
{
    if (!m) return;
    if (m->tokenizer) fox_tokenizer_close(m->tokenizer);
    free(m->layers);
    if (m->weights) fox_weights_close(m->weights);
    free(m);
}

const fox_model_info *fox_model_get_info(const fox_model *m)
{
    return m ? &m->info : NULL;
}

const fox_tokenizer *fox_model_tokenizer(const fox_model *m)
{
    return m ? m->tokenizer : NULL;
}

fox_weights *fox_model_weights(fox_model *m)
{
    return m ? m->weights : NULL;
}
