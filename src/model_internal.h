#ifndef FOX_MODEL_INTERNAL_H
#define FOX_MODEL_INTERNAL_H

#include "fox/fox.h"

#define FOX_TENSOR_ABSENT ((size_t)-1)

typedef struct {
    size_t attn_norm;
    size_t attn_q;
    size_t attn_k;
    size_t attn_v;
    size_t attn_out;
    size_t ffn_norm;
    size_t ffn_gate;
    size_t ffn_up;
    size_t ffn_down;
} fox_layer_tensors;

struct fox_model {
    fox_model_info      info;
    fox_weights        *weights;
    fox_tokenizer      *tokenizer;
    const fox_gguf     *gguf;
    fox_layer_tensors  *layers;
    size_t              token_embd;
    size_t              output_norm;
    size_t              output;
    fox_ggml_type       type_embd;
    fox_ggml_type       type_output;
};

void  fox_rmsnorm(float *out, const float *x, const float *weight,
                  size_t n, float eps);
void  fox_softmax(float *x, size_t n);
void  fox_silu_mul(float *gate, const float *up, size_t n);
void  fox_add(float *dst, const float *src, size_t n);
void  fox_rope(float *vec, size_t n_heads, size_t head_dim,
               const float *cos_row, const float *sin_row);
float fox_dot_f32(const float *a, const float *b, size_t n);

#endif
