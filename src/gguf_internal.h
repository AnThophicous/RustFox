#ifndef FOX_GGUF_INTERNAL_H
#define FOX_GGUF_INTERNAL_H

#include "fox/fox.h"

typedef struct {
    char *key;
    fox_gguf_value_type type;
    uint64_t u64;
    char *string;
} fox_gguf_meta;

struct fox_gguf {
    uint32_t version;
    uint64_t file_size;
    uint64_t data_offset;
    uint32_t alignment;
    size_t n_meta;
    size_t n_tensors;
    fox_gguf_meta *meta;
    fox_gguf_tensor *tensors;
};

#endif
