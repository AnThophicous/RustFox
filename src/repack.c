#include "fox/fox.h"
#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPACK_COPY_BYTES (1u << 20)

typedef struct {
    size_t   tensor_index;
    int      rank;
    uint32_t layer;
    int      role;
} repack_item;

static int tensor_role(const char *name, uint32_t *layer, int *role)
{
    const char *p;
    uint64_t value = 0;

    if (strcmp(name, "token_embd.weight") == 0) {
        *layer = 0;
        *role = 0;
        return 1;
    }
    if (strcmp(name, "output_norm.weight") == 0) {
        *layer = 0;
        *role = 0;
        return 2;
    }
    if (strcmp(name, "output.weight") == 0) {
        *layer = 0;
        *role = 0;
        return 3;
    }
    if (strncmp(name, "blk.", 4) != 0) return 0;

    p = name + 4;
    if (*p < '0' || *p > '9') return 0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10u + (uint64_t)(*p - '0');
        if (value > UINT32_MAX) return 0;
        p++;
    }
    if (*p != '.') return 0;

    *layer = (uint32_t)value;
    if (strcmp(p + 1, "attn_norm.weight") == 0) *role = 0;
    else if (strcmp(p + 1, "attn_q.weight") == 0) *role = 1;
    else if (strcmp(p + 1, "attn_k.weight") == 0) *role = 2;
    else if (strcmp(p + 1, "attn_v.weight") == 0) *role = 3;
    else if (strcmp(p + 1, "attn_output.weight") == 0) *role = 4;
    else if (strcmp(p + 1, "ffn_norm.weight") == 0) *role = 5;
    else if (strcmp(p + 1, "ffn_gate.weight") == 0) *role = 6;
    else if (strcmp(p + 1, "ffn_up.weight") == 0) *role = 7;
    else if (strcmp(p + 1, "ffn_down.weight") == 0) *role = 8;
    else *role = 100;
    return 1;
}

static void make_item(const fox_gguf_tensor *tensor, size_t index,
                      repack_item *item)
{
    uint32_t layer;
    int role;

    item->tensor_index = index;
    item->rank = 4;
    item->layer = 0;
    item->role = 0;

    if (!tensor_role(tensor->name, &layer, &role)) return;

    item->layer = layer;
    item->role = role;
    if (strcmp(tensor->name, "token_embd.weight") == 0) item->rank = 0;
    else if (strncmp(tensor->name, "blk.", 4) == 0) item->rank = 1;
    else if (strcmp(tensor->name, "output_norm.weight") == 0) item->rank = 2;
    else if (strcmp(tensor->name, "output.weight") == 0) item->rank = 3;
}

static int item_compare(const void *a, const void *b)
{
    const repack_item *x = (const repack_item *)a;
    const repack_item *y = (const repack_item *)b;

    if (x->rank != y->rank) return x->rank < y->rank ? -1 : 1;
    if (x->rank == 1 && x->layer != y->layer)
        return x->layer < y->layer ? -1 : 1;
    if (x->role != y->role) return x->role < y->role ? -1 : 1;
    if (x->tensor_index != y->tensor_index)
        return x->tensor_index < y->tensor_index ? -1 : 1;
    return 0;
}

static int read_exact(fox_file *file, void *dst, size_t bytes, uint64_t offset)
{
    uint8_t *p = (uint8_t *)dst;
    size_t done = 0;

    while (done < bytes) {
        int64_t got = fox_file_pread(file, p + done, bytes - done,
                                     offset + (uint64_t)done);
        if (got <= 0) return 0;
        done += (size_t)got;
    }
    return 1;
}

static void put_u64(uint8_t *p, uint64_t value)
{
    int i;
    for (i = 0; i < 8; i++) p[i] = (uint8_t)(value >> (8 * i));
}

static int align_cursor(uint64_t value, uint32_t alignment, uint64_t *out)
{
    uint64_t mask = (uint64_t)alignment - 1;
    uint64_t aligned = (value + mask) & ~mask;

    if (aligned < value) return 0;
    *out = aligned;
    return 1;
}

static int write_zeros(FILE *out, uint64_t bytes)
{
    static const uint8_t zeros[65536] = { 0 };

    while (bytes) {
        size_t chunk = bytes > sizeof(zeros) ? sizeof(zeros) : (size_t)bytes;
        if (fwrite(zeros, 1, chunk, out) != chunk) return 0;
        bytes -= chunk;
    }
    return 1;
}

static fox_status copy_tensor(fox_file *source, FILE *out,
                              const fox_gguf *model,
                              const fox_gguf_tensor *tensor,
                              uint8_t *buffer)
{
    uint64_t source_offset;
    uint64_t remaining = tensor->size_bytes;

    if (tensor->offset > UINT64_MAX - model->data_offset)
        return fox_fail(FOX_ERR_FORMAT, "repack: tensor '%s' offset overflows",
                        tensor->name);
    source_offset = model->data_offset + tensor->offset;

    while (remaining) {
        size_t chunk = remaining > REPACK_COPY_BYTES
                     ? REPACK_COPY_BYTES : (size_t)remaining;
        if (!read_exact(source, buffer, chunk, source_offset))
            return fox_fail(FOX_ERR_IO, "repack: cannot read tensor '%s'",
                            tensor->name);
        if (fwrite(buffer, 1, chunk, out) != chunk)
            return fox_fail(FOX_ERR_IO, "repack: cannot write tensor '%s'",
                            tensor->name);
        source_offset += chunk;
        remaining -= chunk;
    }
    return FOX_OK;
}

fox_status fox_repack_gguf(const char *input_path, const char *output_path)
{
    fox_gguf *model = NULL;
    fox_gguf *check = NULL;
    fox_file *source = NULL;
    FILE *out = NULL;
    repack_item *items = NULL;
    uint64_t *new_offsets = NULL;
    uint8_t *header = NULL;
    uint8_t *buffer = NULL;
    char *part_path = NULL;
    size_t n, i;
    uint64_t cursor = 0;
    uint64_t written = 0;
    size_t header_bytes;
    int renamed = 0;
    fox_status st = FOX_OK;

    if (!input_path || !output_path)
        return fox_fail(FOX_ERR_ARG, "repack: null input or output path");
    if (strcmp(input_path, output_path) == 0)
        return fox_fail(FOX_ERR_ARG, "repack: input and output must differ");

    st = fox_gguf_open(input_path, &model);
    if (st != FOX_OK) return st;
    if (model->data_offset > SIZE_MAX) {
        st = fox_fail(FOX_ERR_FORMAT, "repack: GGUF header is too large");
        goto done;
    }

    n = model->n_tensors;
    header_bytes = (size_t)model->data_offset;
    header = (uint8_t *)malloc(header_bytes ? header_bytes : 1);
    items = (repack_item *)calloc(n ? n : 1, sizeof(*items));
    new_offsets = (uint64_t *)calloc(n ? n : 1, sizeof(*new_offsets));
    buffer = (uint8_t *)malloc(REPACK_COPY_BYTES);
    part_path = (char *)malloc(strlen(output_path) + 6);
    if (!header || !items || !new_offsets || !buffer || !part_path) {
        st = fox_fail(FOX_ERR_NOMEM, "repack: allocation failed");
        goto done;
    }

    source = fox_file_open_read(input_path, FOX_OPEN_RANDOM, NULL);
    if (!source || !read_exact(source, header, header_bytes, 0)) {
        st = fox_fail(FOX_ERR_IO, "repack: cannot read GGUF header");
        goto done;
    }

    for (i = 0; i < n; i++) {
        fox_gguf_tensor tensor;
        if (fox_gguf_tensor_at(model, i, &tensor) != FOX_OK) {
            st = fox_fail(FOX_ERR_FORMAT, "repack: cannot inspect tensor %llu",
                          (unsigned long long)i);
            goto done;
        }
        make_item(&tensor, i, &items[i]);
    }
    qsort(items, n, sizeof(*items), item_compare);

    for (i = 0; i < n; i++) {
        fox_gguf_tensor tensor;
        uint64_t aligned;

        if (!align_cursor(cursor, model->alignment, &aligned)) {
            st = fox_fail(FOX_ERR_FORMAT, "repack: output offset overflows");
            goto done;
        }
        if (fox_gguf_tensor_at(model, items[i].tensor_index, &tensor) != FOX_OK ||
            tensor.size_bytes > UINT64_MAX - aligned) {
            st = fox_fail(FOX_ERR_FORMAT, "repack: tensor size overflows");
            goto done;
        }
        new_offsets[items[i].tensor_index] = aligned;
        cursor = aligned + tensor.size_bytes;
    }

    for (i = 0; i < n; i++) {
        uint64_t position = model->tensor_offset_pos[i];
        if (position > model->data_offset || model->data_offset - position < 8) {
            st = fox_fail(FOX_ERR_FORMAT,
                          "repack: tensor offset is outside the GGUF header");
            goto done;
        }
        put_u64(header + (size_t)position, new_offsets[i]);
    }

    snprintf(part_path, strlen(output_path) + 6, "%s.part", output_path);
    out = fopen(part_path, "wb");
    if (!out) {
        st = fox_fail(FOX_ERR_IO, "repack: cannot create '%s'", part_path);
        goto done;
    }
    if (header_bytes && fwrite(header, 1, header_bytes, out) != header_bytes) {
        st = fox_fail(FOX_ERR_IO, "repack: cannot write GGUF header");
        goto done;
    }

    for (i = 0; i < n; i++) {
        fox_gguf_tensor tensor;
        uint64_t offset = new_offsets[items[i].tensor_index];

        if (offset < written || !write_zeros(out, offset - written) ||
            fox_gguf_tensor_at(model, items[i].tensor_index, &tensor) != FOX_OK) {
            st = fox_fail(FOX_ERR_IO, "repack: cannot position tensor payload");
            goto done;
        }
        st = copy_tensor(source, out, model, &tensor, buffer);
        if (st != FOX_OK) goto done;
        written = offset + tensor.size_bytes;
    }

    if (fflush(out) != 0 || fclose(out) != 0) {
        out = NULL;
        st = fox_fail(FOX_ERR_IO, "repack: cannot finish '%s'", part_path);
        goto done;
    }
    out = NULL;

    if (rename(part_path, output_path) != 0) {
        if ((errno != EEXIST && errno != EACCES) ||
            (remove(output_path) != 0 && errno != ENOENT) ||
            rename(part_path, output_path) != 0) {
            st = fox_fail(FOX_ERR_IO, "repack: cannot replace '%s'", output_path);
            goto done;
        }
    }
    renamed = 1;

    st = fox_gguf_open(output_path, &check);
    if (st != FOX_OK)
        st = fox_fail(FOX_ERR_FORMAT, "repack: output validation failed: %s",
                      fox_last_error());

done:
    if (out) fclose(out);
    if (!renamed && part_path) remove(part_path);
    fox_gguf_close(check);
    fox_file_close(source);
    fox_gguf_close(model);
    free(part_path);
    free(buffer);
    free(header);
    free(new_offsets);
    free(items);
    return st;
}
