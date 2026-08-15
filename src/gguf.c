#include "fox/fox.h"
#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define GGUF_MAX_ITEMS  1000000u
#define GGUF_MAX_META   65536u
#define GGUF_MAX_STRING (16u * 1024u * 1024u)
#define GGUF_MAX_ALIGN  4096u

typedef struct {
    fox_file *f;
    uint64_t  pos;
    uint64_t  size;
} reader;

static int add_ok(uint64_t a, uint64_t b, uint64_t *r)
{
    if (b > UINT64_MAX - a) return 0;
    *r = a + b;
    return 1;
}

static int mul_ok(uint64_t a, uint64_t b, uint64_t *r)
{
    if (a && b > UINT64_MAX / a) return 0;
    *r = a * b;
    return 1;
}

static int read_bytes(reader *r, void *p, size_t n)
{
    if ((uint64_t)n > r->size - r->pos) return 0;
    if (fox_file_pread(r->f, p, n, r->pos) != (int64_t)n) return 0;
    r->pos += n;
    return 1;
}


static int u32(reader *r, uint32_t *v)
{
    uint8_t b[4];
    if (!read_bytes(r, b, 4)) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int u64(reader *r, uint64_t *v)
{
    uint8_t b[8];
    int i;
    if (!read_bytes(r, b, 8)) return 0;
    *v = 0;
    for (i = 0; i < 8; i++) *v |= (uint64_t)b[i] << (8 * i);
    return 1;
}

static int i32(reader *r, int32_t *v)
{
    uint32_t x;
    if (!u32(r, &x)) return 0;
    *v = (int32_t)x;
    return 1;
}

static int value_size(fox_gguf_value_type t, uint64_t *n)
{
    static const uint8_t sizes[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

    if (t > FOX_GGUF_FLOAT64) return 0;
    if (t == FOX_GGUF_STRING || t == FOX_GGUF_ARRAY) return 0;
    *n = sizes[t];
    return 1;
}

static int scalar(reader *r, fox_gguf_value_type t, uint64_t *out)
{
    uint8_t raw[8] = { 0 };
    uint64_t n;
    uint64_t i;

    if (!value_size(t, &n)) return 0;
    if (!read_bytes(r, raw, (size_t)n)) return 0;

    *out = 0;
    for (i = 0; i < n; i++) *out |= (uint64_t)raw[i] << (8 * i);
    return 1;
}

static int str_read(reader *r, char **out, fox_status *rc)
{
    uint64_t n;
    char *s;

    if (!u64(r, &n)) return 0;
    if (n > GGUF_MAX_STRING || n > r->size - r->pos || n > SIZE_MAX - 1) return 0;

    s = (char *)malloc((size_t)n + 1);
    if (!s) {
        *rc = FOX_ERR_NOMEM;
        return 0;
    }
    if (!read_bytes(r, s, (size_t)n)) {
        free(s);
        return 0;
    }
    s[n] = '\0';
    *out = s;
    return 1;
}

static void free_array(fox_gguf_meta *m)
{
    size_t i;
    if (m->array_strings) {
        for (i = 0; i < m->array_count; i++) free(m->array_strings[i]);
        free(m->array_strings);
    }
    free(m->array_values);
    m->array_strings = NULL;
    m->array_values = NULL;
}

static int read_array(reader *r, fox_gguf_meta *m, fox_status *rc)
{
    int32_t elem;
    uint64_t count, k, value;
    uint64_t size;

    if (!i32(r, &elem) || elem < 0 || elem > FOX_GGUF_FLOAT64) return 0;
    if (!u64(r, &count) || count > GGUF_MAX_ITEMS || count > SIZE_MAX) return 0;
    m->array_type = (fox_gguf_value_type)elem;
    m->array_count = (size_t)count;

    if ((fox_gguf_value_type)elem == FOX_GGUF_STRING) {
        m->array_strings = (char **)calloc(m->array_count, sizeof(*m->array_strings));
        if (m->array_count && !m->array_strings) { *rc = FOX_ERR_NOMEM; return 0; }
        for (k = 0; k < count; k++) {
            if (!str_read(r, &m->array_strings[k], rc)) { free_array(m); return 0; }
        }
        return 1;
    }

    if ((fox_gguf_value_type)elem == FOX_GGUF_ARRAY) return 0;

    if (!value_size((fox_gguf_value_type)elem, &size)) return 0;
    m->array_values = (uint64_t *)calloc(m->array_count, sizeof(*m->array_values));
    if (m->array_count && !m->array_values) { *rc = FOX_ERR_NOMEM; return 0; }
    for (k = 0; k < count; k++) {
        if (!scalar(r, (fox_gguf_value_type)elem, &value)) { free_array(m); return 0; }
        m->array_values[k] = value;
    }
    return 1;
}

static int tensor_layout(fox_ggml_type t, uint32_t *block, uint32_t *bytes)
{
    switch (t) {
    case FOX_GGML_F32:   *block = 1;   *bytes = 4;   return 1;
    case FOX_GGML_F16:   *block = 1;   *bytes = 2;   return 1;
    case FOX_GGML_BF16:  *block = 1;   *bytes = 2;   return 1;
    case FOX_GGML_I8:    *block = 1;   *bytes = 1;   return 1;
    case FOX_GGML_I16:   *block = 1;   *bytes = 2;   return 1;
    case FOX_GGML_I32:   *block = 1;   *bytes = 4;   return 1;
    case FOX_GGML_I64:   *block = 1;   *bytes = 8;   return 1;
    case FOX_GGML_F64:   *block = 1;   *bytes = 8;   return 1;
    case FOX_GGML_Q4_0:  *block = 32;  *bytes = 18;  return 1;
    case FOX_GGML_Q4_1:  *block = 32;  *bytes = 20;  return 1;
    case FOX_GGML_Q5_0:  *block = 32;  *bytes = 22;  return 1;
    case FOX_GGML_Q5_1:  *block = 32;  *bytes = 24;  return 1;
    case FOX_GGML_Q8_0:  *block = 32;  *bytes = 34;  return 1;
    case FOX_GGML_Q8_K:  *block = 256; *bytes = 292; return 1;
    case FOX_GGML_Q2_K:  *block = 256; *bytes = 84;  return 1;
    case FOX_GGML_Q3_K:  *block = 256; *bytes = 110; return 1;
    case FOX_GGML_Q4_K:  *block = 256; *bytes = 144; return 1;
    case FOX_GGML_Q5_K:  *block = 256; *bytes = 176; return 1;
    case FOX_GGML_Q6_K:  *block = 256; *bytes = 210; return 1;
    case FOX_GGML_IQ2_XXS: *block = 256; *bytes = 66;  return 1;
    case FOX_GGML_IQ2_XS:  *block = 256; *bytes = 74;  return 1;
    case FOX_GGML_IQ3_XXS: *block = 256; *bytes = 98;  return 1;
    case FOX_GGML_IQ1_S:   *block = 256; *bytes = 50;  return 1;
    case FOX_GGML_IQ4_NL:  *block = 32;  *bytes = 18;  return 1;
    case FOX_GGML_IQ3_S:   *block = 256; *bytes = 110; return 1;
    case FOX_GGML_IQ2_S:   *block = 256; *bytes = 82;  return 1;
    case FOX_GGML_IQ4_XS:  *block = 256; *bytes = 136; return 1;
    case FOX_GGML_IQ1_M:   *block = 256; *bytes = 56;  return 1;
    case FOX_GGML_TQ1_0: *block = 256; *bytes = 54;  return 1;
    case FOX_GGML_TQ2_0: *block = 256; *bytes = 66;  return 1;
    default: return 0;
    }
}

static void dispose(fox_gguf *g)
{
    size_t i;

    if (!g) return;

    if (g->meta) {
        for (i = 0; i < g->n_meta; i++) {
            free(g->meta[i].key);
            free(g->meta[i].string);
            free_array(&g->meta[i]);
        }
        free(g->meta);
    }
    if (g->tensors) {
        for (i = 0; i < g->n_tensors; i++) free((char *)g->tensors[i].name);
        free(g->tensors);
    }
    free(g->path);
    free(g);
}

static fox_status read_metadata(reader *r, fox_gguf *g, uint32_t *align)
{
    size_t i, j;

    for (i = 0; i < g->n_meta; i++) {
        fox_gguf_meta *m = &g->meta[i];
        fox_status rc = FOX_ERR_FORMAT;
        int32_t type;

        if (!str_read(r, &m->key, &rc))
            return fox_fail(rc, "gguf: bad metadata key at index %llu",
                            (unsigned long long)i);

        for (j = 0; j < i; j++) {
            if (strcmp(m->key, g->meta[j].key) == 0)
                return fox_fail(FOX_ERR_FORMAT,
                                "gguf: duplicate metadata key '%s'", m->key);
        }

        if (!i32(r, &type) || type < 0 || type > FOX_GGUF_FLOAT64)
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: bad value type for key '%s'", m->key);
        m->type = (fox_gguf_value_type)type;

        if (m->type == FOX_GGUF_STRING) {
            rc = FOX_ERR_FORMAT;
            if (!str_read(r, &m->string, &rc))
                return fox_fail(rc, "gguf: bad string value for key '%s'", m->key);
        } else if (m->type == FOX_GGUF_ARRAY) {
            if (!read_array(r, m, &rc))
                return fox_fail(rc,
                                "gguf: bad array for key '%s'", m->key);
        } else {
            if (!scalar(r, m->type, &m->u64))
                return fox_fail(FOX_ERR_FORMAT,
                                "gguf: bad scalar for key '%s'", m->key);
        }

        if (m->type == FOX_GGUF_UINT32 &&
            strcmp(m->key, "general.alignment") == 0)
            *align = (uint32_t)m->u64;
    }
    return FOX_OK;
}

static fox_status read_tensors(reader *r, fox_gguf *g)
{
    size_t i, j;

    for (i = 0; i < g->n_tensors; i++) {
        fox_gguf_tensor *t = &g->tensors[i];
        fox_status rc = FOX_ERR_FORMAT;
        uint32_t dims = 0, type = 0, block = 0, block_bytes = 0;
        uint64_t elements = 1;
        uint64_t offset = 0, bytes = 0;

        if (!str_read(r, (char **)&t->name, &rc))
            return fox_fail(rc, "gguf: bad tensor name at index %llu",
                            (unsigned long long)i);

        for (j = 0; j < i; j++) {
            if (strcmp(t->name, g->tensors[j].name) == 0)
                return fox_fail(FOX_ERR_FORMAT,
                                "gguf: duplicate tensor name '%s'", t->name);
        }

        if (!u32(r, &dims) || dims == 0 || dims > FOX_GGUF_MAX_DIMS)
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: tensor '%s' declares %lu dimensions",
                            t->name, (unsigned long)dims);
        t->n_dims = dims;

        for (j = 0; j < dims; j++) {
            uint64_t d;
            if (!u64(r, &d) || d == 0 || !mul_ok(elements, d, &elements))
                return fox_fail(FOX_ERR_FORMAT,
                                "gguf: tensor '%s' has a bad dimension", t->name);
            t->ne[j] = d;
        }

        if (!u32(r, &type) || !u64(r, &offset))
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: truncated descriptor for tensor '%s'", t->name);

        if (!tensor_layout((fox_ggml_type)type, &block, &block_bytes))
            return fox_fail(FOX_ERR_UNSUPPORTED,
                            "gguf: tensor '%s' uses quantisation type %lu, "
                            "which this build cannot lay out",
                            t->name, (unsigned long)type);

        if (elements % block != 0)
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: tensor '%s' holds %llu elements, not a "
                            "multiple of its %lu-element block",
                            t->name, (unsigned long long)elements,
                            (unsigned long)block);

        if (!mul_ok(elements / block, block_bytes, &bytes))
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: tensor '%s' overflows a 64-bit size", t->name);

        t->type       = (fox_ggml_type)type;
        t->offset     = offset;
        t->size_bytes = bytes;
    }
    return FOX_OK;
}

static fox_status check_extents(fox_gguf *g, uint64_t file_size)
{
    size_t i;

    for (i = 0; i < g->n_tensors; i++) {
        uint64_t end;
        if (!add_ok(g->data_offset, g->tensors[i].offset, &end) ||
            !add_ok(end, g->tensors[i].size_bytes, &end) || end > file_size)
            return fox_fail(FOX_ERR_FORMAT,
                            "gguf: tensor '%s' extends past the end of the file",
                            g->tensors[i].name);
    }
    return FOX_OK;
}

fox_status fox_gguf_open(const char *path, fox_gguf **out)
{
    reader r;
    fox_gguf *g = NULL;
    char magic[4];
    uint64_t n_tensors, n_meta, aligned;
    uint32_t version, align = 32;
    int64_t file_size;
    fox_status rc;

    if (!path || !out) return fox_fail(FOX_ERR_ARG, "gguf: null path or output");
    *out = NULL;

    r.f = fox_file_open_read(path, FOX_OPEN_SEQ, NULL);
    if (!r.f) return FOX_ERR_IO;

    file_size = fox_file_size(r.f);
    if (file_size < 0) {
        fox_file_close(r.f);
        return fox_fail(FOX_ERR_IO, "gguf: cannot determine the size of %s", path);
    }
    r.size = (uint64_t)file_size;
    r.pos = 0;

    g = (fox_gguf *)calloc(1, sizeof(*g));
    if (!g) {
        rc = fox_fail(FOX_ERR_NOMEM, "gguf: header allocation failed");
        goto fail;
    }
    g->path = (char *)malloc(strlen(path) + 1);
    if (!g->path) {
        rc = fox_fail(FOX_ERR_NOMEM, "gguf: path allocation failed");
        goto fail;
    }
    strcpy(g->path, path);

    if (!read_bytes(&r, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        rc = fox_fail(FOX_ERR_FORMAT, "gguf: %s is not a GGUF file", path);
        goto fail;
    }
    if (!u32(&r, &version) || !u64(&r, &n_tensors) || !u64(&r, &n_meta)) {
        rc = fox_fail(FOX_ERR_FORMAT, "gguf: truncated header in %s", path);
        goto fail;
    }
    if (version < 2 || version > 3) {
        rc = fox_fail(FOX_ERR_UNSUPPORTED,
                      "gguf: file format version %lu is not supported",
                      (unsigned long)version);
        goto fail;
    }
    if (n_tensors > GGUF_MAX_ITEMS || n_meta > GGUF_MAX_META) {
        rc = fox_fail(FOX_ERR_FORMAT,
                      "gguf: implausible header counts (%llu tensors, %llu keys)",
                      (unsigned long long)n_tensors, (unsigned long long)n_meta);
        goto fail;
    }

    g->version   = version;
    g->file_size = r.size;
    g->n_meta    = (size_t)n_meta;
    g->n_tensors = (size_t)n_tensors;

    if (g->n_meta) {
        g->meta = (fox_gguf_meta *)calloc(g->n_meta, sizeof(*g->meta));
        if (!g->meta) {
            rc = fox_fail(FOX_ERR_NOMEM, "gguf: metadata allocation failed");
            goto fail;
        }
    }
    if (g->n_tensors) {
        g->tensors = (fox_gguf_tensor *)calloc(g->n_tensors, sizeof(*g->tensors));
        if (!g->tensors) {
            rc = fox_fail(FOX_ERR_NOMEM, "gguf: tensor table allocation failed");
            goto fail;
        }
    }

    rc = read_metadata(&r, g, &align);
    if (rc != FOX_OK) goto fail;

    if (align == 0 || (align & (align - 1)) != 0 || align > GGUF_MAX_ALIGN) {
        rc = fox_fail(FOX_ERR_FORMAT, "gguf: bad general.alignment %lu",
                      (unsigned long)align);
        goto fail;
    }
    g->alignment = align;

    rc = read_tensors(&r, g);
    if (rc != FOX_OK) goto fail;

    if (!add_ok(r.pos, align - 1, &aligned)) {
        rc = fox_fail(FOX_ERR_FORMAT, "gguf: header size overflows on alignment");
        goto fail;
    }
    g->data_offset = aligned & ~((uint64_t)align - 1);

    if (g->data_offset < r.pos || g->data_offset > r.size) {
        rc = fox_fail(FOX_ERR_FORMAT, "gguf: data section starts past the end");
        goto fail;
    }

    rc = check_extents(g, r.size);
    if (rc != FOX_OK) goto fail;

    fox_file_close(r.f);
    *out = g;
    return FOX_OK;

fail:
    dispose(g);
    fox_file_close(r.f);
    return rc;
}

void fox_gguf_close(fox_gguf *g)
{
    dispose(g);
}

uint32_t fox_gguf_version(const fox_gguf *g)
{
    return g ? g->version : 0;
}

size_t fox_gguf_metadata_count(const fox_gguf *g)
{
    return g ? g->n_meta : 0;
}

size_t fox_gguf_tensor_count(const fox_gguf *g)
{
    return g ? g->n_tensors : 0;
}

size_t fox_gguf_find(const fox_gguf *g, const char *key)
{
    size_t i;

    if (!g || !key) return SIZE_MAX;
    for (i = 0; i < g->n_meta; i++) {
        if (strcmp(g->meta[i].key, key) == 0) return i;
    }
    return SIZE_MAX;
}

fox_gguf_value_type fox_gguf_metadata_type(const fox_gguf *g, size_t index)
{
    if (!g || index >= g->n_meta) return FOX_GGUF_ARRAY;
    return g->meta[index].type;
}

size_t fox_gguf_metadata_array_count(const fox_gguf *g, size_t index)
{
    if (!g || index >= g->n_meta || g->meta[index].type != FOX_GGUF_ARRAY) return 0;
    return g->meta[index].array_count;
}

fox_gguf_value_type fox_gguf_metadata_array_type(const fox_gguf *g, size_t index)
{
    if (!g || index >= g->n_meta || g->meta[index].type != FOX_GGUF_ARRAY)
        return FOX_GGUF_ARRAY;
    return g->meta[index].array_type;
}

fox_status fox_gguf_get_u32(const fox_gguf *g, size_t index, uint32_t *out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_UINT32) return FOX_ERR_FORMAT;
    *out = (uint32_t)g->meta[index].u64;
    return FOX_OK;
}

fox_status fox_gguf_get_bool(const fox_gguf *g, size_t index, int *out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_BOOL) return FOX_ERR_FORMAT;
    *out = g->meta[index].u64 != 0;
    return FOX_OK;
}

fox_status fox_gguf_get_f32(const fox_gguf *g, size_t index, float *out)
{
    union { uint32_t u; float f; } v;

    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_FLOAT32) return FOX_ERR_FORMAT;
    v.u = (uint32_t)g->meta[index].u64;
    *out = v.f;
    return FOX_OK;
}

fox_status fox_gguf_get_uint(const fox_gguf *g, size_t index, uint64_t *out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;

    switch (g->meta[index].type) {
    case FOX_GGUF_UINT8:
    case FOX_GGUF_UINT16:
    case FOX_GGUF_UINT32:
    case FOX_GGUF_UINT64:
    case FOX_GGUF_BOOL:
        *out = g->meta[index].u64;
        return FOX_OK;
    case FOX_GGUF_INT8:
        *out = (uint64_t)(int64_t)(int8_t)g->meta[index].u64;
        return FOX_OK;
    case FOX_GGUF_INT16:
        *out = (uint64_t)(int64_t)(int16_t)g->meta[index].u64;
        return FOX_OK;
    case FOX_GGUF_INT32:
        *out = (uint64_t)(int64_t)(int32_t)g->meta[index].u64;
        return FOX_OK;
    case FOX_GGUF_INT64:
        *out = g->meta[index].u64;
        return FOX_OK;
    default:
        return FOX_ERR_FORMAT;
    }
}

fox_status fox_gguf_get_string(const fox_gguf *g, size_t index, const char **out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_STRING) return FOX_ERR_FORMAT;
    *out = g->meta[index].string;
    return FOX_OK;
}

fox_status fox_gguf_get_array_string(const fox_gguf *g, size_t index,
                                     size_t element, const char **out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_ARRAY ||
        g->meta[index].array_type != FOX_GGUF_STRING) return FOX_ERR_FORMAT;
    if (element >= g->meta[index].array_count) return FOX_ERR_NOTFOUND;
    *out = g->meta[index].array_strings[element];
    return FOX_OK;
}

fox_status fox_gguf_get_array_i32(const fox_gguf *g, size_t index,
                                  size_t element, int32_t *out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_ARRAY ||
        g->meta[index].array_type != FOX_GGUF_INT32) return FOX_ERR_FORMAT;
    if (element >= g->meta[index].array_count) return FOX_ERR_NOTFOUND;
    *out = (int32_t)g->meta[index].array_values[element];
    return FOX_OK;
}

fox_status fox_gguf_get_array_f32(const fox_gguf *g, size_t index,
                                  size_t element, float *out)
{
    union { uint32_t u; float f; } value;
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_meta) return FOX_ERR_NOTFOUND;
    if (g->meta[index].type != FOX_GGUF_ARRAY ||
        g->meta[index].array_type != FOX_GGUF_FLOAT32) return FOX_ERR_FORMAT;
    if (element >= g->meta[index].array_count) return FOX_ERR_NOTFOUND;
    value.u = (uint32_t)g->meta[index].array_values[element];
    *out = value.f;
    return FOX_OK;
}

fox_status fox_gguf_tensor_at(const fox_gguf *g, size_t index, fox_gguf_tensor *out)
{
    if (!g || !out) return FOX_ERR_ARG;
    if (index >= g->n_tensors) return FOX_ERR_NOTFOUND;
    *out = g->tensors[index];
    return FOX_OK;
}

size_t fox_gguf_tensor_find(const fox_gguf *g, const char *name)
{
    size_t i;

    if (!g || !name) return SIZE_MAX;
    for (i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return i;
    }
    return SIZE_MAX;
}

fox_status fox_gguf_read_tensor(const fox_gguf *g, size_t index,
                                void *dst, size_t capacity)
{
    fox_file *f;
    uint64_t file_offset;
    uint64_t remaining;
    uint8_t *out;

    if (!g || !dst) return FOX_ERR_ARG;
    if (index >= g->n_tensors) return FOX_ERR_NOTFOUND;
    if (g->tensors[index].size_bytes > capacity) return FOX_ERR_ARG;
    if (!add_ok(g->data_offset, g->tensors[index].offset, &file_offset))
        return FOX_ERR_FORMAT;

    f = fox_file_open_read(g->path, FOX_OPEN_RANDOM, NULL);
    if (!f) return FOX_ERR_IO;
    remaining = g->tensors[index].size_bytes;
    out = (uint8_t *)dst;
    while (remaining) {
        size_t chunk = remaining > (1u << 20) ? (1u << 20) : (size_t)remaining;
        if (fox_file_pread(f, out, chunk, file_offset) != (int64_t)chunk) {
            fox_file_close(f);
            return FOX_ERR_IO;
        }
        out += chunk;
        file_offset += chunk;
        remaining -= chunk;
    }
    fox_file_close(f);
    return FOX_OK;
}

const char *fox_ggml_type_name(fox_ggml_type type)
{
    switch (type) {
    case FOX_GGML_F32:   return "F32";
    case FOX_GGML_F16:   return "F16";
    case FOX_GGML_BF16:  return "BF16";
    case FOX_GGML_F64:   return "F64";
    case FOX_GGML_I8:    return "I8";
    case FOX_GGML_I16:   return "I16";
    case FOX_GGML_I32:   return "I32";
    case FOX_GGML_I64:   return "I64";
    case FOX_GGML_Q4_0:  return "Q4_0";
    case FOX_GGML_Q4_1:  return "Q4_1";
    case FOX_GGML_Q5_0:  return "Q5_0";
    case FOX_GGML_Q5_1:  return "Q5_1";
    case FOX_GGML_Q8_0:  return "Q8_0";
    case FOX_GGML_Q8_K:  return "Q8_K";
    case FOX_GGML_Q2_K:  return "Q2_K";
    case FOX_GGML_Q3_K:  return "Q3_K";
    case FOX_GGML_Q4_K:  return "Q4_K";
    case FOX_GGML_Q5_K:  return "Q5_K";
    case FOX_GGML_Q6_K:  return "Q6_K";
    case FOX_GGML_IQ2_XXS: return "IQ2_XXS";
    case FOX_GGML_IQ2_XS:  return "IQ2_XS";
    case FOX_GGML_IQ3_XXS: return "IQ3_XXS";
    case FOX_GGML_IQ1_S:   return "IQ1_S";
    case FOX_GGML_IQ4_NL:  return "IQ4_NL";
    case FOX_GGML_IQ3_S:   return "IQ3_S";
    case FOX_GGML_IQ2_S:   return "IQ2_S";
    case FOX_GGML_IQ4_XS:  return "IQ4_XS";
    case FOX_GGML_IQ1_M:   return "IQ1_M";
    case FOX_GGML_TQ1_0: return "TQ1_0";
    case FOX_GGML_TQ2_0: return "TQ2_0";
    default:             return "unknown";
    }
}

uint32_t fox_ggml_type_block_size(fox_ggml_type type)
{
    uint32_t block, bytes;
    return tensor_layout(type, &block, &bytes) ? block : 0;
}

uint32_t fox_ggml_type_block_bytes(fox_ggml_type type)
{
    uint32_t block, bytes;
    return tensor_layout(type, &block, &bytes) ? bytes : 0;
}
