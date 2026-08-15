#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define FOX_SLOT_COUNT (FOX_WEIGHTS_MAX_LIVE_LEASES + 4)

typedef struct {
    uint8_t *buf;
    uint64_t bytes;
    uint64_t file_offset;
    size_t   tensor_index;
    uint64_t last_used;
    unsigned pins;
    int      reserved;
} fox_slot;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
} fox_pack_span;

struct fox_weights {
    fox_gguf         *model;
    fox_weights_mode  mode;
    uint64_t          budget_bytes;
    fox_file         *file;
    uint64_t          data_offset;
    size_t            n_tensors;
    uint64_t          largest_tensor;
    fox_pack_span    *pack_spans;
    int               is_pack;

    uint8_t         **resident;

    fox_slot          slots[FOX_SLOT_COUNT];
    uint64_t          clock;
    uint64_t          reads;
    uint64_t          hits;
    uint64_t          evictions;

    uint64_t          held_bytes;
    size_t            live_leases;
    fox_mutex        *lock;
    fox_thread        *prefetch_thread;
    fox_cond          *prefetch_cond;
    int                prefetch_stop;
    int                prefetch_pending;
    int                prefetch_busy;
    int                prefetch_complete;
    size_t             prefetch_tensor;
    fox_status         prefetch_status;
};

static const size_t SLOT_EMPTY = (size_t)-1;

static fox_status read_range_into(fox_weights *w, uint64_t offset,
                                  uint64_t bytes, uint8_t *dst,
                                  const char *name)
{
    int64_t got;

    if (offset > UINT64_MAX - w->data_offset || bytes > SIZE_MAX)
        return fox_fail(FOX_ERR_FORMAT, "weights: range for '%s' overflows",
                        name ? name : "span");
    got = fox_file_pread(w->file, dst, (size_t)bytes,
                         w->data_offset + offset);
    if (got != (int64_t)bytes)
        return fox_fail(FOX_ERR_IO, "weights: short read on '%s'",
                        name ? name : "span");
    return FOX_OK;
}

static fox_status read_tensor_into(fox_weights *w, const fox_gguf_tensor *t,
                                   uint8_t *dst)
{
    fox_status st = read_range_into(w, t->offset, t->size_bytes, dst, t->name);
    if (st == FOX_OK) w->reads++;
    return st;
}

static void slot_free(fox_weights *w, fox_slot *s)
{
    if (!s->buf) return;
    fox_aligned_free(s->buf);
    w->held_bytes -= s->bytes;
    s->buf = NULL;
    s->bytes = 0;
    s->file_offset = 0;
    s->tensor_index = SLOT_EMPTY;
    s->pins = 0;
    s->reserved = 0;
}

static void slot_cancel_reservation(fox_weights *w, fox_slot *s)
{
    if (!s->reserved) return;
    w->held_bytes -= s->bytes;
    s->bytes = 0;
    s->file_offset = 0;
    s->tensor_index = SLOT_EMPTY;
    s->pins = 0;
    s->reserved = 0;
}

static int layer_number(const char *name, uint32_t *out)
{
    const char *p;
    uint64_t value = 0;

    if (!name || strncmp(name, "blk.", 4) != 0) return 0;
    p = name + 4;
    if (*p < '0' || *p > '9') return 0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10u + (uint64_t)(*p - '0');
        if (value > UINT32_MAX) return 0;
        p++;
    }
    if (*p != '.') return 0;
    *out = (uint32_t)value;
    return 1;
}

static int pack_suffix(const char *path)
{
    size_t n;
    if (!path) return 0;
    n = strlen(path);
    return n >= 8 && strcmp(path + n - 8, ".foxpack") == 0;
}

static int ranges_overlap(uint64_t a, uint64_t an,
                          uint64_t b, uint64_t bn)
{
    uint64_t ae, be;
    if (an > UINT64_MAX - a || bn > UINT64_MAX - b) return 1;
    ae = a + an;
    be = b + bn;
    return a < be && b < ae;
}

static fox_status build_pack_spans(fox_weights *w)
{
    unsigned char *done;
    size_t i, j, k;

    if (!w->is_pack) return FOX_OK;
    w->pack_spans = (fox_pack_span *)calloc(w->n_tensors ? w->n_tensors : 1,
                                            sizeof(*w->pack_spans));
    done = (unsigned char *)calloc(w->n_tensors ? w->n_tensors : 1, 1);
    if (!w->pack_spans || !done) {
        free(done);
        return fox_fail(FOX_ERR_NOMEM, "weights: .foxpack span table");
    }

    for (i = 0; i < w->n_tensors; i++) {
        fox_gguf_tensor ti;
        uint32_t layer_i;
        uint64_t start, end;
        int has_layer;
        int valid = 1;

        if (done[i]) continue;
        if (fox_gguf_tensor_at(w->model, i, &ti) != FOX_OK) {
            free(done);
            return FOX_ERR_FORMAT;
        }
        has_layer = layer_number(ti.name, &layer_i);
        start = ti.offset;
        end = ti.offset + ti.size_bytes;
        if (end < ti.offset) valid = 0;

        if (has_layer && valid) {
            for (j = 0; j < w->n_tensors; j++) {
                fox_gguf_tensor tj;
                uint32_t layer_j;
                uint64_t tj_end;

                if (fox_gguf_tensor_at(w->model, j, &tj) != FOX_OK) {
                    free(done);
                    return FOX_ERR_FORMAT;
                }
                if (!layer_number(tj.name, &layer_j) || layer_j != layer_i)
                    continue;
                tj_end = tj.offset + tj.size_bytes;
                if (tj_end < tj.offset) { valid = 0; break; }
                if (tj.offset < start) start = tj.offset;
                if (tj_end > end) end = tj_end;
            }

            for (k = 0; k < w->n_tensors && valid; k++) {
                fox_gguf_tensor tk;
                uint32_t layer_k;

                if (fox_gguf_tensor_at(w->model, k, &tk) != FOX_OK) {
                    free(done);
                    return FOX_ERR_FORMAT;
                }
                if (layer_number(tk.name, &layer_k) && layer_k == layer_i)
                    continue;
                if (ranges_overlap(start, end - start, tk.offset,
                                   tk.size_bytes))
                    valid = 0;
            }
        }

        if (!valid || !has_layer) {
            start = ti.offset;
            end = ti.offset + ti.size_bytes;
        }
        if (end < start) {
            free(done);
            return fox_fail(FOX_ERR_FORMAT, "weights: invalid .foxpack span");
        }

        for (j = 0; j < w->n_tensors; j++) {
            fox_gguf_tensor tj;
            uint32_t layer_j;

            if (fox_gguf_tensor_at(w->model, j, &tj) != FOX_OK) {
                free(done);
                return FOX_ERR_FORMAT;
            }
            if (has_layer && valid && layer_number(tj.name, &layer_j) &&
                layer_j == layer_i) {
                w->pack_spans[j].offset = start;
                w->pack_spans[j].bytes = end - start;
                done[j] = 1;
            }
        }
        if (!done[i]) {
            w->pack_spans[i].offset = start;
            w->pack_spans[i].bytes = end - start;
            done[i] = 1;
        }
    }

    for (i = 0; i < w->n_tensors; i++)
        if (w->pack_spans[i].bytes > w->largest_tensor)
            w->largest_tensor = w->pack_spans[i].bytes;

    free(done);
    return FOX_OK;
}

static int slot_covers(const fox_slot *s, const fox_gguf_tensor *t)
{
    return s->file_offset <= t->offset &&
           t->offset - s->file_offset <= s->bytes &&
           t->size_bytes <= s->bytes - (t->offset - s->file_offset);
}

static fox_slot *slot_find(fox_weights *w, size_t tensor_index)
{
    int i;
    fox_gguf_tensor t;

    if (fox_gguf_tensor_at(w->model, tensor_index, &t) != FOX_OK)
        return NULL;
    for (i = 0; i < FOX_SLOT_COUNT; i++)
        if (w->slots[i].buf && slot_covers(&w->slots[i], &t))
            return &w->slots[i];
    return NULL;
}

static fox_slot *slot_lru_unpinned(fox_weights *w)
{
    fox_slot *best = NULL;
    int i;

    for (i = 0; i < FOX_SLOT_COUNT; i++) {
        fox_slot *s = &w->slots[i];
        if (!s->buf || s->reserved || s->pins > 0) continue;
        if (!best || s->last_used < best->last_used) best = s;
    }
    return best;
}

static fox_slot *slot_empty(fox_weights *w)
{
    int i;
    for (i = 0; i < FOX_SLOT_COUNT; i++)
        if (!w->slots[i].buf && !w->slots[i].reserved) return &w->slots[i];
    return NULL;
}

static fox_status make_room(fox_weights *w, uint64_t need)
{
    if (w->budget_bytes == 0) return FOX_OK;

    if (need > w->budget_bytes)
        return fox_fail(FOX_ERR_NOMEM,
                        "weights: one tensor needs %llu bytes and the whole "
                        "budget is %llu; no amount of eviction makes this fit",
                        (unsigned long long)need,
                        (unsigned long long)w->budget_bytes);

    while (w->held_bytes + need > w->budget_bytes) {
        fox_slot *victim = slot_lru_unpinned(w);
        if (!victim)
            return fox_fail(FOX_ERR_NOMEM,
                            "weights: %llu bytes needed, %llu held, and every "
                            "slot is pinned by a live lease",
                            (unsigned long long)need,
                            (unsigned long long)w->held_bytes);
        slot_free(w, victim);
        w->evictions++;
    }
    return FOX_OK;
}

static fox_status stream_into_slot(fox_weights *w, size_t tensor_index,
                                   fox_slot **out)
{
    fox_gguf_tensor t;
    fox_slot *s;
    fox_status st;
    uint64_t span_offset;
    uint64_t span_bytes;

    st = fox_gguf_tensor_at(w->model, tensor_index, &t);
    if (st != FOX_OK) return st;

    span_offset = t.offset;
    span_bytes = t.size_bytes;
    if (w->pack_spans) {
        span_offset = w->pack_spans[tensor_index].offset;
        span_bytes = w->pack_spans[tensor_index].bytes;
    }
    if (span_bytes > SIZE_MAX)
        return fox_fail(FOX_ERR_FORMAT, "weights: span for '%s' is too large",
                        t.name);

    st = make_room(w, span_bytes);
    if (st != FOX_OK) return st;

    s = slot_empty(w);
    if (!s) {
        s = slot_lru_unpinned(w);
        if (!s)
            return fox_fail(FOX_ERR_NOMEM,
                            "weights: all %d slots are pinned", FOX_SLOT_COUNT);
        slot_free(w, s);
        w->evictions++;
    }

    s->buf = (uint8_t *)fox_aligned_alloc(4096, (size_t)span_bytes);
    if (!s->buf)
        return fox_fail(FOX_ERR_NOMEM, "weights: cannot allocate %llu bytes",
                        (unsigned long long)span_bytes);

    st = read_range_into(w, span_offset, span_bytes, s->buf, t.name);
    if (st != FOX_OK) {
        fox_aligned_free(s->buf);
        s->buf = NULL;
        return st;
    }

    s->bytes        = span_bytes;
    s->file_offset  = span_offset;
    s->tensor_index = tensor_index;
    s->pins         = 0;
    s->last_used    = ++w->clock;
    w->reads++;
    w->held_bytes  += span_bytes;

    *out = s;
    return FOX_OK;
}

static fox_status stream_prefetch_into_slot(fox_weights *w,
                                            size_t tensor_index)
{
    fox_gguf_tensor t;
    fox_slot *s;
    uint8_t *buf;
    uint64_t span_offset;
    uint64_t span_bytes;
    fox_status st;

    st = fox_gguf_tensor_at(w->model, tensor_index, &t);
    if (st != FOX_OK) return st;
    span_offset = t.offset;
    span_bytes = t.size_bytes;
    if (w->pack_spans) {
        span_offset = w->pack_spans[tensor_index].offset;
        span_bytes = w->pack_spans[tensor_index].bytes;
    }
    if (span_bytes > SIZE_MAX)
        return fox_fail(FOX_ERR_FORMAT, "weights: prefetch span for '%s' is too large",
                        t.name);

    fox_mutex_lock(w->lock);
    if (slot_find(w, tensor_index)) {
        fox_mutex_unlock(w->lock);
        return FOX_OK;
    }
    st = make_room(w, span_bytes);
    if (st != FOX_OK) {
        fox_mutex_unlock(w->lock);
        return st;
    }
    s = slot_empty(w);
    if (!s) {
        s = slot_lru_unpinned(w);
        if (!s) {
            fox_mutex_unlock(w->lock);
            return fox_fail(FOX_ERR_NOMEM,
                            "weights: prefetch cannot evict a pinned slot");
        }
        slot_free(w, s);
        w->evictions++;
    }

    s->buf = NULL;
    s->bytes = span_bytes;
    s->file_offset = span_offset;
    s->tensor_index = tensor_index;
    s->pins = 0;
    s->last_used = ++w->clock;
    s->reserved = 1;
    w->held_bytes += span_bytes;
    fox_mutex_unlock(w->lock);

    buf = (uint8_t *)fox_aligned_alloc(4096, (size_t)span_bytes);
    if (!buf) {
        st = fox_fail(FOX_ERR_NOMEM, "weights: prefetch allocation for '%s'",
                      t.name);
        goto cancel;
    }
    st = read_range_into(w, span_offset, span_bytes, buf, t.name);
    if (st != FOX_OK) goto cancel;

    fox_mutex_lock(w->lock);
    if (slot_find(w, tensor_index)) {
        slot_cancel_reservation(w, s);
        w->reads++;
        if (w->prefetch_cond) fox_cond_broadcast(w->prefetch_cond);
        fox_mutex_unlock(w->lock);
        fox_aligned_free(buf);
        return FOX_OK;
    }
    s->buf = buf;
    s->reserved = 0;
    w->reads++;
    if (w->prefetch_cond) fox_cond_broadcast(w->prefetch_cond);
    fox_mutex_unlock(w->lock);
    return FOX_OK;

cancel:
    if (buf) fox_aligned_free(buf);
    fox_mutex_lock(w->lock);
    slot_cancel_reservation(w, s);
    if (w->prefetch_cond) fox_cond_broadcast(w->prefetch_cond);
    fox_mutex_unlock(w->lock);
    return st;
}

static fox_status load_resident(fox_weights *w, size_t index)
{
    fox_gguf_tensor t;
    fox_status st;
    uint8_t *buf;

    st = fox_gguf_tensor_at(w->model, index, &t);
    if (st != FOX_OK) return st;

    if (t.size_bytes == 0)
        return fox_fail(FOX_ERR_FORMAT, "weights: tensor '%s' is empty", t.name);

    if (w->budget_bytes > 0 && w->held_bytes + t.size_bytes > w->budget_bytes)
        return fox_fail(FOX_ERR_NOMEM,
                        "weights: tensor '%s' needs %llu bytes but only %llu of "
                        "the %llu byte budget is left; this model does not fit "
                        "resident, open it with FOX_WEIGHTS_STREAM",
                        t.name, (unsigned long long)t.size_bytes,
                        (unsigned long long)(w->budget_bytes - w->held_bytes),
                        (unsigned long long)w->budget_bytes);

    buf = (uint8_t *)fox_aligned_alloc(4096, (size_t)t.size_bytes);
    if (!buf)
        return fox_fail(FOX_ERR_NOMEM, "weights: cannot allocate %llu bytes for '%s'",
                        (unsigned long long)t.size_bytes, t.name);

    st = read_tensor_into(w, &t, buf);
    if (st != FOX_OK) {
        fox_aligned_free(buf);
        return st;
    }

    w->resident[index] = buf;
    w->held_bytes += t.size_bytes;
    return FOX_OK;
}

static void prefetch_worker(void *arg)
{
    fox_weights *w = (fox_weights *)arg;

    for (;;) {
        size_t tensor_index;
        fox_status st;

        fox_mutex_lock(w->lock);
        while (!w->prefetch_stop && !w->prefetch_pending)
            fox_cond_wait(w->prefetch_cond, w->lock);
        if (w->prefetch_stop) {
            fox_mutex_unlock(w->lock);
            return;
        }
        tensor_index = w->prefetch_tensor;
        w->prefetch_pending = 0;
        w->prefetch_busy = 1;
        fox_mutex_unlock(w->lock);

        st = fox_weights_prefetch(w, tensor_index);

        fox_mutex_lock(w->lock);
        w->prefetch_status = st;
        w->prefetch_busy = 0;
        w->prefetch_complete = 1;
        fox_cond_broadcast(w->prefetch_cond);
        fox_mutex_unlock(w->lock);
    }
}

fox_status fox_weights_open(const char *gguf_path, fox_weights_mode mode,
                            uint64_t budget_bytes, fox_weights **out)
{
    fox_weights *w;
    fox_status st;
    size_t i;

    if (!gguf_path || !out) return fox_fail(FOX_ERR_ARG, "weights: null argument");
    *out = NULL;

    if (mode != FOX_WEIGHTS_RESIDENT && mode != FOX_WEIGHTS_STREAM)
        return fox_fail(FOX_ERR_ARG, "weights: unknown mode %d", (int)mode);

    w = (fox_weights *)calloc(1, sizeof(*w));
    if (!w) return fox_fail(FOX_ERR_NOMEM, "weights: handle alloc");

    w->mode         = mode;
    w->budget_bytes = budget_bytes;
    w->is_pack      = mode == FOX_WEIGHTS_STREAM && pack_suffix(gguf_path);

    for (i = 0; i < FOX_SLOT_COUNT; i++) w->slots[i].tensor_index = SLOT_EMPTY;

    st = fox_gguf_open(gguf_path, &w->model);
    if (st != FOX_OK) {
        free(w);
        return st;
    }

    w->data_offset = w->model->data_offset;
    w->n_tensors   = fox_gguf_tensor_count(w->model);

    for (i = 0; i < w->n_tensors; i++) {
        fox_gguf_tensor t;
        if (fox_gguf_tensor_at(w->model, i, &t) == FOX_OK &&
            t.size_bytes > w->largest_tensor)
            w->largest_tensor = t.size_bytes;
    }

    st = build_pack_spans(w);
    if (st != FOX_OK) {
        fox_gguf_close(w->model);
        free(w->pack_spans);
        free(w);
        return st;
    }

    if (mode == FOX_WEIGHTS_STREAM && budget_bytes > 0 &&
        w->largest_tensor > budget_bytes) {
        uint64_t largest = w->largest_tensor;
        fox_gguf_close(w->model);
        free(w->pack_spans);
        free(w);
        return fox_fail(FOX_ERR_NOMEM,
                        "weights: the largest tensor is %llu bytes and the "
                        "budget is %llu; streaming cannot help when a single "
                        "tensor does not fit",
                        (unsigned long long)largest,
                        (unsigned long long)budget_bytes);
    }

    if (mode == FOX_WEIGHTS_RESIDENT) {
        w->resident = (uint8_t **)calloc(w->n_tensors ? w->n_tensors : 1,
                                         sizeof(*w->resident));
        if (!w->resident) {
            fox_gguf_close(w->model);
            free(w->pack_spans);
            free(w);
            return fox_fail(FOX_ERR_NOMEM, "weights: residency table");
        }
    }

    w->lock = fox_mutex_create();
    if (!w->lock) {
        free(w->resident);
        fox_gguf_close(w->model);
        free(w->pack_spans);
        free(w);
        return fox_fail(FOX_ERR_NOMEM, "weights: lock");
    }

    w->file = fox_file_open_read(gguf_path,
                                 mode == FOX_WEIGHTS_STREAM ? FOX_OPEN_RANDOM
                                                            : FOX_OPEN_SEQ,
                                 NULL);
    if (!w->file) {
        fox_mutex_destroy(w->lock);
        free(w->resident);
        free(w->pack_spans);
        fox_gguf_close(w->model);
        free(w);
        return FOX_ERR_IO;
    }

    if (w->is_pack) {
        w->prefetch_cond = fox_cond_create();
        if (!w->prefetch_cond) {
            fox_file_close(w->file);
            fox_mutex_destroy(w->lock);
            free(w->resident);
            free(w->pack_spans);
            fox_gguf_close(w->model);
            free(w);
            return fox_fail(FOX_ERR_NOMEM, "weights: prefetch condition");
        }
        w->prefetch_thread = fox_thread_start(prefetch_worker, w);
        if (!w->prefetch_thread) {
            fox_cond_destroy(w->prefetch_cond);
            fox_file_close(w->file);
            fox_mutex_destroy(w->lock);
            free(w->resident);
            free(w->pack_spans);
            fox_gguf_close(w->model);
            free(w);
            return fox_fail(FOX_ERR_NOMEM, "weights: prefetch worker");
        }
    }

    *out = w;
    return FOX_OK;
}

void fox_weights_close(fox_weights *w)
{
    size_t i;

    if (!w) return;

    if (w->prefetch_thread) {
        fox_mutex_lock(w->lock);
        w->prefetch_stop = 1;
        fox_cond_broadcast(w->prefetch_cond);
        fox_mutex_unlock(w->lock);
        fox_thread_join(w->prefetch_thread);
        w->prefetch_thread = NULL;
    }

    if (w->live_leases > 0)
        FOX_WARN("weights: closing with %llu lease(s) still held",
                 (unsigned long long)w->live_leases);

    if (w->resident) {
        for (i = 0; i < w->n_tensors; i++)
            if (w->resident[i]) fox_aligned_free(w->resident[i]);
        free(w->resident);
    }
    for (i = 0; i < FOX_SLOT_COUNT; i++)
        if (w->slots[i].buf) fox_aligned_free(w->slots[i].buf);

    free(w->pack_spans);
    if (w->prefetch_cond) fox_cond_destroy(w->prefetch_cond);
    if (w->lock)  fox_mutex_destroy(w->lock);
    if (w->file)  fox_file_close(w->file);
    if (w->model) fox_gguf_close(w->model);
    free(w);
}

const fox_gguf *fox_weights_model(const fox_weights *w)
{
    return w ? w->model : NULL;
}

fox_weights_mode fox_weights_get_mode(const fox_weights *w)
{
    return w ? w->mode : FOX_WEIGHTS_RESIDENT;
}

fox_status fox_weights_prefetch(fox_weights *w, size_t tensor_index)
{
    fox_status st;
    int already_loaded;

    if (!w) return FOX_ERR_ARG;
    if (tensor_index >= w->n_tensors) return FOX_ERR_NOTFOUND;

    if (w->mode == FOX_WEIGHTS_RESIDENT) {
        fox_mutex_lock(w->lock);
        if (!w->resident[tensor_index]) st = load_resident(w, tensor_index);
        else st = FOX_OK;
        fox_mutex_unlock(w->lock);
        return st;
    }

    fox_mutex_lock(w->lock);
    already_loaded = slot_find(w, tensor_index) != NULL;
    fox_mutex_unlock(w->lock);
    if (already_loaded) return FOX_OK;
    return stream_prefetch_into_slot(w, tensor_index);
}

fox_status fox_weights_prefetch_async(fox_weights *w, size_t tensor_index)
{
    if (!w) return FOX_ERR_ARG;
    if (tensor_index >= w->n_tensors) return FOX_ERR_NOTFOUND;
    if (!w->prefetch_thread) return FOX_ERR_UNSUPPORTED;

    fox_mutex_lock(w->lock);
    if (slot_find(w, tensor_index)) {
        fox_mutex_unlock(w->lock);
        return FOX_OK;
    }
    if (w->prefetch_pending || w->prefetch_busy) {
        int same = w->prefetch_tensor == tensor_index;
        fox_mutex_unlock(w->lock);
        return same ? FOX_OK :
            fox_fail(FOX_ERR_INTERNAL,
                     "weights: only one asynchronous prefetch can be in flight");
    }
    w->prefetch_tensor = tensor_index;
    w->prefetch_pending = 1;
    w->prefetch_complete = 0;
    fox_cond_broadcast(w->prefetch_cond);
    fox_mutex_unlock(w->lock);
    return FOX_OK;
}

fox_status fox_weights_prefetch_wait(fox_weights *w, size_t tensor_index)
{
    fox_status st;

    if (!w) return FOX_ERR_ARG;
    if (tensor_index >= w->n_tensors) return FOX_ERR_NOTFOUND;
    if (!w->prefetch_thread) return FOX_ERR_UNSUPPORTED;

    fox_mutex_lock(w->lock);
    if (slot_find(w, tensor_index)) {
        fox_mutex_unlock(w->lock);
        return FOX_OK;
    }
    if (w->prefetch_pending || w->prefetch_busy) {
        if (w->prefetch_tensor != tensor_index) {
            fox_mutex_unlock(w->lock);
            return fox_fail(FOX_ERR_INTERNAL,
                            "weights: waiting for a different prefetch target");
        }
        while (w->prefetch_pending || w->prefetch_busy)
            fox_cond_wait(w->prefetch_cond, w->lock);
    }
    if (w->prefetch_complete && w->prefetch_tensor == tensor_index) {
        st = w->prefetch_status;
        w->prefetch_complete = 0;
        fox_mutex_unlock(w->lock);
        return st;
    }
    fox_mutex_unlock(w->lock);
    return FOX_ERR_UNSUPPORTED;
}

fox_status fox_weights_acquire(fox_weights *w, size_t tensor_index,
                               fox_weights_lease *lease)
{
    fox_gguf_tensor t;
    fox_status st;

    if (!w || !lease) return fox_fail(FOX_ERR_ARG, "weights: null argument");
    if (tensor_index >= w->n_tensors)
        return fox_fail(FOX_ERR_NOTFOUND, "weights: no tensor at index %llu",
                        (unsigned long long)tensor_index);

    fox_mutex_lock(w->lock);

    if (w->live_leases >= FOX_WEIGHTS_MAX_LIVE_LEASES) {
        fox_mutex_unlock(w->lock);
        return fox_fail(FOX_ERR_INTERNAL,
                        "weights: %d leases already held. A lease is only valid "
                        "until it is released; holding tensors across layers "
                        "works resident and starves the streaming slots. "
                        "Release before acquiring the next one.",
                        FOX_WEIGHTS_MAX_LIVE_LEASES);
    }

    st = fox_gguf_tensor_at(w->model, tensor_index, &t);
    if (st != FOX_OK) {
        fox_mutex_unlock(w->lock);
        return st;
    }

    if (w->mode == FOX_WEIGHTS_RESIDENT) {
        if (!w->resident[tensor_index]) {
            st = load_resident(w, tensor_index);
            if (st != FOX_OK) {
                fox_mutex_unlock(w->lock);
                return st;
            }
        }
        lease->data = w->resident[tensor_index];
        lease->slot = NULL;
    } else {
        fox_slot *s;

        for (;;) {
            int waiting = 0;
            int i;

            s = slot_find(w, tensor_index);
            if (s) {
                w->hits++;
                break;
            }

            if (w->prefetch_cond) {
                for (i = 0; i < FOX_SLOT_COUNT; i++) {
                    fox_gguf_tensor reserved_tensor;
                    fox_slot *candidate = &w->slots[i];

                    if (!candidate->reserved ||
                        fox_gguf_tensor_at(w->model,
                                           candidate->tensor_index,
                                           &reserved_tensor) != FOX_OK)
                        continue;
                    if (slot_covers(candidate, &t)) {
                        waiting = 1;
                        break;
                    }
                }
            }
            if (waiting) {
                fox_cond_wait(w->prefetch_cond, w->lock);
                continue;
            }

            st = stream_into_slot(w, tensor_index, &s);
            if (st != FOX_OK) {
                fox_mutex_unlock(w->lock);
                return st;
            }
            break;
        }
        s->pins++;
        s->last_used = ++w->clock;
        lease->data = s->buf + (t.offset - s->file_offset);
        lease->slot = s;
    }

    lease->bytes        = t.size_bytes;
    lease->tensor_index = tensor_index;
    w->live_leases++;

    fox_mutex_unlock(w->lock);
    return FOX_OK;
}

void fox_weights_release(fox_weights *w, fox_weights_lease *lease)
{
    if (!w || !lease) return;

    fox_mutex_lock(w->lock);

    if (w->live_leases == 0) {
        fox_mutex_unlock(w->lock);
        FOX_WARN("weights: release with no lease outstanding");
        return;
    }

    if (lease->slot) {
        fox_slot *s = (fox_slot *)lease->slot;
        if (s->pins > 0) s->pins--;
    }

    w->live_leases--;
    lease->data  = NULL;
    lease->bytes = 0;
    lease->slot  = NULL;

    fox_mutex_unlock(w->lock);
}

size_t fox_weights_live_leases(const fox_weights *w)
{
    return w ? w->live_leases : 0;
}

uint64_t fox_weights_resident_bytes(const fox_weights *w)
{
    return w ? w->held_bytes : 0;
}

uint64_t fox_weights_reads(const fox_weights *w)
{
    return w ? w->reads : 0;
}

uint64_t fox_weights_evictions(const fox_weights *w)
{
    return w ? w->evictions : 0;
}
