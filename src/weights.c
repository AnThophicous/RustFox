#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define FOX_SLOT_COUNT (FOX_WEIGHTS_MAX_LIVE_LEASES + 4)

typedef struct {
    uint8_t *buf;
    uint64_t bytes;
    size_t   tensor_index;
    uint64_t last_used;
    unsigned pins;
} fox_slot;

struct fox_weights {
    fox_gguf         *model;
    fox_weights_mode  mode;
    uint64_t          budget_bytes;
    fox_file         *file;
    uint64_t          data_offset;
    size_t            n_tensors;
    uint64_t          largest_tensor;

    uint8_t         **resident;

    fox_slot          slots[FOX_SLOT_COUNT];
    uint64_t          clock;
    uint64_t          reads;
    uint64_t          hits;
    uint64_t          evictions;

    uint64_t          held_bytes;
    size_t            live_leases;
    fox_mutex        *lock;
};

static const size_t SLOT_EMPTY = (size_t)-1;

static fox_status read_tensor_into(fox_weights *w, const fox_gguf_tensor *t,
                                   uint8_t *dst)
{
    int64_t got = fox_file_pread(w->file, dst, (size_t)t->size_bytes,
                                 w->data_offset + t->offset);
    if (got != (int64_t)t->size_bytes)
        return fox_fail(FOX_ERR_IO, "weights: short read on tensor '%s'", t->name);
    w->reads++;
    return FOX_OK;
}

static void slot_free(fox_weights *w, fox_slot *s)
{
    if (!s->buf) return;
    fox_aligned_free(s->buf);
    w->held_bytes -= s->bytes;
    s->buf = NULL;
    s->bytes = 0;
    s->tensor_index = SLOT_EMPTY;
    s->pins = 0;
}

static fox_slot *slot_find(fox_weights *w, size_t tensor_index)
{
    int i;
    for (i = 0; i < FOX_SLOT_COUNT; i++)
        if (w->slots[i].buf && w->slots[i].tensor_index == tensor_index)
            return &w->slots[i];
    return NULL;
}

static fox_slot *slot_lru_unpinned(fox_weights *w)
{
    fox_slot *best = NULL;
    int i;

    for (i = 0; i < FOX_SLOT_COUNT; i++) {
        fox_slot *s = &w->slots[i];
        if (!s->buf || s->pins > 0) continue;
        if (!best || s->last_used < best->last_used) best = s;
    }
    return best;
}

static fox_slot *slot_empty(fox_weights *w)
{
    int i;
    for (i = 0; i < FOX_SLOT_COUNT; i++)
        if (!w->slots[i].buf) return &w->slots[i];
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

    st = fox_gguf_tensor_at(w->model, tensor_index, &t);
    if (st != FOX_OK) return st;

    st = make_room(w, t.size_bytes);
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

    s->buf = (uint8_t *)fox_aligned_alloc(4096, (size_t)t.size_bytes);
    if (!s->buf)
        return fox_fail(FOX_ERR_NOMEM, "weights: cannot allocate %llu bytes",
                        (unsigned long long)t.size_bytes);

    st = read_tensor_into(w, &t, s->buf);
    if (st != FOX_OK) {
        fox_aligned_free(s->buf);
        s->buf = NULL;
        return st;
    }

    s->bytes        = t.size_bytes;
    s->tensor_index = tensor_index;
    s->pins         = 0;
    s->last_used    = ++w->clock;
    w->held_bytes  += t.size_bytes;

    *out = s;
    return FOX_OK;
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

    if (mode == FOX_WEIGHTS_STREAM && budget_bytes > 0 &&
        w->largest_tensor > budget_bytes) {
        uint64_t largest = w->largest_tensor;
        fox_gguf_close(w->model);
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
            free(w);
            return fox_fail(FOX_ERR_NOMEM, "weights: residency table");
        }
    }

    w->lock = fox_mutex_create();
    if (!w->lock) {
        free(w->resident);
        fox_gguf_close(w->model);
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
        fox_gguf_close(w->model);
        free(w);
        return FOX_ERR_IO;
    }

    *out = w;
    return FOX_OK;
}

void fox_weights_close(fox_weights *w)
{
    size_t i;

    if (!w) return;

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
    fox_status st = FOX_OK;

    if (!w) return FOX_ERR_ARG;
    if (tensor_index >= w->n_tensors) return FOX_ERR_NOTFOUND;

    fox_mutex_lock(w->lock);
    if (w->mode == FOX_WEIGHTS_RESIDENT) {
        if (!w->resident[tensor_index]) st = load_resident(w, tensor_index);
    } else if (!slot_find(w, tensor_index)) {
        fox_slot *s = NULL;
        st = stream_into_slot(w, tensor_index, &s);
    }
    fox_mutex_unlock(w->lock);
    return st;
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
        fox_slot *s = slot_find(w, tensor_index);
        if (s) {
            w->hits++;
        } else {
            st = stream_into_slot(w, tensor_index, &s);
            if (st != FOX_OK) {
                fox_mutex_unlock(w->lock);
                return st;
            }
        }
        s->pins++;
        s->last_used = ++w->clock;
        lease->data = s->buf;
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
