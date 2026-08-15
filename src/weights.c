#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

struct fox_weights {
    fox_gguf         *model;
    fox_weights_mode  mode;
    uint64_t          budget_bytes;
    fox_file         *file;
    uint64_t          data_offset;
    size_t            n_tensors;
    uint8_t         **resident;
    uint64_t          resident_bytes;
    size_t            live_leases;
    size_t            peak_leases;
};

static fox_status load_tensor(fox_weights *w, size_t index)
{
    fox_gguf_tensor t;
    fox_status st;
    uint8_t *buf;
    int64_t got;

    st = fox_gguf_tensor_at(w->model, index, &t);
    if (st != FOX_OK) return st;

    if (t.size_bytes == 0)
        return fox_fail(FOX_ERR_FORMAT, "weights: tensor '%s' is empty", t.name);

    if (w->budget_bytes > 0 &&
        w->resident_bytes + t.size_bytes > w->budget_bytes)
        return fox_fail(FOX_ERR_NOMEM,
                        "weights: tensor '%s' needs %llu bytes but only %llu of "
                        "the %llu byte budget is left; this model does not fit "
                        "resident and needs the streaming backend",
                        t.name, (unsigned long long)t.size_bytes,
                        (unsigned long long)(w->budget_bytes - w->resident_bytes),
                        (unsigned long long)w->budget_bytes);

    buf = (uint8_t *)fox_aligned_alloc(4096, (size_t)t.size_bytes);
    if (!buf)
        return fox_fail(FOX_ERR_NOMEM, "weights: cannot allocate %llu bytes for '%s'",
                        (unsigned long long)t.size_bytes, t.name);

    got = fox_file_pread(w->file, buf, (size_t)t.size_bytes,
                         w->data_offset + t.offset);
    if (got != (int64_t)t.size_bytes) {
        fox_aligned_free(buf);
        return fox_fail(FOX_ERR_IO, "weights: short read on tensor '%s'", t.name);
    }

    w->resident[index] = buf;
    w->resident_bytes += t.size_bytes;
    return FOX_OK;
}

fox_status fox_weights_open(const char *gguf_path, fox_weights_mode mode,
                            uint64_t budget_bytes, fox_weights **out)
{
    fox_weights *w;
    fox_status st;

    if (!gguf_path || !out) return fox_fail(FOX_ERR_ARG, "weights: null argument");
    *out = NULL;

    if (mode != FOX_WEIGHTS_RESIDENT && mode != FOX_WEIGHTS_STREAM)
        return fox_fail(FOX_ERR_ARG, "weights: unknown mode %d", (int)mode);

    if (mode == FOX_WEIGHTS_STREAM)
        return fox_fail(FOX_ERR_UNSUPPORTED,
                        "weights: the streaming backend is not implemented yet; "
                        "open with FOX_WEIGHTS_RESIDENT");

    w = (fox_weights *)calloc(1, sizeof(*w));
    if (!w) return fox_fail(FOX_ERR_NOMEM, "weights: handle alloc");

    w->mode         = mode;
    w->budget_bytes = budget_bytes;

    st = fox_gguf_open(gguf_path, &w->model);
    if (st != FOX_OK) {
        free(w);
        return st;
    }

    w->data_offset = w->model->data_offset;
    w->n_tensors   = fox_gguf_tensor_count(w->model);

    w->resident = (uint8_t **)calloc(w->n_tensors ? w->n_tensors : 1,
                                     sizeof(*w->resident));
    if (!w->resident) {
        fox_gguf_close(w->model);
        free(w);
        return fox_fail(FOX_ERR_NOMEM, "weights: residency table");
    }

    w->file = fox_file_open_read(gguf_path, FOX_OPEN_SEQ, NULL);
    if (!w->file) {
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
    if (!w) return FOX_ERR_ARG;
    if (tensor_index >= w->n_tensors) return FOX_ERR_NOTFOUND;
    if (w->resident[tensor_index]) return FOX_OK;
    return load_tensor(w, tensor_index);
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

    if (w->live_leases >= FOX_WEIGHTS_MAX_LIVE_LEASES)
        return fox_fail(FOX_ERR_INTERNAL,
                        "weights: %d leases already held. A lease is only valid "
                        "until it is released; holding tensors across layers "
                        "works resident and deadlocks the streaming ring. "
                        "Release before acquiring the next one.",
                        FOX_WEIGHTS_MAX_LIVE_LEASES);

    st = fox_gguf_tensor_at(w->model, tensor_index, &t);
    if (st != FOX_OK) return st;

    if (!w->resident[tensor_index]) {
        st = load_tensor(w, tensor_index);
        if (st != FOX_OK) return st;
    }

    lease->data         = w->resident[tensor_index];
    lease->bytes        = t.size_bytes;
    lease->tensor_index = tensor_index;
    lease->slot         = NULL;

    w->live_leases++;
    if (w->live_leases > w->peak_leases) w->peak_leases = w->live_leases;
    return FOX_OK;
}

void fox_weights_release(fox_weights *w, fox_weights_lease *lease)
{
    if (!w || !lease) return;

    if (w->live_leases == 0) {
        FOX_WARN("weights: release with no lease outstanding");
        return;
    }

    w->live_leases--;
    lease->data  = NULL;
    lease->bytes = 0;
    lease->slot  = NULL;
}

size_t fox_weights_live_leases(const fox_weights *w)
{
    return w ? w->live_leases : 0;
}

uint64_t fox_weights_resident_bytes(const fox_weights *w)
{
    return w ? w->resident_bytes : 0;
}
