#include "fox_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define FOX_MAX_THREADS 256

struct fox_threadpool {
    int              n_threads;
    int              n_workers;
    fox_thread     **workers;
    fox_mutex       *lock;
    fox_cond        *work_ready;
    fox_cond        *work_done;
    fox_parallel_fn  fn;
    void            *ctx;
    size_t           n;
    unsigned         generation;
    int              remaining;
    int              stop;
};

typedef struct {
    fox_threadpool *tp;
    int             id;
} worker_arg;

static void run_slice(fox_threadpool *tp, int id)
{
    size_t total = tp->n;
    size_t threads = (size_t)tp->n_threads;
    size_t per = total / threads;
    size_t rem = total % threads;
    size_t extra = (size_t)id < rem ? (size_t)id : rem;
    size_t begin = (size_t)id * per + extra;
    size_t count = per + ((size_t)id < rem ? 1u : 0u);

    if (count > 0) tp->fn(tp->ctx, id, begin, begin + count);
}

static void worker_main(void *p)
{
    worker_arg *wa = (worker_arg *)p;
    fox_threadpool *tp = wa->tp;
    int id = wa->id;
    unsigned seen = 0;

    for (;;) {
        fox_mutex_lock(tp->lock);
        while (!tp->stop && tp->generation == seen)
            fox_cond_wait(tp->work_ready, tp->lock);
        if (tp->stop) {
            fox_mutex_unlock(tp->lock);
            break;
        }
        seen = tp->generation;
        fox_mutex_unlock(tp->lock);

        run_slice(tp, id);

        fox_mutex_lock(tp->lock);
        tp->remaining--;
        if (tp->remaining == 0) fox_cond_broadcast(tp->work_done);
        fox_mutex_unlock(tp->lock);
    }

    free(wa);
}

fox_threadpool *fox_threadpool_create(int n_threads)
{
    fox_threadpool *tp;
    int i;

    if (n_threads <= 0) n_threads = fox_cpu_count_online();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > FOX_MAX_THREADS) n_threads = FOX_MAX_THREADS;

    tp = (fox_threadpool *)calloc(1, sizeof(*tp));
    if (!tp) { fox_fail(FOX_ERR_NOMEM, "threadpool alloc"); return NULL; }

    tp->n_threads = n_threads;
    tp->n_workers = n_threads - 1;

    if (tp->n_workers == 0) return tp;

    tp->lock       = fox_mutex_create();
    tp->work_ready = fox_cond_create();
    tp->work_done  = fox_cond_create();
    tp->workers    = (fox_thread **)calloc((size_t)tp->n_workers, sizeof(*tp->workers));

    if (!tp->lock || !tp->work_ready || !tp->work_done || !tp->workers) {
        fox_threadpool_destroy(tp);
        fox_fail(FOX_ERR_NOMEM, "threadpool primitives");
        return NULL;
    }

    for (i = 0; i < tp->n_workers; i++) {
        worker_arg *wa = (worker_arg *)calloc(1, sizeof(*wa));
        if (!wa) break;
        wa->tp = tp;
        wa->id = i + 1;
        tp->workers[i] = fox_thread_start(worker_main, wa);
        if (!tp->workers[i]) {
            free(wa);
            break;
        }
    }

    if (i < tp->n_workers) {
        FOX_WARN("threadpool: only %d of %d workers started; running with %d",
                 i, tp->n_workers, i + 1);
        tp->n_workers = i;
        tp->n_threads = i + 1;
    }

    return tp;
}

void fox_threadpool_destroy(fox_threadpool *tp)
{
    int i;

    if (!tp) return;

    if (tp->lock && tp->n_workers > 0) {
        fox_mutex_lock(tp->lock);
        tp->stop = 1;
        fox_cond_broadcast(tp->work_ready);
        fox_mutex_unlock(tp->lock);
    }

    if (tp->workers) {
        for (i = 0; i < tp->n_workers; i++)
            if (tp->workers[i]) fox_thread_join(tp->workers[i]);
        free(tp->workers);
    }

    if (tp->work_done)  fox_cond_destroy(tp->work_done);
    if (tp->work_ready) fox_cond_destroy(tp->work_ready);
    if (tp->lock)       fox_mutex_destroy(tp->lock);
    free(tp);
}

int fox_threadpool_size(const fox_threadpool *tp)
{
    return tp ? tp->n_threads : 1;
}

fox_status fox_parallel_for(fox_threadpool *tp, size_t n,
                            fox_parallel_fn fn, void *ctx)
{
    if (!fn) return FOX_ERR_ARG;
    if (n == 0) return FOX_OK;

    if (!tp || tp->n_workers == 0) {
        fn(ctx, 0, 0, n);
        return FOX_OK;
    }

    fox_mutex_lock(tp->lock);
    tp->fn        = fn;
    tp->ctx       = ctx;
    tp->n         = n;
    tp->remaining = tp->n_workers;
    tp->generation++;
    fox_cond_broadcast(tp->work_ready);
    fox_mutex_unlock(tp->lock);

    run_slice(tp, 0);

    fox_mutex_lock(tp->lock);
    while (tp->remaining > 0) fox_cond_wait(tp->work_done, tp->lock);
    fox_mutex_unlock(tp->lock);

    return FOX_OK;
}
