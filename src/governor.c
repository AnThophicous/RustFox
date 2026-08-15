#include "fox_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define FOX_GOV_COOLDOWN_TICKS 4
#define FOX_GOV_QUIET_TICKS    2

struct fox_governor {
    fox_governor_config cfg;
    uint64_t            budget;
    fox_pressure        prev;
    uint64_t            prev_ns;
    int                 have_prev;
    int                 quiet_ticks;
    int                 cooldown;
    int                 psi_ok;
};

void fox_governor_config_default(fox_governor_config *cfg, const fox_sysinfo *si)
{
    uint64_t total, reserve, ceil_b, floor_b;

    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    total = (si && si->mem.total_bytes) ? si->mem.total_bytes : (2ull << 30);

    reserve = total / 4;
    if (reserve < (192ull << 20)) reserve = 192ull << 20;
    if (reserve > (2ull << 30))   reserve = 2ull << 30;

    ceil_b = (total > reserve) ? total - reserve : total / 2;

    if (si && si->mem.available_bytes > 0) {
        uint64_t headroom = si->mem.available_bytes - si->mem.available_bytes / 10;
        if (headroom < ceil_b) ceil_b = headroom;
    }
    if (si && si->mem.cgroup_high != UINT64_MAX && si->mem.cgroup_high > 0) {
        uint64_t h = si->mem.cgroup_high - si->mem.cgroup_high / 10;
        if (h < ceil_b) ceil_b = h;
    }

    floor_b = total / 16;
    if (floor_b < (32ull << 20)) floor_b = 32ull << 20;
    if (floor_b > ceil_b)        floor_b = ceil_b;

    cfg->floor_bytes         = floor_b;
    cfg->ceil_bytes          = ceil_b;
    cfg->target_mem_stall    = 2.0;
    cfg->target_io_stall     = 15.0;
    cfg->target_majflt_per_s = 50.0;
    cfg->increase_frac       = 0.02;
    cfg->decrease_factor     = 0.80;
    cfg->tick_ms             = 250;
}

static void sanitise(fox_governor_config *c)
{
    if (c->ceil_bytes < c->floor_bytes)  c->ceil_bytes = c->floor_bytes;
    if (c->target_mem_stall <= 0.0)      c->target_mem_stall = 2.0;
    if (c->target_io_stall <= 0.0)       c->target_io_stall = 15.0;
    if (c->target_majflt_per_s <= 0.0)   c->target_majflt_per_s = 50.0;
    c->increase_frac   = fox_clamp_f64(c->increase_frac, 0.001, 0.5);
    c->decrease_factor = fox_clamp_f64(c->decrease_factor, 0.25, 0.99);
    if (c->tick_ms == 0) c->tick_ms = 250;
}

fox_governor *fox_governor_create(const fox_governor_config *cfg,
                                  uint64_t initial_budget_bytes)
{
    fox_governor *g = (fox_governor *)calloc(1, sizeof(*g));
    if (!g) { fox_fail(FOX_ERR_NOMEM, "governor alloc"); return NULL; }

    if (cfg) g->cfg = *cfg;
    else     fox_governor_config_default(&g->cfg, NULL);
    sanitise(&g->cfg);

    if (initial_budget_bytes == 0)
        initial_budget_bytes = g->cfg.floor_bytes +
                               (g->cfg.ceil_bytes - g->cfg.floor_bytes) / 2;

    g->budget = fox_clamp_u64(initial_budget_bytes,
                              g->cfg.floor_bytes, g->cfg.ceil_bytes);
    g->psi_ok = 1;
    return g;
}

void fox_governor_destroy(fox_governor *g) { free(g); }

uint64_t fox_governor_budget(const fox_governor *g)
{
    return g ? g->budget : 0;
}

static uint64_t counter_delta(uint64_t cur, uint64_t prev)
{
    return cur >= prev ? cur - prev : 0;
}

static void apply_cut(fox_governor *g, fox_gov_tick *r, const char *reason)
{
    uint64_t next = (uint64_t)((double)g->budget * g->cfg.decrease_factor);

    if (next < g->cfg.floor_bytes) next = g->cfg.floor_bytes;
    g->budget       = next;
    g->quiet_ticks  = 0;
    g->cooldown     = FOX_GOV_COOLDOWN_TICKS;
    r->action       = FOX_GOV_CUT;
    r->budget_bytes = next;
    r->reason       = reason;
}

static fox_status tick_without_psi(fox_governor *g, fox_gov_tick *r)
{
    fox_mem_info m;
    uint64_t want_free;

    fox_probe_mem(&m);

    want_free = g->cfg.ceil_bytes / 8;
    if (want_free < (128ull << 20)) want_free = 128ull << 20;

    if (m.available_bytes > 0 && m.available_bytes < want_free) {
        apply_cut(g, r, "available memory below reserve (no PSI on this host)");
        return FOX_OK;
    }

    if (g->cooldown > 0) {
        g->cooldown--;
        r->reason = "cooling down after a cut";
        return FOX_OK;
    }

    if (m.available_bytes > want_free * 3 && g->budget < g->cfg.ceil_bytes) {
        uint64_t step = (uint64_t)((double)g->cfg.ceil_bytes * g->cfg.increase_frac);
        if (step == 0) step = 1u << 20;
        g->budget       = fox_min_u64(g->budget + step, g->cfg.ceil_bytes);
        r->action       = FOX_GOV_GROW;
        r->budget_bytes = g->budget;
        r->reason       = "memory available and quiet (no PSI on this host)";
    }
    return FOX_OK;
}

fox_status fox_governor_tick(fox_governor *g, fox_gov_tick *out)
{
    fox_pressure now;
    fox_gov_tick r;
    uint64_t t, events = 0;
    double elapsed;
    int cut = 0;
    const char *reason = "no change";

    if (!g) return FOX_ERR_ARG;

    t = fox_now_ns();

    memset(&r, 0, sizeof(r));
    r.action            = FOX_GOV_HOLD;
    r.prev_budget_bytes = g->budget;
    r.budget_bytes      = g->budget;
    r.reason            = reason;

    if (fox_pressure_read(&now) != FOX_OK) {
        fox_status st;
        g->psi_ok = 0;
        st = tick_without_psi(g, &r);
        g->prev_ns = t;
        if (out) *out = r;
        return st;
    }
    g->psi_ok = 1;

    elapsed = g->have_prev ? (double)(t - g->prev_ns) / 1e9 : 0.0;
    if (elapsed < 1e-6) elapsed = (double)g->cfg.tick_ms / 1000.0;

    r.mem_stall = now.mem.some_avg10;
    r.io_stall  = now.io.some_avg10;

    if (g->have_prev) {
        r.majflt_per_s = (double)counter_delta(now.pgmajfault, g->prev.pgmajfault) / elapsed;
        r.swapin_per_s = (double)counter_delta(now.pswpin, g->prev.pswpin) / elapsed;
        events = counter_delta(now.cgroup_high_events, g->prev.cgroup_high_events) +
                 counter_delta(now.cgroup_max_events,  g->prev.cgroup_max_events);
    }

    if (events > 0) {
        cut = 1; reason = "cgroup memory.events fired";
    } else if (now.mem.full_avg10 > 0.5) {
        cut = 1; reason = "every task stalled on memory";
    } else if (r.mem_stall > g->cfg.target_mem_stall) {
        cut = 1; reason = "memory stall above target";
    } else if (r.majflt_per_s > g->cfg.target_majflt_per_s) {
        cut = 1; reason = "major fault storm: resident set is a lie";
    } else if (r.swapin_per_s > 1.0) {
        cut = 1; reason = "swapping back in";
    }

    if (cut) {
        apply_cut(g, &r, reason);
    } else if (g->cooldown > 0) {
        g->cooldown--;
        r.reason = "cooling down after a cut";
    } else if (r.mem_stall > g->cfg.target_mem_stall / 2.0) {
        g->quiet_ticks = 0;
        r.reason = "memory stall approaching target, holding";
    } else if (g->budget >= g->cfg.ceil_bytes) {
        r.reason = "at ceiling";
    } else {
        g->quiet_ticks++;
        if (g->quiet_ticks >= FOX_GOV_QUIET_TICKS) {
            double frac = g->cfg.increase_frac;
            uint64_t step;

            if (r.io_stall > g->cfg.target_io_stall) {
                frac *= 4.0;
                reason = "io stall high and memory quiet: growing fast";
            } else {
                reason = "quiet: growing";
            }

            step = (uint64_t)((double)g->cfg.ceil_bytes * frac);
            if (step == 0) step = 1u << 20;

            g->budget      = fox_min_u64(g->budget + step, g->cfg.ceil_bytes);
            r.action       = FOX_GOV_GROW;
            r.budget_bytes = g->budget;
            r.reason       = reason;
        } else {
            r.reason = "quiet, waiting out hysteresis";
        }
    }

    g->prev      = now;
    g->prev_ns   = t;
    g->have_prev = 1;

    if (out) *out = r;
    return FOX_OK;
}

void fox_governor_panic(fox_governor *g, const char *reason)
{
    uint64_t next;

    if (!g) return;

    next = g->budget / 2;
    if (next < g->cfg.floor_bytes) next = g->cfg.floor_bytes;

    FOX_WARN("governor panic: %s (budget %llu -> %llu)",
             reason ? reason : "unspecified",
             (unsigned long long)g->budget, (unsigned long long)next);

    g->budget      = next;
    g->quiet_ticks = 0;
    g->cooldown    = FOX_GOV_COOLDOWN_TICKS * 2;
}
