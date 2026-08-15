#include "fox/fox.h"
#include "check.h"

#include <string.h>

static void make_sysinfo(fox_sysinfo *si, uint64_t total, uint64_t available)
{
    memset(si, 0, sizeof(*si));
    si->mem.total_bytes      = total;
    si->mem.host_total_bytes = total;
    si->mem.available_bytes  = available;
    si->mem.page_size        = 4096;
    si->mem.cgroup_limit     = UINT64_MAX;
    si->mem.cgroup_high      = UINT64_MAX;
    si->cpu.logical_cores    = 4;
    si->cpu.physical_cores   = 4;
}

int main(void)
{
    fox_governor_config cfg;
    fox_sysinfo si;
    fox_governor *g;
    fox_gov_tick tick;
    fox_pressure pr;
    fox_status st;
    uint64_t prev;
    int i;

    make_sysinfo(&si, 4ull << 30, 3ull << 30);
    fox_governor_config_default(&cfg, &si);

    CHECK(cfg.floor_bytes > 0, "default floor is non-zero");
    CHECK(cfg.ceil_bytes >= cfg.floor_bytes, "default ceiling is above the floor");
    CHECK(cfg.ceil_bytes < si.mem.total_bytes,
          "ceiling leaves headroom below total memory");
    CHECK(cfg.ceil_bytes <= si.mem.available_bytes,
          "ceiling never exceeds what is actually available");
    CHECK(cfg.decrease_factor > 0.0 && cfg.decrease_factor < 1.0,
          "multiplicative decrease actually decreases");
    CHECK(cfg.increase_frac > 0.0 && cfg.increase_frac < 1.0,
          "additive increase step is a sane fraction");
    CHECK(cfg.tick_ms > 0, "tick period is non-zero");

    make_sysinfo(&si, 512ull << 20, 400ull << 20);
    fox_governor_config_default(&cfg, &si);
    CHECK(cfg.ceil_bytes >= cfg.floor_bytes,
          "a tiny container still produces a valid range");
    CHECK(cfg.floor_bytes <= 512ull << 20, "floor fits inside a tiny container");

    fox_governor_config_default(&cfg, NULL);
    CHECK(cfg.ceil_bytes >= cfg.floor_bytes,
          "defaults are valid with no probe at all");

    make_sysinfo(&si, 4ull << 30, 3ull << 30);
    si.mem.cgroup_high = 1ull << 30;
    fox_governor_config_default(&cfg, &si);
    CHECK(cfg.ceil_bytes < (1ull << 30),
          "memory.high pulls the ceiling below the throttle point");

    make_sysinfo(&si, 4ull << 30, 3ull << 30);
    fox_governor_config_default(&cfg, &si);

    g = fox_governor_create(&cfg, cfg.ceil_bytes * 4);
    CHECK(g != NULL, "governor is created");
    CHECK(fox_governor_budget(g) == cfg.ceil_bytes,
          "an oversized initial budget is clamped to the ceiling");
    fox_governor_destroy(g);

    g = fox_governor_create(&cfg, 1);
    CHECK(fox_governor_budget(g) == cfg.floor_bytes,
          "an undersized initial budget is raised to the floor");
    fox_governor_destroy(g);

    g = fox_governor_create(&cfg, 0);
    CHECK(g != NULL, "a zero initial budget picks a midpoint");
    CHECK(fox_governor_budget(g) >= cfg.floor_bytes &&
          fox_governor_budget(g) <= cfg.ceil_bytes,
          "the midpoint lands inside the range");

    prev = fox_governor_budget(g);
    fox_governor_panic(g, "test");
    CHECK(fox_governor_budget(g) < prev, "panic cuts the budget");

    for (i = 0; i < 64; i++) fox_governor_panic(g, "test");
    CHECK(fox_governor_budget(g) == cfg.floor_bytes,
          "repeated panic converges on the floor and stops");

    fox_governor_destroy(g);

    g = fox_governor_create(&cfg, 0);
    for (i = 0; i < 12; i++) {
        st = fox_governor_tick(g, &tick);
        CHECK(st == FOX_OK, "tick succeeds");
        CHECK(tick.budget_bytes >= cfg.floor_bytes &&
              tick.budget_bytes <= cfg.ceil_bytes,
              "budget stays inside the configured range");
        CHECK(tick.budget_bytes == fox_governor_budget(g),
              "reported budget matches the governor state");
        CHECK(tick.reason != NULL && tick.reason[0] != '\0',
              "every tick explains itself");
        CHECK(tick.mem_stall >= 0.0 && tick.io_stall >= 0.0,
              "stall percentages are never negative");
        if (tick.action == FOX_GOV_CUT)
            CHECK(tick.budget_bytes <= tick.prev_budget_bytes,
                  "a cut never raises the budget");
        if (tick.action == FOX_GOV_GROW)
            CHECK(tick.budget_bytes >= tick.prev_budget_bytes,
                  "a grow never lowers the budget");
    }
    fox_governor_destroy(g);

    CHECK(fox_governor_budget(NULL) == 0, "a null governor reports no budget");
    fox_governor_destroy(NULL);
    fox_governor_panic(NULL, "test");
    CHECK(fox_governor_tick(NULL, &tick) == FOX_ERR_ARG,
          "ticking a null governor is an argument error");

    st = fox_pressure_read(&pr);
    CHECK(st == FOX_OK || st == FOX_ERR_UNSUPPORTED,
          "pressure read either works or says the platform lacks it");
    if (st == FOX_OK) {
        CHECK(pr.available == 1, "a successful pressure read is marked available");
        CHECK(pr.mem.some_avg10 >= 0.0, "memory stall is non-negative");
    }
    CHECK(fox_pressure_read(NULL) == FOX_ERR_ARG,
          "a null pressure destination is an argument error");

    CHECK_DONE();
}
