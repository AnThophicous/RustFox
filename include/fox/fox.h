#ifndef FOX_H
#define FOX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FOX_VERSION_MAJOR 0
#define FOX_VERSION_MINOR 1
#define FOX_VERSION_PATCH 0

const char *fox_version(void);

typedef enum {
    FOX_OK = 0,
    FOX_ERR_ARG,
    FOX_ERR_IO,
    FOX_ERR_NOMEM,
    FOX_ERR_FORMAT,
    FOX_ERR_UNSUPPORTED,
    FOX_ERR_NOTFOUND,
    FOX_ERR_INTERNAL
} fox_status;

const char *fox_status_str(fox_status s);

const char *fox_last_error(void);

#define FOX_PATH_MAX   1024
#define FOX_NAME_MAX   128

enum {
    FOX_CPU_SSE2      = 1u << 0,
    FOX_CPU_SSSE3     = 1u << 1,
    FOX_CPU_SSE41     = 1u << 2,
    FOX_CPU_SSE42     = 1u << 3,
    FOX_CPU_POPCNT    = 1u << 4,
    FOX_CPU_AVX       = 1u << 5,
    FOX_CPU_F16C      = 1u << 6,
    FOX_CPU_FMA       = 1u << 7,
    FOX_CPU_AVX2      = 1u << 8,
    FOX_CPU_AVX512F   = 1u << 9,
    FOX_CPU_AVX512BW  = 1u << 10,
    FOX_CPU_AVX512VNNI= 1u << 11,
    FOX_CPU_AVXVNNI   = 1u << 12,
    FOX_CPU_NEON      = 1u << 16,
    FOX_CPU_DOTPROD   = 1u << 17,
    FOX_CPU_I8MM      = 1u << 18,
    FOX_CPU_SVE       = 1u << 19
};

typedef struct {
    char     vendor[16];
    char     brand[64];
    char     uarch[32];
    int      logical_cores;
    int      physical_cores;
    int      numa_nodes;
    uint32_t features;
    int      cacheline_bytes;
    uint64_t l1d_bytes;
    uint64_t l2_bytes;
    uint64_t l3_bytes;
    double   cpu_quota;
} fox_cpu_info;

char *fox_cpu_features_str(uint32_t features, char *buf, size_t buflen);

typedef struct {
    uint64_t page_size;
    uint64_t total_bytes;
    uint64_t host_total_bytes;
    uint64_t available_bytes;
    uint64_t free_bytes;
    uint64_t cached_bytes;
    uint64_t swap_total_bytes;
    uint64_t swap_free_bytes;

    int      in_container;
    int      cgroup_version;
    uint64_t cgroup_limit;
    uint64_t cgroup_high;
    uint64_t cgroup_current;

    int      swappiness;
    int      overcommit_mode;
    uint64_t hugepage_bytes;
} fox_mem_info;

typedef struct {
    char     path[FOX_PATH_MAX];
    char     device[FOX_NAME_MAX];
    char     fstype[32];
    int      rotational;
    int      supports_odirect;
    uint64_t capacity_bytes;
    uint64_t free_bytes;

    double   seq_read_mbps;
    double   rand_read_4k_iops;
    double   rand_read_lat_us;
    int      measured;
} fox_storage_info;

typedef struct {
    int      present;
    char     name[FOX_NAME_MAX];
    char     api[16];
    uint64_t vram_bytes;
    int      unified_memory;
} fox_gpu_info;

typedef struct {
    fox_cpu_info     cpu;
    fox_mem_info     mem;
    fox_storage_info storage;
    fox_gpu_info     gpu;
    char             os[64];
    char             kernel[64];
} fox_sysinfo;

fox_status fox_probe(fox_sysinfo *out, const char *scratch_dir, int measure_storage);

void fox_sysinfo_print(const fox_sysinfo *si, void *fp);

fox_status fox_sysinfo_to_json(const fox_sysinfo *si, char *buf, size_t buflen,
                               size_t *written);

typedef struct {
    int    available;
    double some_avg10, some_avg60, some_avg300;
    double full_avg10, full_avg60, full_avg300;
    uint64_t some_total_us, full_total_us;
} fox_psi;

typedef struct {
    fox_psi  cpu, mem, io;
    uint64_t pgmajfault;
    uint64_t pswpin;
    uint64_t cgroup_high_events;
    uint64_t cgroup_max_events;
    int      available;
} fox_pressure;

fox_status fox_pressure_read(fox_pressure *out);

typedef struct {
    uint64_t floor_bytes;
    uint64_t ceil_bytes;
    double   target_mem_stall;
    double   target_io_stall;
    double   target_majflt_per_s;
    double   increase_frac;
    double   decrease_factor;
    uint32_t tick_ms;
} fox_governor_config;

void fox_governor_config_default(fox_governor_config *cfg, const fox_sysinfo *si);

typedef struct fox_governor fox_governor;

fox_governor *fox_governor_create(const fox_governor_config *cfg,
                                  uint64_t initial_budget_bytes);
void          fox_governor_destroy(fox_governor *g);

typedef enum {
    FOX_GOV_HOLD = 0,
    FOX_GOV_GROW,
    FOX_GOV_CUT
} fox_gov_action;

typedef struct {
    fox_gov_action action;
    uint64_t       budget_bytes;
    uint64_t       prev_budget_bytes;
    const char    *reason;
    double         mem_stall;
    double         io_stall;
    double         majflt_per_s;
    double         swapin_per_s;
} fox_gov_tick;

fox_status fox_governor_tick(fox_governor *g, fox_gov_tick *out);

uint64_t fox_governor_budget(const fox_governor *g);

void fox_governor_panic(fox_governor *g, const char *reason);

typedef enum {
    FOX_LOG_ERROR = 0,
    FOX_LOG_WARN,
    FOX_LOG_INFO,
    FOX_LOG_DEBUG,
    FOX_LOG_TRACE
} fox_log_level;

void fox_log_set_level(fox_log_level lv);
fox_log_level fox_log_get_level(void);
void fox_log_init_from_env(void);

#ifdef __cplusplus
}
#endif
#endif
