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

#define FOX_GGUF_MAX_DIMS 4

typedef enum {
    FOX_GGUF_UINT8 = 0, FOX_GGUF_INT8, FOX_GGUF_UINT16, FOX_GGUF_INT16,
    FOX_GGUF_UINT32, FOX_GGUF_INT32, FOX_GGUF_FLOAT32, FOX_GGUF_BOOL,
    FOX_GGUF_STRING, FOX_GGUF_ARRAY, FOX_GGUF_UINT64, FOX_GGUF_INT64,
    FOX_GGUF_FLOAT64
} fox_gguf_value_type;

typedef enum {
    FOX_GGML_F32 = 0, FOX_GGML_F16 = 1, FOX_GGML_Q4_0 = 2,
    FOX_GGML_Q4_1 = 3, FOX_GGML_Q5_0 = 6, FOX_GGML_Q5_1 = 7,
    FOX_GGML_Q8_0 = 8, FOX_GGML_Q2_K = 10, FOX_GGML_Q3_K = 11,
    FOX_GGML_Q4_K = 12, FOX_GGML_Q5_K = 13, FOX_GGML_Q6_K = 14,
    FOX_GGML_Q8_K = 15, FOX_GGML_IQ2_XXS = 16, FOX_GGML_IQ2_XS = 17,
    FOX_GGML_IQ3_XXS = 18, FOX_GGML_IQ1_S = 19, FOX_GGML_IQ4_NL = 20,
    FOX_GGML_IQ3_S = 21, FOX_GGML_IQ2_S = 22, FOX_GGML_IQ4_XS = 23,
    FOX_GGML_I8 = 24, FOX_GGML_I16 = 25, FOX_GGML_I32 = 26,
    FOX_GGML_I64 = 27, FOX_GGML_F64 = 28, FOX_GGML_IQ1_M = 29,
    FOX_GGML_BF16 = 30, FOX_GGML_TQ1_0 = 34, FOX_GGML_TQ2_0 = 35
} fox_ggml_type;

typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t ne[FOX_GGUF_MAX_DIMS];
    fox_ggml_type type;
    uint64_t offset;
    uint64_t size_bytes;
} fox_gguf_tensor;

typedef struct fox_gguf fox_gguf;

fox_status fox_gguf_open(const char *path, fox_gguf **out);
void fox_gguf_close(fox_gguf *g);
uint32_t fox_gguf_version(const fox_gguf *g);
size_t fox_gguf_metadata_count(const fox_gguf *g);
size_t fox_gguf_tensor_count(const fox_gguf *g);
size_t fox_gguf_find(const fox_gguf *g, const char *key);
fox_gguf_value_type fox_gguf_metadata_type(const fox_gguf *g, size_t index);
size_t fox_gguf_metadata_array_count(const fox_gguf *g, size_t index);
fox_gguf_value_type fox_gguf_metadata_array_type(const fox_gguf *g, size_t index);
fox_status fox_gguf_get_u32(const fox_gguf *g, size_t index, uint32_t *out);
fox_status fox_gguf_get_bool(const fox_gguf *g, size_t index, int *out);
fox_status fox_gguf_get_string(const fox_gguf *g, size_t index, const char **out);
fox_status fox_gguf_get_array_string(const fox_gguf *g, size_t index,
                                     size_t element, const char **out);
fox_status fox_gguf_get_array_i32(const fox_gguf *g, size_t index,
                                  size_t element, int32_t *out);
fox_status fox_gguf_get_array_f32(const fox_gguf *g, size_t index,
                                  size_t element, float *out);
fox_status fox_gguf_tensor_at(const fox_gguf *g, size_t index, fox_gguf_tensor *out);
size_t fox_gguf_tensor_find(const fox_gguf *g, const char *name);
fox_status fox_gguf_read_tensor(const fox_gguf *g, size_t index,
                                 void *dst, size_t capacity);

#define FOX_TOKEN_INVALID UINT32_MAX

typedef struct fox_tokenizer fox_tokenizer;

fox_status fox_tokenizer_open(const fox_gguf *model, fox_tokenizer **out);
void fox_tokenizer_close(fox_tokenizer *tokenizer);
size_t fox_tokenizer_vocab_size(const fox_tokenizer *tokenizer);
uint32_t fox_tokenizer_bos(const fox_tokenizer *tokenizer);
uint32_t fox_tokenizer_eos(const fox_tokenizer *tokenizer);
fox_status fox_tokenizer_encode(const fox_tokenizer *tokenizer, const char *text,
                                uint32_t *tokens, size_t capacity, size_t *written);
fox_status fox_tokenizer_decode(const fox_tokenizer *tokenizer,
                                const uint32_t *tokens, size_t count,
                                char *text, size_t capacity, size_t *written);
const char *fox_ggml_type_name(fox_ggml_type type);
uint32_t fox_ggml_type_block_size(fox_ggml_type type);
uint32_t fox_ggml_type_block_bytes(fox_ggml_type type);

typedef enum {
    FOX_PLAN_PIN = 0,
    FOX_PLAN_STREAM,
    FOX_PLAN_MMAP
} fox_plan_placement;

typedef struct {
    uint64_t residency_budget_bytes;
    double seq_read_mbps;
    double compute_tok_s;
} fox_plan_config;

typedef struct {
    size_t tensor_index;
    fox_plan_placement placement;
    uint64_t bytes;
    uint32_t priority;
    const char *reason;
} fox_plan_item;

typedef struct {
    uint64_t model_bytes;
    uint64_t pinned_bytes;
    uint64_t mmap_bytes;
    uint64_t streamed_bytes;
    double io_seconds;
    double compute_seconds;
    double predicted_tok_s;
} fox_plan_summary;

typedef struct fox_plan fox_plan;

fox_status fox_plan_build(const fox_gguf *model, const fox_plan_config *config,
                          fox_plan **out);
void fox_plan_destroy(fox_plan *plan);
size_t fox_plan_count(const fox_plan *plan);
fox_status fox_plan_item_at(const fox_plan *plan, size_t index,
                            fox_plan_item *out);
fox_status fox_plan_get_summary(const fox_plan *plan, fox_plan_summary *out);

fox_status fox_tq1_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out);
fox_status fox_tq2_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out);
fox_status fox_tq1_dot_i8_scalar(const int8_t *activations, const void *block_data,
                                 size_t n, float *out);
fox_status fox_tq2_dot_i8_scalar(const int8_t *activations, const void *block_data,
                                 size_t n, float *out);

typedef struct fox_threadpool fox_threadpool;

fox_threadpool *fox_threadpool_create(int n_threads);
void            fox_threadpool_destroy(fox_threadpool *tp);
int             fox_threadpool_size(const fox_threadpool *tp);

typedef void (*fox_parallel_fn)(void *ctx, int worker, size_t begin, size_t end);

fox_status fox_parallel_for(fox_threadpool *tp, size_t n,
                            fox_parallel_fn fn, void *ctx);

fox_status fox_quantize_activations_i8(const float *x, size_t n,
                                       int8_t *q, float *scale);

fox_status fox_gemv_tq1_f32(fox_threadpool *tp, const float *x, const void *weights,
                            size_t n_rows, size_t n_cols, float *out);
fox_status fox_gemv_tq2_f32(fox_threadpool *tp, const float *x, const void *weights,
                            size_t n_rows, size_t n_cols, float *out);

fox_status fox_gemv_tq1_i8(fox_threadpool *tp, const int8_t *q, float act_scale,
                           const void *weights, size_t n_rows, size_t n_cols,
                           float *out);
fox_status fox_gemv_tq2_i8(fox_threadpool *tp, const int8_t *q, float act_scale,
                           const void *weights, size_t n_rows, size_t n_cols,
                           float *out);

#define FOX_WEIGHTS_MAX_LIVE_LEASES 8

typedef enum {
    FOX_WEIGHTS_RESIDENT = 0,
    FOX_WEIGHTS_STREAM   = 1
} fox_weights_mode;

typedef struct fox_weights fox_weights;

typedef struct {
    const void *data;
    uint64_t    bytes;
    size_t      tensor_index;
    void       *slot;
} fox_weights_lease;

fox_status fox_weights_open(const char *gguf_path, fox_weights_mode mode,
                            uint64_t budget_bytes, fox_weights **out);
void       fox_weights_close(fox_weights *w);

const fox_gguf *fox_weights_model(const fox_weights *w);
fox_weights_mode fox_weights_get_mode(const fox_weights *w);

fox_status fox_weights_acquire(fox_weights *w, size_t tensor_index,
                               fox_weights_lease *lease);
void       fox_weights_release(fox_weights *w, fox_weights_lease *lease);
fox_status fox_weights_prefetch(fox_weights *w, size_t tensor_index);

size_t     fox_weights_live_leases(const fox_weights *w);
uint64_t   fox_weights_resident_bytes(const fox_weights *w);

#ifdef __cplusplus
}
#endif
#endif
