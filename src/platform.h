#ifndef FOX_PLATFORM_H
#define FOX_PLATFORM_H

#include "fox/fox.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define FOX_OS_WINDOWS 1
#elif defined(__APPLE__)
#  define FOX_OS_MACOS 1
#  define FOX_OS_POSIX 1
#elif defined(__linux__)
#  define FOX_OS_LINUX 1
#  define FOX_OS_POSIX 1
#else
#  define FOX_OS_POSIX 1
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define FOX_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define FOX_ARCH_ARM64 1
#endif

uint64_t fox_now_ns(void);
void     fox_sleep_ms(uint32_t ms);

size_t fox_page_size(void);

void  *fox_aligned_alloc(size_t alignment, size_t size);
void   fox_aligned_free(void *p);

typedef struct fox_file fox_file;

enum {
    FOX_OPEN_DIRECT = 1u << 0,
    FOX_OPEN_RANDOM = 1u << 1,
    FOX_OPEN_SEQ    = 1u << 2
};

fox_file *fox_file_open_read(const char *path, unsigned flags, int *direct_granted);
void      fox_file_close(fox_file *f);
int64_t   fox_file_size(fox_file *f);

int64_t   fox_file_pread(fox_file *f, void *buf, size_t n, uint64_t offset);

int       fox_file_drop_cache(fox_file *f);

void      fox_sync_path(const char *path);

int64_t   fox_read_small_file(const char *path, char *buf, size_t cap);

fox_status fox_read_u64_file(const char *path, uint64_t *out);

int fox_path_exists(const char *path);

char *fox_temp_dir(char *buf, size_t cap);

char *fox_cache_dir(char *buf, size_t cap);

fox_status fox_mkdir_p(const char *path);

fox_status fox_fs_space(const char *path, uint64_t *capacity, uint64_t *avail);

int fox_cpu_count_online(void);

typedef struct fox_thread fox_thread;
typedef struct fox_mutex  fox_mutex;
typedef struct fox_cond   fox_cond;

fox_thread *fox_thread_start(void (*entry)(void *), void *arg);
void        fox_thread_join(fox_thread *t);

fox_mutex *fox_mutex_create(void);
void       fox_mutex_destroy(fox_mutex *m);
void       fox_mutex_lock(fox_mutex *m);
void       fox_mutex_unlock(fox_mutex *m);

fox_cond *fox_cond_create(void);
void      fox_cond_destroy(fox_cond *c);
void      fox_cond_wait(fox_cond *c, fox_mutex *m);
void      fox_cond_broadcast(fox_cond *c);

void fox_probe_cpu(fox_cpu_info *out);

void fox_plat_cpu_topology(fox_cpu_info *out);

void fox_probe_mem(fox_mem_info *out);
void fox_probe_gpu(fox_gpu_info *out);
void fox_probe_os(char *os, size_t os_cap, char *kernel, size_t kernel_cap);

void fox_probe_storage_static(const char *path, fox_storage_info *out);

fox_status fox_bench_storage(const char *dir, uint64_t probe_bytes,
                             fox_storage_info *io);

#endif
