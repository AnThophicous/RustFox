#include "platform.h"

#if defined(FOX_OS_POSIX)

#include "fox_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(FOX_OS_LINUX)
#  include <sched.h>
#  include <sys/utsname.h>
#endif
#if defined(FOX_OS_MACOS)
#  include <sys/sysctl.h>
#  include <sys/utsname.h>
#endif

uint64_t fox_now_ns(void)
{
    struct timespec ts;
    int ok = -1;
#if defined(CLOCK_MONOTONIC_RAW) && defined(FOX_OS_LINUX)
    ok = clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#endif
    if (ok != 0) ok = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (ok != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void fox_sleep_ms(uint32_t ms)
{
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {  }
}

size_t fox_page_size(void)
{
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t)p : 4096u;
}

void *fox_aligned_alloc(size_t alignment, size_t size)
{
    void *p = NULL;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if ((alignment & (alignment - 1)) != 0) return NULL;
    if (size == 0) size = 1;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}

void fox_aligned_free(void *p) { free(p); }

struct fox_file {
    int fd;
    int direct;
};

fox_file *fox_file_open_read(const char *path, unsigned flags, int *direct_granted)
{
    fox_file *f;
    int fd = -1;
    int granted = 0;
    int base = O_RDONLY;

#ifdef O_CLOEXEC
    base |= O_CLOEXEC;
#endif

#if defined(FOX_OS_LINUX) && defined(O_DIRECT)
    if (flags & FOX_OPEN_DIRECT) {
        fd = open(path, base | O_DIRECT);
        if (fd >= 0) granted = 1;
    }
#endif
    if (fd < 0) fd = open(path, base);
    if (fd < 0) {
        fox_fail(FOX_ERR_IO, "open %s: %s", path, strerror(errno));
        if (direct_granted) *direct_granted = 0;
        return NULL;
    }

#if defined(FOX_OS_MACOS)
    if ((flags & FOX_OPEN_DIRECT) && fcntl(fd, F_NOCACHE, 1) == 0) granted = 1;
#endif
#if defined(POSIX_FADV_RANDOM)
    if (flags & FOX_OPEN_RANDOM) (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
#if defined(POSIX_FADV_SEQUENTIAL)
    if (flags & FOX_OPEN_SEQ) (void)posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    f = (fox_file *)calloc(1, sizeof(*f));
    if (!f) {
        close(fd);
        fox_fail(FOX_ERR_NOMEM, "fox_file alloc");
        if (direct_granted) *direct_granted = 0;
        return NULL;
    }
    f->fd = fd;
    f->direct = granted;
    if (direct_granted) *direct_granted = granted;
    return f;
}

void fox_file_close(fox_file *f)
{
    if (!f) return;
    if (f->fd >= 0) close(f->fd);
    free(f);
}

int64_t fox_file_size(fox_file *f)
{
    struct stat st;
    if (!f || fstat(f->fd, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

int64_t fox_file_pread(fox_file *f, void *buf, size_t n, uint64_t offset)
{
    size_t done = 0;
    if (!f || !buf) return -1;
    while (done < n) {
        ssize_t r = pread(f->fd, (char *)buf + done, n - done,
                          (off_t)(offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return done > 0 ? (int64_t)done : -1;
        }
        if (r == 0) break;
        done += (size_t)r;
    }
    return (int64_t)done;
}

int fox_file_drop_cache(fox_file *f)
{
    if (!f) return 0;
#if defined(FOX_OS_LINUX) && defined(POSIX_FADV_DONTNEED)
    (void)fsync(f->fd);
    return posix_fadvise(f->fd, 0, 0, POSIX_FADV_DONTNEED) == 0;
#elif defined(FOX_OS_MACOS)
    return fcntl(f->fd, F_NOCACHE, 1) == 0;
#else
    return 0;
#endif
}

void fox_sync_path(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    (void)fsync(fd);
#if defined(FOX_OS_LINUX) && defined(POSIX_FADV_DONTNEED)
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(fd);
}

int fox_path_exists(const char *path)
{
    struct stat st;
    return path && stat(path, &st) == 0;
}

char *fox_temp_dir(char *buf, size_t cap)
{
    const char *e = getenv("TMPDIR");
    if (!e || !*e) e = getenv("TMP");
    if (!e || !*e) e = "/tmp";
    fox_strlcpy(buf, e, cap);
    return buf;
}

fox_status fox_mkdir_p(const char *path)
{
    char tmp[FOX_PATH_MAX];
    size_t len, i;
    struct stat st;

    if (!path || !*path) return FOX_ERR_ARG;
    fox_strlcpy(tmp, path, sizeof(tmp));
    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') tmp[--len] = '\0';

    for (i = 1; i < len; i++) {
        if (tmp[i] != '/') continue;
        tmp[i] = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
            return fox_fail(FOX_ERR_IO, "mkdir %s: %s", tmp, strerror(errno));
        tmp[i] = '/';
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        return fox_fail(FOX_ERR_IO, "mkdir %s: %s", tmp, strerror(errno));
    if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode))
        return fox_fail(FOX_ERR_IO, "%s is not a directory", tmp);
    return FOX_OK;
}

fox_status fox_fs_space(const char *path, uint64_t *capacity, uint64_t *avail)
{
    struct statvfs vfs;
    uint64_t unit;
    if (!path) return FOX_ERR_ARG;
    if (statvfs(path, &vfs) != 0)
        return fox_fail(FOX_ERR_IO, "statvfs %s: %s", path, strerror(errno));
    unit = vfs.f_frsize ? (uint64_t)vfs.f_frsize : (uint64_t)vfs.f_bsize;
    if (capacity) *capacity = (uint64_t)vfs.f_blocks * unit;
    if (avail)    *avail    = (uint64_t)vfs.f_bavail * unit;
    return FOX_OK;
}

int fox_cpu_count_online(void)
{
#if defined(FOX_OS_LINUX) && defined(CPU_COUNT)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return n > 0 ? (int)n : 1;
    }
}

#if defined(FOX_OS_LINUX)

static uint64_t linux_sysfs_cache_size(int cpu, int level, const char *type)
{
    char path[256];
    char buf[64];
    int idx;
    for (idx = 0; idx < 10; idx++) {
        int lv = 0;
        long v;
        char *end;
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, idx);
        if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) continue;
        lv = atoi(buf);
        if (lv != level) continue;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpu, idx);
        if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) continue;
        if (type && strncmp(buf, type, strlen(type)) != 0) continue;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, idx);
        if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) continue;
        v = strtol(buf, &end, 10);
        if (v <= 0) continue;
        if (end && (*end == 'K' || *end == 'k')) return (uint64_t)v * 1024ull;
        if (end && (*end == 'M' || *end == 'm')) return (uint64_t)v * 1024ull * 1024ull;
        return (uint64_t)v;
    }
    return 0;
}

static int linux_physical_cores(int logical)
{
    int seen_pkg[64];
    int seen_core[64];
    int n = 0;
    int cpu;

    for (cpu = 0; cpu < 4096 && n < 64; cpu++) {
        char path[256];
        char buf[64];
        int core_id, pkg_id, i, dup = 0;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) {
            if (cpu > logical + 8) break;
            continue;
        }
        core_id = atoi(buf);

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        pkg_id = (fox_read_small_file(path, buf, sizeof(buf)) > 0) ? atoi(buf) : 0;

        for (i = 0; i < n; i++)
            if (seen_core[i] == core_id && seen_pkg[i] == pkg_id) { dup = 1; break; }
        if (dup) continue;
        seen_core[n] = core_id;
        seen_pkg[n]  = pkg_id;
        n++;
    }
    return n > 0 ? n : logical;
}

static int linux_numa_nodes(void)
{
    int n = 0, i;
    for (i = 0; i < 256; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
        if (!fox_path_exists(path)) break;
        n++;
    }
    return n > 0 ? n : 1;
}

static double linux_cpu_quota(void)
{
    char buf[128];
    if (fox_read_small_file("/sys/fs/cgroup/cpu.max", buf, sizeof(buf)) > 0) {
        char q[32];
        unsigned long long period = 0;
        if (sscanf(buf, "%31s %llu", q, &period) == 2 && period > 0) {
            if (strcmp(q, "max") == 0) return 0.0;
            return (double)strtoull(q, NULL, 10) / (double)period;
        }
    }
    {
        char pbuf[64];
        long long quota;
        long long period;
        if (fox_read_small_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", buf, sizeof(buf)) > 0 &&
            fox_read_small_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", pbuf, sizeof(pbuf)) > 0) {
            quota  = strtoll(buf, NULL, 10);
            period = strtoll(pbuf, NULL, 10);
            if (quota > 0 && period > 0) return (double)quota / (double)period;
        }
    }
    return 0.0;
}

#endif

void fox_plat_cpu_topology(fox_cpu_info *out)
{
    out->logical_cores = fox_cpu_count_online();

#if defined(FOX_OS_LINUX)
    out->physical_cores  = linux_physical_cores(out->logical_cores);
    out->numa_nodes      = linux_numa_nodes();
    out->l1d_bytes       = linux_sysfs_cache_size(0, 1, "Data");
    out->l2_bytes        = linux_sysfs_cache_size(0, 2, NULL);
    out->l3_bytes        = linux_sysfs_cache_size(0, 3, NULL);
    out->cpu_quota       = linux_cpu_quota();
#elif defined(FOX_OS_MACOS)
    {
        size_t sz;
        int64_t v;
        uint32_t u32;
        sz = sizeof(u32);
        if (sysctlbyname("hw.physicalcpu", &u32, &sz, NULL, 0) == 0)
            out->physical_cores = (int)u32;
        sz = sizeof(v);
        if (sysctlbyname("hw.l1dcachesize", &v, &sz, NULL, 0) == 0) out->l1d_bytes = (uint64_t)v;
        sz = sizeof(v);
        if (sysctlbyname("hw.l2cachesize", &v, &sz, NULL, 0) == 0)  out->l2_bytes  = (uint64_t)v;
        sz = sizeof(v);
        if (sysctlbyname("hw.l3cachesize", &v, &sz, NULL, 0) == 0)  out->l3_bytes  = (uint64_t)v;
        out->numa_nodes = 1;
    }
#endif

    if (out->physical_cores <= 0) out->physical_cores = out->logical_cores;
    if (out->numa_nodes <= 0)     out->numa_nodes = 1;
}

#if defined(FOX_OS_LINUX)

static uint64_t meminfo_field(const char *text, const char *key)
{
    size_t klen = strlen(key);
    const char *p = text;

    while (p && *p) {
        if (strncmp(p, key, klen) == 0 && p[klen] == ':') {
            unsigned long long value;
            const char *q = p + klen + 1;
            while (*q == ' ' || *q == '\t') q++;
            value = strtoull(q, NULL, 10);
            while (*q >= '0' && *q <= '9') q++;
            while (*q == ' ' || *q == '\t') q++;
            if (q[0] == 'k' && q[1] == 'B') return (uint64_t)value * 1024ull;
            return (uint64_t)value;
        }
        p = strchr(p, '\n');
        if (p) p++;
    }
    return 0;
}

static int linux_in_container(void)
{
    char buf[4096];
    if (fox_path_exists("/.dockerenv")) return 1;
    if (fox_path_exists("/run/.containerenv")) return 1;
    if (fox_read_small_file("/proc/1/cgroup", buf, sizeof(buf)) > 0) {
        if (strstr(buf, "docker") || strstr(buf, "kubepods") ||
            strstr(buf, "containerd") || strstr(buf, "lxc") ||
            strstr(buf, "libpod"))
            return 1;
    }
    return 0;
}

static void linux_cgroup_mem(fox_mem_info *m)
{
    char path[FOX_PATH_MAX + 64];
    char cg[1024];
    uint64_t v;

    m->cgroup_version = 0;
    m->cgroup_limit   = UINT64_MAX;
    m->cgroup_high    = UINT64_MAX;
    m->cgroup_current = 0;

    if (fox_read_u64_file("/sys/fs/cgroup/memory.max", &v) == FOX_OK) {
        m->cgroup_version = 2;
        m->cgroup_limit = v;
        if (fox_read_u64_file("/sys/fs/cgroup/memory.high", &v) == FOX_OK) m->cgroup_high = v;
        if (fox_read_u64_file("/sys/fs/cgroup/memory.current", &v) == FOX_OK) m->cgroup_current = v;
        return;
    }

    if (fox_read_small_file("/proc/self/cgroup", cg, sizeof(cg)) > 0) {
        char *line = strstr(cg, "0::");
        if (line) {
            char *nl;
            line += 3;
            nl = strchr(line, '\n');
            if (nl) *nl = '\0';
            snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.max", line);
            if (fox_read_u64_file(path, &v) == FOX_OK) {
                m->cgroup_version = 2;
                m->cgroup_limit = v;
                snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.high", line);
                if (fox_read_u64_file(path, &v) == FOX_OK) m->cgroup_high = v;
                snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.current", line);
                if (fox_read_u64_file(path, &v) == FOX_OK) m->cgroup_current = v;
                return;
            }
        }
    }

    if (fox_read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", &v) == FOX_OK) {
        m->cgroup_version = 1;
        m->cgroup_limit = (v >= (1ull << 62)) ? UINT64_MAX : v;
        if (fox_read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes", &v) == FOX_OK)
            m->cgroup_current = v;
    }
}

#endif

void fox_probe_mem(fox_mem_info *out)
{
    memset(out, 0, sizeof(*out));
    out->page_size      = (uint64_t)fox_page_size();
    out->cgroup_limit   = UINT64_MAX;
    out->cgroup_high    = UINT64_MAX;
    out->swappiness     = -1;
    out->overcommit_mode = -1;

#if defined(FOX_OS_LINUX)
    {
        char mi[8192];
        char small[64];
        if (fox_read_small_file("/proc/meminfo", mi, sizeof(mi)) > 0) {
            out->host_total_bytes  = meminfo_field(mi, "MemTotal");
            out->free_bytes        = meminfo_field(mi, "MemFree");
            out->available_bytes   = meminfo_field(mi, "MemAvailable");
            out->cached_bytes      = meminfo_field(mi, "Cached");
            out->swap_total_bytes  = meminfo_field(mi, "SwapTotal");
            out->swap_free_bytes   = meminfo_field(mi, "SwapFree");
            out->hugepage_bytes    = meminfo_field(mi, "Hugepagesize");
            if (out->available_bytes == 0) out->available_bytes = out->free_bytes;
        }
        if (fox_read_small_file("/proc/sys/vm/swappiness", small, sizeof(small)) > 0)
            out->swappiness = atoi(small);
        if (fox_read_small_file("/proc/sys/vm/overcommit_memory", small, sizeof(small)) > 0)
            out->overcommit_mode = atoi(small);

        out->in_container = linux_in_container();
        linux_cgroup_mem(out);
    }
#elif defined(FOX_OS_MACOS)
    {
        size_t sz = sizeof(uint64_t);
        uint64_t v = 0;
        if (sysctlbyname("hw.memsize", &v, &sz, NULL, 0) == 0) out->host_total_bytes = v;
        out->available_bytes = out->host_total_bytes / 2;
        out->free_bytes      = out->available_bytes;
    }
#else
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long avail = -1;
#  if defined(_SC_AVPHYS_PAGES)
        avail = sysconf(_SC_AVPHYS_PAGES);
#  endif
        if (pages > 0) out->host_total_bytes = (uint64_t)pages * out->page_size;
        if (avail > 0) out->available_bytes  = (uint64_t)avail * out->page_size;
        else           out->available_bytes  = out->host_total_bytes / 2;
        out->free_bytes = out->available_bytes;
    }
#endif

    out->total_bytes = out->host_total_bytes;
    if (out->cgroup_limit != UINT64_MAX && out->cgroup_limit > 0 &&
        out->cgroup_limit < out->total_bytes)
        out->total_bytes = out->cgroup_limit;

    if (out->cgroup_limit != UINT64_MAX && out->cgroup_limit > 0) {
        uint64_t headroom = out->cgroup_limit > out->cgroup_current
                          ? out->cgroup_limit - out->cgroup_current : 0;
        if (headroom < out->available_bytes) out->available_bytes = headroom;
    }
}

void fox_probe_gpu(fox_gpu_info *out)
{
    memset(out, 0, sizeof(*out));
    fox_strlcpy(out->api, "none", sizeof(out->api));
    fox_strlcpy(out->name, "none", sizeof(out->name));

#if defined(FOX_OS_MACOS)
    out->present = 1;
    out->unified_memory = 1;
    fox_strlcpy(out->api, "metal", sizeof(out->api));
    fox_strlcpy(out->name, "Apple GPU", sizeof(out->name));
    return;
#elif defined(FOX_OS_LINUX)
    if (fox_path_exists("/dev/nvidiactl") || fox_path_exists("/proc/driver/nvidia")) {
        out->present = 1;
        fox_strlcpy(out->api, "cuda", sizeof(out->api));
        fox_strlcpy(out->name, "NVIDIA GPU", sizeof(out->name));
        return;
    }
    if (fox_path_exists("/dev/dri/renderD128")) {
        char buf[64];
        out->present = 1;
        fox_strlcpy(out->api, "vulkan", sizeof(out->api));
        if (fox_read_small_file("/sys/class/drm/renderD128/device/vendor",
                                buf, sizeof(buf)) > 0) {
            unsigned long vend = strtoul(buf, NULL, 0);
            if (vend == 0x8086) {
                fox_strlcpy(out->name, "Intel integrated GPU", sizeof(out->name));
                out->unified_memory = 1;
            } else if (vend == 0x1002 || vend == 0x1022) {
                uint64_t vram = 0;
                if (fox_read_u64_file(
                        "/sys/class/drm/renderD128/device/mem_info_vram_total",
                        &vram) == FOX_OK && vram > 0) {
                    fox_strlcpy(out->name, "AMD discrete GPU", sizeof(out->name));
                    out->vram_bytes = vram;
                } else {
                    fox_strlcpy(out->name, "AMD integrated GPU", sizeof(out->name));
                    out->unified_memory = 1;
                }
            } else {
                fox_strlcpy(out->name, "DRM render device", sizeof(out->name));
                out->unified_memory = 1;
            }
        } else {
            fox_strlcpy(out->name, "DRM render device", sizeof(out->name));
            out->unified_memory = 1;
        }
        return;
    }
#endif
}

void fox_probe_os(char *os, size_t os_cap, char *kernel, size_t kernel_cap)
{
#if defined(FOX_OS_LINUX) || defined(FOX_OS_MACOS)
    struct utsname u;
    if (uname(&u) == 0) {
        fox_strlcpy(os, u.sysname, os_cap);
        fox_strlcpy(kernel, u.release, kernel_cap);
        return;
    }
#endif
    fox_strlcpy(os, "posix", os_cap);
    fox_strlcpy(kernel, "unknown", kernel_cap);
}

#if defined(FOX_OS_LINUX)

static void linux_find_mount(const char *path, char *dev, size_t dev_cap,
                             char *fstype, size_t fs_cap)
{
    FILE *fp;
    char line[1024];
    char best_dev[FOX_NAME_MAX * 2] = {0};
    char best_fs[64] = {0};
    size_t best_len = 0;
    char real[FOX_PATH_MAX];

    if (!realpath(path, real)) fox_strlcpy(real, path, sizeof(real));

    fp = fopen("/proc/mounts", "r");
    if (!fp) return;
    while (fgets(line, sizeof(line), fp)) {
        char d[512], mp[512], fs[64];
        size_t mlen;
        if (sscanf(line, "%511s %511s %63s", d, mp, fs) != 3) continue;
        mlen = strlen(mp);
        if (mlen == 0) continue;
        if (strncmp(real, mp, mlen) != 0) continue;
        if (mlen > 1 && real[mlen] != '\0' && real[mlen] != '/') continue;
        if (mlen < best_len) continue;
        best_len = mlen;
        fox_strlcpy(best_dev, d, sizeof(best_dev));
        fox_strlcpy(best_fs, fs, sizeof(best_fs));
    }
    fclose(fp);
    fox_strlcpy(dev, best_dev, dev_cap);
    fox_strlcpy(fstype, best_fs, fs_cap);
}

static int linux_base_block_device(const char *devpath, char *out, size_t cap)
{
    char name[FOX_NAME_MAX];
    char probe[256];
    const char *base;
    size_t len;

    if (!devpath || strncmp(devpath, "/dev/", 5) != 0) return 0;
    base = devpath + 5;
    fox_strlcpy(name, base, sizeof(name));

    {
        char *slash = strrchr(name, '/');
        if (slash) memmove(name, slash + 1, strlen(slash + 1) + 1);
    }

    for (;;) {
        snprintf(probe, sizeof(probe), "/sys/block/%s", name);
        if (fox_path_exists(probe)) { fox_strlcpy(out, name, cap); return 1; }

        len = strlen(name);
        if (len == 0) return 0;
        if (name[len - 1] >= '0' && name[len - 1] <= '9') {
            while (len > 0 && name[len - 1] >= '0' && name[len - 1] <= '9') len--;
            if (len > 1 && name[len - 1] == 'p' &&
                name[len - 2] >= '0' && name[len - 2] <= '9') len--;
            name[len] = '\0';
            continue;
        }
        return 0;
    }
}

#endif

void fox_probe_storage_static(const char *path, fox_storage_info *out)
{
    memset(out, 0, sizeof(*out));
    out->rotational = -1;
    fox_strlcpy(out->path, path ? path : ".", sizeof(out->path));
    fox_strlcpy(out->device, "unknown", sizeof(out->device));
    fox_strlcpy(out->fstype, "unknown", sizeof(out->fstype));

    (void)fox_fs_space(out->path, &out->capacity_bytes, &out->free_bytes);

#if defined(FOX_OS_LINUX)
    {
        char devpath[FOX_NAME_MAX * 2];
        char fstype[64];
        char base[FOX_NAME_MAX];
        char rot[64];
        char probe[256];

        linux_find_mount(out->path, devpath, sizeof(devpath), fstype, sizeof(fstype));
        if (fstype[0]) fox_strlcpy(out->fstype, fstype, sizeof(out->fstype));

        if (linux_base_block_device(devpath, base, sizeof(base))) {
            fox_strlcpy(out->device, base, sizeof(out->device));
            snprintf(probe, sizeof(probe), "/sys/block/%s/queue/rotational", base);
            if (fox_read_small_file(probe, rot, sizeof(rot)) > 0)
                out->rotational = atoi(rot);
        } else if (devpath[0]) {
            fox_strlcpy(out->device, devpath, sizeof(out->device));
        }

        if (strcmp(out->fstype, "tmpfs") == 0 || strcmp(out->fstype, "ramfs") == 0)
            out->rotational = 0;
    }
#endif
}

#endif
