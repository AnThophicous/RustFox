#include "platform.h"

#if defined(FOX_OS_WINDOWS)

#include "fox_internal.h"

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t fox_now_ns(void)
{
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER now;

    if (!have_freq) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) return 0;
        have_freq = 1;
    }
    if (!QueryPerformanceCounter(&now)) return 0;

    return (uint64_t)(now.QuadPart / freq.QuadPart) * 1000000000ull +
           (uint64_t)((now.QuadPart % freq.QuadPart) * 1000000000ll / freq.QuadPart);
}

void fox_sleep_ms(uint32_t ms) { Sleep((DWORD)ms); }

size_t fox_page_size(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize ? (size_t)si.dwPageSize : 4096u;
}

void *fox_aligned_alloc(size_t alignment, size_t size)
{
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if ((alignment & (alignment - 1)) != 0) return NULL;
    if (size == 0) size = 1;
    return _aligned_malloc(size, alignment);
}

void fox_aligned_free(void *p) { if (p) _aligned_free(p); }

struct fox_file {
    HANDLE h;
    int    direct;
};

fox_file *fox_file_open_read(const char *path, unsigned flags, int *direct_granted)
{
    fox_file *f;
    HANDLE h = INVALID_HANDLE_VALUE;
    DWORD attr = FILE_ATTRIBUTE_NORMAL;
    int granted = 0;

    if (flags & FOX_OPEN_RANDOM) attr |= FILE_FLAG_RANDOM_ACCESS;
    if (flags & FOX_OPEN_SEQ)    attr |= FILE_FLAG_SEQUENTIAL_SCAN;

    if (flags & FOX_OPEN_DIRECT) {
        h = CreateFileA(path, GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                        attr | FILE_FLAG_NO_BUFFERING, NULL);
        if (h != INVALID_HANDLE_VALUE) granted = 1;
    }
    if (h == INVALID_HANDLE_VALUE) {
        h = CreateFileA(path, GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                        attr, NULL);
    }
    if (h == INVALID_HANDLE_VALUE) {
        fox_fail(FOX_ERR_IO, "CreateFile %s: error %lu",
                 path, (unsigned long)GetLastError());
        if (direct_granted) *direct_granted = 0;
        return NULL;
    }

    f = (fox_file *)calloc(1, sizeof(*f));
    if (!f) {
        CloseHandle(h);
        fox_fail(FOX_ERR_NOMEM, "fox_file alloc");
        if (direct_granted) *direct_granted = 0;
        return NULL;
    }
    f->h = h;
    f->direct = granted;
    if (direct_granted) *direct_granted = granted;
    return f;
}

void fox_file_close(fox_file *f)
{
    if (!f) return;
    if (f->h != INVALID_HANDLE_VALUE) CloseHandle(f->h);
    free(f);
}

int64_t fox_file_size(fox_file *f)
{
    LARGE_INTEGER sz;
    if (!f || !GetFileSizeEx(f->h, &sz)) return -1;
    return (int64_t)sz.QuadPart;
}

int64_t fox_file_pread(fox_file *f, void *buf, size_t n, uint64_t offset)
{
    size_t done = 0;

    if (!f || !buf) return -1;
    while (done < n) {
        OVERLAPPED ov;
        DWORD got = 0;
        DWORD want = (DWORD)((n - done) > 0x7FFFFFFFu ? 0x7FFFFFFFu : (n - done));
        uint64_t at = offset + done;

        memset(&ov, 0, sizeof(ov));
        ov.Offset     = (DWORD)(at & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD)(at >> 32);

        if (!ReadFile(f->h, (char *)buf + done, want, &got, &ov)) {
            DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            return done > 0 ? (int64_t)done : -1;
        }
        if (got == 0) break;
        done += (size_t)got;
    }
    return (int64_t)done;
}

int fox_file_drop_cache(fox_file *f)
{
    return f ? f->direct : 0;
}

void fox_sync_path(const char *path)
{
    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    FlushFileBuffers(h);
    CloseHandle(h);
}

int fox_path_exists(const char *path)
{
    return path && GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
}

char *fox_temp_dir(char *buf, size_t cap)
{
    char tmp[MAX_PATH + 1];
    DWORD n = GetTempPathA((DWORD)sizeof(tmp), tmp);
    if (n == 0 || n > sizeof(tmp) - 1) { fox_strlcpy(buf, ".", cap); return buf; }
    while (n > 1 && (tmp[n - 1] == '\\' || tmp[n - 1] == '/')) tmp[--n] = '\0';
    fox_strlcpy(buf, tmp, cap);
    return buf;
}

fox_status fox_mkdir_p(const char *path)
{
    char tmp[FOX_PATH_MAX];
    size_t len, i;

    if (!path || !*path) return FOX_ERR_ARG;
    fox_strlcpy(tmp, path, sizeof(tmp));
    len = strlen(tmp);
    while (len > 1 && (tmp[len - 1] == '\\' || tmp[len - 1] == '/')) tmp[--len] = '\0';

    for (i = 1; i < len; i++) {
        if (tmp[i] != '\\' && tmp[i] != '/') continue;
        if (i == 2 && tmp[1] == ':') continue;
        {
            char save = tmp[i];
            tmp[i] = '\0';
            if (!CreateDirectoryA(tmp, NULL) &&
                GetLastError() != ERROR_ALREADY_EXISTS)
                return fox_fail(FOX_ERR_IO, "mkdir %s: error %lu", tmp,
                                (unsigned long)GetLastError());
            tmp[i] = save;
        }
    }
    if (!CreateDirectoryA(tmp, NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        return fox_fail(FOX_ERR_IO, "mkdir %s: error %lu", tmp,
                        (unsigned long)GetLastError());
    return FOX_OK;
}

fox_status fox_fs_space(const char *path, uint64_t *capacity, uint64_t *avail)
{
    ULARGE_INTEGER free_to_caller, total, total_free;
    if (!path) return FOX_ERR_ARG;
    if (!GetDiskFreeSpaceExA(path, &free_to_caller, &total, &total_free))
        return fox_fail(FOX_ERR_IO, "GetDiskFreeSpaceEx %s: error %lu", path,
                        (unsigned long)GetLastError());
    if (capacity) *capacity = (uint64_t)total.QuadPart;
    if (avail)    *avail    = (uint64_t)free_to_caller.QuadPart;
    return FOX_OK;
}

int fox_cpu_count_online(void)
{
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) &&
        proc_mask != 0) {
        int n = 0;
        DWORD_PTR m = proc_mask;
        while (m) { n += (int)(m & 1u); m >>= 1; }
        if (n > 0) return n;
    }
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
    }
}

void fox_plat_cpu_topology(fox_cpu_info *out)
{
    DWORD len = 0;
    out->logical_cores = fox_cpu_count_online();
    out->physical_cores = out->logical_cores;
    out->numa_nodes = 1;

    GetLogicalProcessorInformationEx(RelationAll, NULL, &len);
    if (len > 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        char *raw = (char *)malloc(len);
        if (raw) {
            if (GetLogicalProcessorInformationEx(RelationAll,
                    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)raw, &len)) {
                DWORD off = 0;
                int cores = 0, nodes = 0;
                while (off < len) {
                    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *p =
                        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(raw + off);
                    if (p->Size == 0) break;
                    switch (p->Relationship) {
                    case RelationProcessorCore: cores++; break;
                    case RelationNumaNode:      nodes++; break;
                    case RelationCache: {
                        uint64_t sz = (uint64_t)p->Cache.CacheSize;
                        if (p->Cache.Level == 1 && p->Cache.Type == CacheData)
                            { if (sz > out->l1d_bytes) out->l1d_bytes = sz; }
                        else if (p->Cache.Level == 2)
                            { if (sz > out->l2_bytes) out->l2_bytes = sz; }
                        else if (p->Cache.Level == 3)
                            { if (sz > out->l3_bytes) out->l3_bytes = sz; }
                        break;
                    }
                    default: break;
                    }
                    off += p->Size;
                }
                if (cores > 0) out->physical_cores = cores;
                if (nodes > 0) out->numa_nodes = nodes;
            }
            free(raw);
        }
    }
}

void fox_probe_mem(fox_mem_info *out)
{
    MEMORYSTATUSEX ms;

    memset(out, 0, sizeof(*out));
    out->page_size       = (uint64_t)fox_page_size();
    out->cgroup_limit    = UINT64_MAX;
    out->cgroup_high     = UINT64_MAX;
    out->swappiness      = -1;
    out->overcommit_mode = -1;

    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out->host_total_bytes = (uint64_t)ms.ullTotalPhys;
        out->free_bytes       = (uint64_t)ms.ullAvailPhys;
        out->available_bytes  = (uint64_t)ms.ullAvailPhys;
        if (ms.ullTotalPageFile > ms.ullTotalPhys)
            out->swap_total_bytes = (uint64_t)(ms.ullTotalPageFile - ms.ullTotalPhys);
        if (ms.ullAvailPageFile > ms.ullAvailPhys)
            out->swap_free_bytes = (uint64_t)(ms.ullAvailPageFile - ms.ullAvailPhys);
    }
    out->total_bytes = out->host_total_bytes;
}

void fox_probe_gpu(fox_gpu_info *out)
{
    memset(out, 0, sizeof(*out));
    fox_strlcpy(out->api, "none", sizeof(out->api));
    fox_strlcpy(out->name, "unprobed (native windows build)", sizeof(out->name));
}

void fox_probe_os(char *os, size_t os_cap, char *kernel, size_t kernel_cap)
{
    typedef LONG (WINAPI *rtl_get_version_fn)(void *);
    HMODULE ntdll;

    fox_strlcpy(os, "Windows", os_cap);
    fox_strlcpy(kernel, "unknown", kernel_cap);

    ntdll = GetModuleHandleA("ntdll.dll");
    if (ntdll) {
        rtl_get_version_fn fn =
            (rtl_get_version_fn)(void *)GetProcAddress(ntdll, "RtlGetVersion");
        if (fn) {
            struct {
                ULONG dwOSVersionInfoSize;
                ULONG dwMajorVersion;
                ULONG dwMinorVersion;
                ULONG dwBuildNumber;
                ULONG dwPlatformId;
                WCHAR szCSDVersion[128];
            } vi;
            memset(&vi, 0, sizeof(vi));
            vi.dwOSVersionInfoSize = sizeof(vi);
            if (fn(&vi) == 0) {
                snprintf(kernel, kernel_cap, "%lu.%lu.%lu",
                         (unsigned long)vi.dwMajorVersion,
                         (unsigned long)vi.dwMinorVersion,
                         (unsigned long)vi.dwBuildNumber);
            }
        }
    }
}

void fox_probe_storage_static(const char *path, fox_storage_info *out)
{
    char root[MAX_PATH + 1];
    char fsname[64];

    memset(out, 0, sizeof(*out));
    out->rotational = -1;
    fox_strlcpy(out->path, path ? path : ".", sizeof(out->path));
    fox_strlcpy(out->device, "unknown", sizeof(out->device));
    fox_strlcpy(out->fstype, "unknown", sizeof(out->fstype));

    (void)fox_fs_space(out->path, &out->capacity_bytes, &out->free_bytes);

    if (GetVolumePathNameA(out->path, root, (DWORD)sizeof(root))) {
        fox_strlcpy(out->device, root, sizeof(out->device));
        if (GetVolumeInformationA(root, NULL, 0, NULL, NULL, NULL,
                                  fsname, (DWORD)sizeof(fsname)))
            fox_strlcpy(out->fstype, fsname, sizeof(out->fstype));
    }
}

#endif
