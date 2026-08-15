#include "platform.h"
#include "fox_internal.h"

#include <stdio.h>
#include <string.h>

#if defined(FOX_ARCH_X86)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

#if defined(FOX_ARCH_ARM64) && defined(FOX_OS_LINUX)
#  include <sys/auxv.h>
#  ifndef HWCAP_ASIMD
#    define HWCAP_ASIMD (1 << 1)
#  endif
#  ifndef HWCAP_ASIMDDP
#    define HWCAP_ASIMDDP (1 << 20)
#  endif
#  ifndef HWCAP_SVE
#    define HWCAP_SVE (1 << 22)
#  endif
#  ifndef HWCAP2_I8MM
#    define HWCAP2_I8MM (1 << 13)
#  endif
#endif

#if defined(FOX_ARCH_ARM64) && defined(FOX_OS_MACOS)
#  include <sys/sysctl.h>
#endif

#if defined(FOX_ARCH_X86)

static int cpuid_count(uint32_t leaf, uint32_t sub, uint32_t r[4])
{
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)sub);
    r[0] = (uint32_t)regs[0]; r[1] = (uint32_t)regs[1];
    r[2] = (uint32_t)regs[2]; r[3] = (uint32_t)regs[3];
    return 1;
#else
    return __get_cpuid_count(leaf, sub, &r[0], &r[1], &r[2], &r[3]) ? 1 : 0;
#endif
}

static uint32_t cpuid_max_leaf(uint32_t base)
{
    uint32_t r[4] = {0, 0, 0, 0};
    if (!cpuid_count(base, 0, r)) return 0;
    return r[0];
}

static uint64_t read_xcr0(void)
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | (uint64_t)eax;
#endif
}

struct uarch_entry {
    uint32_t    family;
    uint32_t    model;
    const char *name;
};

static const struct uarch_entry intel_uarch[] = {
    { 6, 0x37, "silvermont"    }, { 6, 0x4C, "airmont"       },
    { 6, 0x4D, "silvermont-d"  }, { 6, 0x5C, "goldmont"      },
    { 6, 0x5F, "goldmont-d"    }, { 6, 0x7A, "goldmont-plus" },
    { 6, 0x86, "tremont"       }, { 6, 0x96, "tremont"       },
    { 6, 0x9C, "tremont"       },
    { 6, 0x3D, "broadwell"     }, { 6, 0x47, "broadwell"     },
    { 6, 0x4E, "skylake"       }, { 6, 0x5E, "skylake"       },
    { 6, 0x55, "skylake-sp"    },
    { 6, 0x8E, "kabylake"      }, { 6, 0x9E, "coffeelake"    },
    { 6, 0xA5, "cometlake"     }, { 6, 0xA6, "cometlake"     },
    { 6, 0x7D, "icelake"       }, { 6, 0x7E, "icelake"       },
    { 6, 0x6A, "icelake-sp"    }, { 6, 0x6C, "icelake-sp"    },
    { 6, 0x8C, "tigerlake"     }, { 6, 0x8D, "tigerlake"     },
    { 6, 0x97, "alderlake"     }, { 6, 0x9A, "alderlake"     },
    { 6, 0xB7, "raptorlake"    }, { 6, 0xBA, "raptorlake"    },
    { 6, 0xBF, "raptorlake"    }, { 6, 0xCF, "emeraldrapids" }
};

static const struct uarch_entry amd_uarch[] = {
    { 0x15, 0, "bulldozer" }, { 0x16, 0, "jaguar" },
    { 0x17, 0, "zen"       }, { 0x19, 0, "zen3"   },
    { 0x1A, 0, "zen5"      }
};

static void classify_uarch(const char *vendor, uint32_t family, uint32_t model,
                           char *out, size_t cap)
{
    size_t i;

    if (strcmp(vendor, "GenuineIntel") == 0) {
        for (i = 0; i < FOX_ARRAY_LEN(intel_uarch); i++) {
            if (intel_uarch[i].family == family && intel_uarch[i].model == model) {
                fox_strlcpy(out, intel_uarch[i].name, cap);
                return;
            }
        }
    } else if (strcmp(vendor, "AuthenticAMD") == 0) {
        for (i = 0; i < FOX_ARRAY_LEN(amd_uarch); i++) {
            if (amd_uarch[i].family == family) {
                fox_strlcpy(out, amd_uarch[i].name, cap);
                return;
            }
        }
    }
    snprintf(out, cap, "x86-f%u-m%02x", (unsigned)family, (unsigned)model);
}

static void probe_x86(fox_cpu_info *out)
{
    uint32_t r[4] = {0, 0, 0, 0};
    uint32_t max_basic, max_ext;
    uint32_t family = 0, model = 0;
    int os_avx = 0, os_avx512 = 0;

    out->cacheline_bytes = 64;

    max_basic = cpuid_max_leaf(0);
    if (cpuid_count(0, 0, r)) {
        memcpy(out->vendor + 0, &r[1], 4);
        memcpy(out->vendor + 4, &r[3], 4);
        memcpy(out->vendor + 8, &r[2], 4);
        out->vendor[12] = '\0';
    }

    max_ext = cpuid_max_leaf(0x80000000u);
    if (max_ext >= 0x80000004u) {
        uint32_t i;
        char brand[49];
        for (i = 0; i < 3; i++) {
            if (!cpuid_count(0x80000002u + i, 0, r)) break;
            memcpy(brand + i * 16 +  0, &r[0], 4);
            memcpy(brand + i * 16 +  4, &r[1], 4);
            memcpy(brand + i * 16 +  8, &r[2], 4);
            memcpy(brand + i * 16 + 12, &r[3], 4);
        }
        brand[48] = '\0';
        {
            char *p = brand;
            while (*p == ' ') p++;
            fox_strlcpy(out->brand, p, sizeof(out->brand));
        }
    }

    if (max_basic < 1) return;

    if (cpuid_count(1, 0, r)) {
        uint32_t base_family = (r[0] >> 8)  & 0xF;
        uint32_t base_model  = (r[0] >> 4)  & 0xF;
        uint32_t ext_family  = (r[0] >> 20) & 0xFF;
        uint32_t ext_model   = (r[0] >> 16) & 0xF;

        family = base_family;
        model  = base_model;
        if (base_family == 0xF) family += ext_family;
        if (base_family == 0x6 || base_family == 0xF) model += (ext_model << 4);

        if (r[3] & (1u << 26)) out->features |= FOX_CPU_SSE2;
        if (r[2] & (1u <<  9)) out->features |= FOX_CPU_SSSE3;
        if (r[2] & (1u << 19)) out->features |= FOX_CPU_SSE41;
        if (r[2] & (1u << 20)) out->features |= FOX_CPU_SSE42;
        if (r[2] & (1u << 23)) out->features |= FOX_CPU_POPCNT;

        if (r[2] & (1u << 27)) {
            uint64_t xcr0 = read_xcr0();
            os_avx    = (xcr0 & 0x6ull) == 0x6ull;
            os_avx512 = os_avx && ((xcr0 & 0xE0ull) == 0xE0ull);
        }
        if (os_avx) {
            if (r[2] & (1u << 28)) out->features |= FOX_CPU_AVX;
            if (r[2] & (1u << 29)) out->features |= FOX_CPU_F16C;
            if (r[2] & (1u << 12)) out->features |= FOX_CPU_FMA;
        }
    }

    if (max_basic >= 7) {
        if (cpuid_count(7, 0, r)) {
            if (os_avx && (r[1] & (1u << 5)))  out->features |= FOX_CPU_AVX2;
            if (os_avx512) {
                if (r[1] & (1u << 16)) out->features |= FOX_CPU_AVX512F;
                if (r[1] & (1u << 30)) out->features |= FOX_CPU_AVX512BW;
                if (r[2] & (1u << 11)) out->features |= FOX_CPU_AVX512VNNI;
            }
        }
        if (cpuid_count(7, 1, r)) {
            if (os_avx && (r[0] & (1u << 4))) out->features |= FOX_CPU_AVXVNNI;
        }
    }

    classify_uarch(out->vendor, family, model, out->uarch, sizeof(out->uarch));
}

#endif

#if defined(FOX_ARCH_ARM64)

static void probe_arm64(fox_cpu_info *out)
{
    out->cacheline_bytes = 64;
    out->features |= FOX_CPU_NEON;
    fox_strlcpy(out->vendor, "aarch64", sizeof(out->vendor));
    fox_strlcpy(out->uarch, "aarch64", sizeof(out->uarch));

#if defined(FOX_OS_LINUX)
    {
        unsigned long hw  = getauxval(AT_HWCAP);
        unsigned long hw2 = getauxval(AT_HWCAP2);
        if (hw & HWCAP_ASIMDDP) out->features |= FOX_CPU_DOTPROD;
        if (hw & HWCAP_SVE)     out->features |= FOX_CPU_SVE;
        if (hw2 & HWCAP2_I8MM)  out->features |= FOX_CPU_I8MM;
    }
    {
        char buf[4096];
        if (fox_read_small_file("/proc/cpuinfo", buf, sizeof(buf)) > 0) {
            char *p = strstr(buf, "CPU implementer");
            if (p) fox_strlcpy(out->brand, "aarch64 CPU", sizeof(out->brand));
        }
    }
#elif defined(FOX_OS_MACOS)
    {
        size_t sz;
        int v = 0;
        char brand[64];
        sz = sizeof(v);
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) == 0 && v)
            out->features |= FOX_CPU_DOTPROD;
        v = 0; sz = sizeof(v);
        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, NULL, 0) == 0 && v)
            out->features |= FOX_CPU_I8MM;
        sz = sizeof(brand);
        if (sysctlbyname("machdep.cpu.brand_string", brand, &sz, NULL, 0) == 0)
            fox_strlcpy(out->brand, brand, sizeof(out->brand));
        fox_strlcpy(out->vendor, "Apple", sizeof(out->vendor));
    }
#endif
}

#endif

void fox_probe_cpu(fox_cpu_info *out)
{
    memset(out, 0, sizeof(*out));
    fox_strlcpy(out->vendor, "unknown", sizeof(out->vendor));
    fox_strlcpy(out->brand,  "unknown", sizeof(out->brand));
    fox_strlcpy(out->uarch,  "unknown", sizeof(out->uarch));
    out->cacheline_bytes = 64;

#if defined(FOX_ARCH_X86)
    probe_x86(out);
#elif defined(FOX_ARCH_ARM64)
    probe_arm64(out);
#endif

    fox_plat_cpu_topology(out);
}

struct feature_name { uint32_t bit; const char *name; };

static const struct feature_name feature_names[] = {
    { FOX_CPU_SSE2,       "sse2"        },
    { FOX_CPU_SSSE3,      "ssse3"       },
    { FOX_CPU_SSE41,      "sse4.1"      },
    { FOX_CPU_SSE42,      "sse4.2"      },
    { FOX_CPU_POPCNT,     "popcnt"      },
    { FOX_CPU_AVX,        "avx"         },
    { FOX_CPU_F16C,       "f16c"        },
    { FOX_CPU_FMA,        "fma"         },
    { FOX_CPU_AVX2,       "avx2"        },
    { FOX_CPU_AVX512F,    "avx512f"     },
    { FOX_CPU_AVX512BW,   "avx512bw"    },
    { FOX_CPU_AVX512VNNI, "avx512vnni"  },
    { FOX_CPU_AVXVNNI,    "avxvnni"     },
    { FOX_CPU_NEON,       "neon"        },
    { FOX_CPU_DOTPROD,    "dotprod"     },
    { FOX_CPU_I8MM,       "i8mm"        },
    { FOX_CPU_SVE,        "sve"         }
};

char *fox_cpu_features_str(uint32_t features, char *buf, size_t buflen)
{
    size_t used = 0;
    size_t i;

    if (!buf || buflen == 0) return buf;
    buf[0] = '\0';

    for (i = 0; i < FOX_ARRAY_LEN(feature_names); i++) {
        size_t need;
        if (!(features & feature_names[i].bit)) continue;
        need = strlen(feature_names[i].name) + (used ? 1u : 0u);
        if (used + need + 1 > buflen) break;
        if (used) buf[used++] = ' ';
        memcpy(buf + used, feature_names[i].name, strlen(feature_names[i].name));
        used += strlen(feature_names[i].name);
        buf[used] = '\0';
    }
    if (used == 0) fox_strlcpy(buf, "none", buflen);
    return buf;
}
