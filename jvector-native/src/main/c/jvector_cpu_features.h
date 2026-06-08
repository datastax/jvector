/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

#include <array>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

// Features needed by the ISA dispatch table.  Extend as new targets are added.
//
// ICX  = Intel Ice Lake-SP (Xeon Scalable 3rd Gen)
// SPR  = Intel Sapphire Rapids (Xeon Scalable 4th Gen)
enum class CpuFeature : uint32_t {
    // ---- Base AVX2 / AVX-512 foundation (all SKUs) ----------------------
    AVX2 = 0,
    AVX512F = 1,
    AVX512BW = 2,
    AVX512CD = 3,
    AVX512DQ = 4,
    AVX512VL = 5,
    // ---- ICX additions --------------------------------------------------
    AVX512_VNNI = 6, // INT8 dot-product (CPUID 7.0 ECX[11])
    AVX512_VBMI = 7, // byte permute/shuffle (CPUID 7.0 ECX[1])
    AVX512_VBMI2 = 8, // byte/word expand+compress (CPUID 7.0 ECX[6])
    AVX512_IFMA = 9, // 52-bit integer multiply-add (CPUID 7.0 EBX[21])
    AVX512_BITALG = 10, // bit-manipulation (CPUID 7.0 ECX[12])
    AVX512_VPOPCNTDQ = 11, // vector popcount dword/qword (CPUID 7.0 ECX[14])
    GFNI = 12, // Galois-field instructions (CPUID 7.0 ECX[8])
    VAES = 13, // 256/512-bit AES (CPUID 7.0 ECX[9])
    VPCLMULQDQ = 14, // wide carry-less multiply (CPUID 7.0 ECX[10])
    // ---- SPR additions --------------------------------------------------
    AVX512_FP16 = 15, // FP16 arithmetic (CPUID 7.0 EDX[23])
    AVX512_BF16 = 16, // BFloat16 arithmetic (CPUID 7.1 EAX[5])
    AVX_VNNI = 17, // VEX-encoded VNNI 256-bit (CPUID 7.1 EAX[4])
    COUNT
};

// Populate `features` by issuing CPUID and XGETBV.
// All entries are false on non-x86 architectures.
inline void
populate_cpu_features(std::array<bool, static_cast<uint32_t>(CpuFeature::COUNT)>
                              &features) noexcept
{
    features.fill(false);

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) \
        || defined(_M_X64)

    // Portable CPUID: GCC/Clang use <cpuid.h>; MSVC uses <intrin.h>.
    auto run_cpuid = [](uint32_t leaf,
                        uint32_t subleaf,
                        uint32_t &eax,
                        uint32_t &ebx,
                        uint32_t &ecx,
                        uint32_t &edx) noexcept {
#if defined(_MSC_VER)
        int info[4];
        __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
        eax = static_cast<uint32_t>(info[0]);
        ebx = static_cast<uint32_t>(info[1]);
        ecx = static_cast<uint32_t>(info[2]);
        edx = static_cast<uint32_t>(info[3]);
#else
        __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
#endif
    };

    // Read XCR0.  Must only be called when OSXSAVE (CPUID.1:ECX[27]) is set;
    // otherwise executing XGETBV raises #UD.
    auto read_xcr0 = []() noexcept -> uint64_t {
#if defined(_MSC_VER)
        return static_cast<uint64_t>(_xgetbv(0));
#else
        uint32_t lo, hi;
        __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0u));
        return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    };

    uint32_t eax, ebx, ecx, edx;

    // CPUID leaf 1 — check OSXSAVE: OS enabled XSAVE/XRSTOR (ECX bit 27).
    run_cpuid(1u, 0u, eax, ebx, ecx, edx);
    const bool osxsave = (ecx >> 27) & 1u;

    // XCR0 encodes which register state the OS saves on context switch.
    // Read it only if the CPU and OS have declared XSAVE support.
    uint64_t xcr0 = 0u;
    if (osxsave) { xcr0 = read_xcr0(); }

    // Bits 1‥2: XMM and YMM state — required for AVX / AVX2.
    const bool ymm_enabled = (xcr0 & 0x06u) == 0x06u;
    // Bits 5‥7: opmask, ZMM_Hi256, Hi16_ZMM — required for AVX-512.
    const bool zmm_enabled = ymm_enabled && ((xcr0 & 0xe0u) == 0xe0u);

    // CPUID leaf 7, subleaf 0 — extended feature flags.
    // EAX returns the maximum supported subleaf index for leaf 7.
    run_cpuid(7u, 0u, eax, ebx, ecx, edx);
    const uint32_t leaf7_max_subleaf = eax;

    // ---- Base AVX2 / AVX-512 foundation ---------------------------------
    features[static_cast<uint32_t>(CpuFeature::AVX2)]
            = ymm_enabled && ((ebx >> 5) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512F)]
            = zmm_enabled && ((ebx >> 16) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512BW)]
            = zmm_enabled && ((ebx >> 30) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512CD)]
            = zmm_enabled && ((ebx >> 28) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512DQ)]
            = zmm_enabled && ((ebx >> 17) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512VL)]
            = zmm_enabled && ((ebx >> 31) & 1u);

    // ---- ICX: leaf 7.0 EBX additions ------------------------------------
    features[static_cast<uint32_t>(CpuFeature::AVX512_IFMA)]
            = zmm_enabled && ((ebx >> 21) & 1u);

    // ---- ICX: leaf 7.0 ECX additions ------------------------------------
    features[static_cast<uint32_t>(CpuFeature::AVX512_VBMI)]
            = zmm_enabled && ((ecx >> 1) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512_VBMI2)]
            = zmm_enabled && ((ecx >> 6) & 1u);
    features[static_cast<uint32_t>(CpuFeature::GFNI)]
            = zmm_enabled && ((ecx >> 8) & 1u);
    features[static_cast<uint32_t>(CpuFeature::VAES)]
            = zmm_enabled && ((ecx >> 9) & 1u);
    features[static_cast<uint32_t>(CpuFeature::VPCLMULQDQ)]
            = zmm_enabled && ((ecx >> 10) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512_VNNI)]
            = zmm_enabled && ((ecx >> 11) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512_BITALG)]
            = zmm_enabled && ((ecx >> 12) & 1u);
    features[static_cast<uint32_t>(CpuFeature::AVX512_VPOPCNTDQ)]
            = zmm_enabled && ((ecx >> 14) & 1u);

    // ---- SPR: leaf 7.0 EDX additions ------------------------------------
    features[static_cast<uint32_t>(CpuFeature::AVX512_FP16)]
            = zmm_enabled && ((edx >> 23) & 1u);

    // ---- SPR: leaf 7.1 EAX additions (guarded by max subleaf) ----------
    if (leaf7_max_subleaf >= 1u) {
        run_cpuid(7u, 1u, eax, ebx, ecx, edx);
        features[static_cast<uint32_t>(CpuFeature::AVX_VNNI)]
                = ymm_enabled && ((eax >> 4) & 1u);
        features[static_cast<uint32_t>(CpuFeature::AVX512_BF16)]
                = zmm_enabled && ((eax >> 5) & 1u);
    }

#endif // x86 / x86_64
}

#endif // CPU_FEATURES_H
