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

// Runtime SIMD dispatch: selects AVX3 (AVX-512), AVX2, or SSE42 at startup.
// SSE42 is the baseline and is assumed always available without a CPUID check.
// AVX2 and AVX3 are probed via CPUID.  Function pointers are resolved once at
// static-init time; each public call is a single indirect branch.
#include "jvector_simd.h"
#include "jvector_simd_kernels.h" // AVX3::, AVX2::, SSE42:: kernel declarations
#include "jvector_cpu_features.h"  // populate_cpu_features(), CpuFeature enum

#include <array>
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp

// Everything in this anonymous namespace is translation-unit private; none of
// these symbols are exported from the shared library.
namespace {

// Enumerates the ISA tiers in ascending capability order so that numeric
// comparisons (e.g. max_isa != MaxIsa::AVX2) correctly gate higher tiers.
// Unset (-1) means "no override; use best available CPU capability".
enum class MaxIsa { Unset = -1, SSE42 = 0, AVX2 = 1, AVX3 = 2 };

// Reads the JVECTOR_MAX_ISA environment variable and maps it to a MaxIsa
// value.  This lets callers cap the ISA at runtime without recompiling —
// useful for benchmarking or working around CPU errata.
// Accepted values (case-sensitive): "avx3", "avx2", "sse42".
static MaxIsa read_max_isa() noexcept
{
    const char *val = std::getenv("JVECTOR_MAX_ISA");
    if (!val) return MaxIsa::Unset;
    if (std::strcmp(val, "avx3")  == 0) return MaxIsa::AVX3;
    if (std::strcmp(val, "avx2")  == 0) return MaxIsa::AVX2;
    if (std::strcmp(val, "sse42") == 0) return MaxIsa::SSE42;
    return MaxIsa::Unset; // unrecognised value: ignore and use CPU detection
}

// KernelVTable holds one function pointer per public kernel.  Storing them
// together in a struct means a single pointer comparison during dispatch_kernels()
// selects all kernels for the chosen ISA in one shot.
// Auto-generated from jvector_simd_kernel_list.h
struct KernelVTable {
#define KERNEL_ENTRY(ret_type, name, params, names) \
    ret_type (*name) params;
    JVECTOR_SIMD_KERNEL_LIST
#undef KERNEL_ENTRY
};

// One pre-filled vtable per ISA.  These are constant data; no heap allocation.
// Auto-generated from jvector_simd_kernel_list.h

#define KERNEL_ENTRY(ret_type, name, params, names) AVX3::name,
static const KernelVTable AVX3_vtable = {
    JVECTOR_SIMD_KERNEL_LIST
};
#undef KERNEL_ENTRY

#define KERNEL_ENTRY(ret_type, name, params, names) AVX2::name,
static const KernelVTable AVX2_vtable = {
    JVECTOR_SIMD_KERNEL_LIST
};
#undef KERNEL_ENTRY

#define KERNEL_ENTRY(ret_type, name, params, names) SSE42::name,
static const KernelVTable SSE42_vtable = {
    JVECTOR_SIMD_KERNEL_LIST
};
#undef KERNEL_ENTRY

// Selects and returns the best vtable for the current CPU and environment.
// Called exactly once during static initialisation (before main()).
static KernelVTable dispatch_kernels() noexcept
{
    // Check whether the caller has capped the ISA via the environment variable.
    const MaxIsa max_isa = read_max_isa();

    // Populate a boolean feature array by issuing CPUID and reading XCR0.
    std::array<bool, static_cast<uint32_t>(CpuFeature::COUNT)> features;
    populate_cpu_features(features);

    auto has = [&](CpuFeature f) noexcept {
        return features[static_cast<uint32_t>(f)];
    };

    // AVX3 tier requires the full Skylake-AVX512 (SKX) baseline:
    // AVX512F (foundation) + BW (byte/word) + CD (conflict detect)
    // + DQ (dword/qword) + VL (vector length extensions).
    if (max_isa != MaxIsa::SSE42 && max_isa != MaxIsa::AVX2
        && has(CpuFeature::AVX512F) && has(CpuFeature::AVX512BW)
        && has(CpuFeature::AVX512CD) && has(CpuFeature::AVX512DQ)
        && has(CpuFeature::AVX512VL)) {
        return AVX3_vtable;
    }
    // AVX2 tier: 256-bit integer/FP SIMD, available on Haswell and later.
    if (max_isa != MaxIsa::SSE42 && has(CpuFeature::AVX2)) {
        return AVX2_vtable;
    }
    // SSE42 is the baseline — assumed always present, no CPUID check needed.
    return SSE42_vtable;
}

// 'kernels' is initialised once at static-init time to the vtable chosen by
// dispatch_kernels().  After that every public API call goes through one
// indirect branch to the right ISA implementation — no runtime comparisons.
static const KernelVTable kernels = dispatch_kernels();

} // namespace

// ---- Public API ------------------------------------------------------------
// Each function is a thin wrapper that forwards to the pre-resolved function
// pointer in `kernels`.
// Auto-generated from jvector_simd_kernel_list.h
//
// NOTE: When adding a new kernel, just add it to jvector_simd_kernel_list.h
// Everything else (vtable, namespace decls, public API, wrappers) is auto-generated!

#define KERNEL_ENTRY(ret_type, name, params, names) \
    ret_type name params { \
        return kernels.name names; \
    }

JVECTOR_SIMD_KERNEL_LIST

#undef KERNEL_ENTRY
