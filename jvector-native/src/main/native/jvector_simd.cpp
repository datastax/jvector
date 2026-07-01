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

// Runtime SIMD dispatch: selects the best available ISA tier at startup.
// Tiers in descending capability order: AVX3_SPR, AVX3_DL, AVX3, AVX2, SSE42.
// SSE42 is the baseline and is assumed always available without a CPUID check.
// Function pointers are resolved once at static-init time; each public call is
// a single indirect branch.
#include "jvector_simd.h"
#include "jvector_simd_kernels.h" // AVX3_SPR::, AVX3_DL::, AVX3::, AVX2::, SSE42:: kernel declarations
#include "jvector_cpu_features.h"  // populate_cpu_features(), CpuFeature enum

#include <array>
#include <climits> // INT_MAX
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp

// Everything in this anonymous namespace is translation-unit private; none of
// these symbols are exported from the shared library.
namespace {

// Enumerates the ISA tiers in ascending capability order so that numeric
// comparisons (e.g. max_isa > MaxIsa::AVX2) correctly gate higher tiers.
// Unset (INT_MAX) means "no override; use best available CPU capability"
// and must be greater than every named tier so that all guards pass.
enum class MaxIsa { SSE42 = 0, AVX2 = 1, AVX3 = 2, AVX3_DL = 3, AVX3_SPR = 4,
                    Unset = INT_MAX };
static_assert(
    (int)MaxIsa::SSE42    < (int)MaxIsa::AVX2
    && (int)MaxIsa::AVX2  < (int)MaxIsa::AVX3
    && (int)MaxIsa::AVX3  < (int)MaxIsa::AVX3_DL
    && (int)MaxIsa::AVX3_DL < (int)MaxIsa::AVX3_SPR
    && (int)MaxIsa::AVX3_SPR < (int)MaxIsa::Unset,
    "MaxIsa values must be in strict ascending capability order with Unset at the top");

// Reads the JVECTOR_MAX_ISA environment variable and maps it to a MaxIsa
// value.  This lets callers cap the ISA at runtime without recompiling —
// useful for benchmarking or working around CPU errata.
// Accepted values (case-sensitive): "avx3", "avx2", "sse42".
static MaxIsa read_max_isa() noexcept
{
    const char *val = std::getenv("JVECTOR_MAX_ISA");
    if (!val) return MaxIsa::Unset;
    if (std::strcmp(val, "avx3_spr") == 0) return MaxIsa::AVX3_SPR;
    if (std::strcmp(val, "avx3_dl")  == 0) return MaxIsa::AVX3_DL;
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

// AVX3_DL (Ice Lake) inherits all slots from AVX3 unchanged for now.
// To override a slot: t.kernel_name = AVX3_DL::kernel_name;
// The implementation must exist in jvector_avx3_dl_kernels.cpp.
static const KernelVTable AVX3_DL_vtable = []() {
    KernelVTable t = AVX3_vtable;
    return t;
}();

// AVX3_SPR (Sapphire Rapids) inherits all slots from AVX3_DL unchanged for now.
// To override a slot: t.kernel_name = AVX3_SPR::kernel_name;
// The implementation must exist in jvector_avx3_spr_kernels.cpp.
static const KernelVTable AVX3_SPR_vtable = []() {
    KernelVTable t = AVX3_DL_vtable;   // inherits DL overrides automatically
    return t;
}();

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

// Bundles the chosen vtable and the tier that was selected together so that
// both can be initialised atomically from a single dispatch call.
struct DispatchResult {
    KernelVTable vtable;
    MaxIsa       tier;
};

// Selects and returns the best vtable for the current CPU and environment.
// Called exactly once during static initialisation (before main()).
static DispatchResult dispatch_kernels() noexcept
{
    // Check whether the caller has capped the ISA via the environment variable.
    const MaxIsa max_isa = read_max_isa();

    // Populate a boolean feature array by issuing CPUID and reading XCR0.
    std::array<bool, static_cast<uint32_t>(CpuFeature::COUNT)> features;
    populate_cpu_features(features);

    auto has = [&](CpuFeature f) noexcept {
        return features[static_cast<uint32_t>(f)];
    };

    // Select the highest tier the CPU supports and the cap allows.
    // max_isa > MaxIsa::X means "user has not capped at X or below".
    // Adding a new tier above AVX3_SPR only requires one new if at the top.
    if (max_isa > MaxIsa::AVX3_DL && has(CpuFeature::AVX3_SPR))
        return { AVX3_SPR_vtable, MaxIsa::AVX3_SPR };
    if (max_isa > MaxIsa::AVX3    && has(CpuFeature::AVX3_DL))
        return { AVX3_DL_vtable, MaxIsa::AVX3_DL };
    if (max_isa > MaxIsa::AVX2    && has(CpuFeature::AVX3))
        return { AVX3_vtable, MaxIsa::AVX3 };
    if (max_isa > MaxIsa::SSE42   && has(CpuFeature::AVX2))
        return { AVX2_vtable, MaxIsa::AVX2 };
    // SSE42 is the baseline — assumed always present, no CPUID check needed.
    return { SSE42_vtable, MaxIsa::SSE42 };
}

// Both are initialised once at static-init time from a single dispatch call.
// After that every public API call goes through one indirect branch to the
// right ISA implementation — no runtime comparisons.
static const DispatchResult dispatch = dispatch_kernels();
static const KernelVTable  &kernels  = dispatch.vtable;
static const MaxIsa          active_isa = dispatch.tier;

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
