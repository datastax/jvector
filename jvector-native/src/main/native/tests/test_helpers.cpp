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

#include "test_helpers.h"

// ---------------------------------------------------------------------------
// Global test environment — prints the active ISA once for the whole binary.
// Registered via AddGlobalTestEnvironment at static-init time so it fires
// before any test suite runs, regardless of which .cpp files are linked.
// ---------------------------------------------------------------------------

class JVectorIsaEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        std::printf("[  ISA     ] Active dispatch tier: %s\n",
                    jvector_simd_get_active_isa());
    }
};

static ::testing::Environment* const kIsaEnv =
    ::testing::AddGlobalTestEnvironment(new JVectorIsaEnvironment);

// ---------------------------------------------------------------------------
// make_vec: deterministic test vectors.
// Values in roughly (-2, 2] with a mix of signs so no element is zero.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Canonical test sizes — shared by all parametrised suites.
// Covers tail-only, single-register, and 4x-unrolled main-loop paths for
// SSE42 (4 lanes), AVX2 (8 lanes), and AVX512 (16 lanes).
// ---------------------------------------------------------------------------

const std::vector<KernelTestParam> kKernelTestParams = {
    // ---- tail-only (< 4 lanes for any ISA) --------------------------------
    {  1, "tail_1_all_isa"},
    {  3, "tail_3_all_isa"},
    // ---- SSE42 boundary (4-lane register) ----------------------------------
    {  4, "sse42_exact_4"},
    {  5, "sse42_1full_tail_1"},
    {  7, "sse42_1full_tail_3"},
    // ---- AVX2 boundary (8-lane register) -----------------------------------
    {  8, "avx2_exact_8"},
    {  9, "avx2_1full_tail_1"},
    { 15, "avx2_1full_tail_7"},
    // ---- SSE42 4x-unrolled main loop (16 elements = 4 × 4 lanes) ----------
    { 16, "sse42_4x_main_exact"},
    { 17, "sse42_4x_main_tail_1"},
    { 19, "sse42_4x_main_tail_3"},
    // ---- AVX2 4x-unrolled main loop (32 elements = 4 × 8 lanes) -----------
    { 32, "avx2_4x_main_exact"},
    { 33, "avx2_4x_main_tail_1"},
    { 37, "avx2_4x_main_tail_5"},
    // ---- AVX512 boundary (16-lane register) --------------------------------
    { 64, "avx512_4x_main_exact"},
    { 71, "avx512_4x_main_tail_7"},
    // ---- Odd large size exercising all loop stages -------------------------
    {100, "large_mixed_tail"},
    {128, "large_power_of_2"},
    {255, "large_odd_tail_15"},
};

std::vector<float> make_vec(size_t n, float seed)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = seed * (1.0f + static_cast<float>(i % 7) * 0.13f);
        if (i % 3 == 0) v[i] = -v[i]; // mix of signs
        v[i] += 0.5f;                  // ensure non-zero even after sign flip
    }
    return v;
}

