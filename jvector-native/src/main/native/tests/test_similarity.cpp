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

// Tests for vector similarity kernels: cosine_f32, dot_product_f32, euclidean_f32.
//
// The library dispatches to the best ISA available on the host CPU at
// static-init time. Vector sizes are chosen to hit every code path in the
// kernel loops regardless of which ISA is selected:
//
//   SSE42  (4 lanes):
//     sizes  1, 3       — tail only (< 4)
//     size   4          — exact one vector, no tail
//     size   7          — one full + 3-element tail
//     size  16          — 4x unrolled main loop, no tail
//     size  19          — 4x main + 3-element tail
//
//   AVX2  (8 lanes):
//     sizes  1, 3       — capped fast path (≤4), tail only
//     size   4          — capped fast path (≤4), one vector no tail
//     size   7          — capped fast path (≤8), tail = 7 < 8
//     size   8          — capped fast path (≤8), exact no tail
//     size  15          — one full + 7-element tail
//     size  32          — 4x unrolled main loop, no tail
//     size  37          — 4x main + 5-element tail
//
//   AVX3/AVX512  (16 lanes):
//     sizes  1, 3       — capped (≤4), tail only
//     size   4          — capped (≤4), exact
//     size   8          — capped (≤8), exact
//     size  15          — one full (16 lanes) – 1 = tail
//     size  16          — exact one full register
//     size  64          — 4x unrolled, no tail
//     size  71          — 4x main + 7-element tail

#include "test_helpers.h"

// ---------------------------------------------------------------------------
// Reference scalar implementations
// ---------------------------------------------------------------------------

static float ref_dot(const std::vector<float>& a, const std::vector<float>& b)
{
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static float ref_euclidean(const std::vector<float>& a, const std::vector<float>& b)
{
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static float ref_cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    float ab = 0.0f, aa = 0.0f, bb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

// ---------------------------------------------------------------------------
// Parametrised test fixture
// ---------------------------------------------------------------------------

class SimilarityTest : public ::testing::TestWithParam<KernelTestParam> {};

// ---------------------------------------------------------------------------
// dot_product_f32 — SIMD result must match the scalar reference
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, DotProduct)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    float want = ref_dot(a, b);
    float got  = dot_product_f32(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, want, 1e-4f * std::abs(want));
}

// ---------------------------------------------------------------------------
// dot_product_f32 with non-zero offsets — exercises the aoffset/boffset path
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, DotProductWithOffset)
{
    const size_t n      = GetParam().length;
    const size_t prefix = 3; // arbitrary prefix that must be ignored

    // Pad the front with values that must not contribute to the result.
    std::vector<float> a_pad(prefix + n);
    std::vector<float> b_pad(prefix + n);
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);
    std::copy(a.begin(), a.end(), a_pad.begin() + prefix);
    std::copy(b.begin(), b.end(), b_pad.begin() + prefix);

    float want = ref_dot(a, b);
    float got  = dot_product_f32(a_pad.data(), prefix, b_pad.data(), prefix, n);

    EXPECT_NEAR(got, want, 1e-4f * std::abs(want));
}

// ---------------------------------------------------------------------------
// euclidean_f32 — squared L2 distance
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, Euclidean)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    float want = ref_euclidean(a, b);
    float got  = euclidean_f32(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, want, 1e-4f * std::abs(want));
}

// ---------------------------------------------------------------------------
// euclidean_f32 — identical vectors should give exactly 0.0
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, EuclideanSameVector)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n, 0.9f);

    float got = euclidean_f32(a.data(), 0, a.data(), 0, n);

    // Exact zero is expected since a == b; scale tolerance with length to
    // allow for FMA reassociation differences across ISAs.
    EXPECT_NEAR(got, 0.0f, 1e-6f * static_cast<float>(n));
}

// ---------------------------------------------------------------------------
// cosine_f32 — cosine similarity
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, Cosine)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    float want = ref_cosine(a, b);
    float got  = cosine_f32(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, want, 1e-4f * std::abs(want));
}

// ---------------------------------------------------------------------------
// cosine_f32 — parallel vectors should give similarity = 1.0
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, CosineParallelVectors)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n, 1.0f);

    // b = 2*a — same direction, different magnitude → cosine = 1.0
    std::vector<float> b(n);
    for (size_t i = 0; i < n; ++i) b[i] = 2.0f * a[i];

    float got = cosine_f32(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// cosine_f32 — orthogonal vectors should give similarity ≈ 0.0
//
// Orthogonality is constructed analytically for even n (alternating +/-):
//   a = [+1, +1, +1, ...]
//   b = [+1, -1, +1, -1, ...] — then a·b = 0 if n is even.
// For odd n we only use n-1 elements (prefix) so the dot is still zero.
// ---------------------------------------------------------------------------

TEST_P(SimilarityTest, CosineOrthogonalVectors)
{
    const size_t n = GetParam().length;
    if (n < 2) GTEST_SKIP() << "need at least 2 elements for orthogonality";

    const size_t even_n = n - (n % 2); // largest even prefix

    std::vector<float> a(n, 0.0f), b(n, 0.0f);
    for (size_t i = 0; i < even_n; ++i) {
        a[i] = 1.0f;
        b[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    float got = cosine_f32(a.data(), 0, b.data(), 0, n);

    // Generous tolerance: FP accumulation order differs between ISA tiers.
    EXPECT_NEAR(got, 0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Instantiation — named using the description field
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    SimilarityTest,
    ::testing::ValuesIn(kKernelTestParams),
    [](const ::testing::TestParamInfo<KernelTestParam>& info) {
        return info.param.description;
    });

// ---------------------------------------------------------------------------
// ISA-tier sanity test: confirm JVECTOR_MAX_ISA cap is respected when set
// ---------------------------------------------------------------------------

TEST(IsaDispatch, MaxIsaEnvHonoured)
{
    const char* env    = jvector_simd_get_max_isa_env();
    const char* active = jvector_simd_get_active_isa();

    if (env == nullptr) {
        // No override — just report which tier was auto-detected.
        SUCCEED() << "No JVECTOR_MAX_ISA set; auto-selected: " << active;
        return;
    }

    // Tiers ordered by capability (ascending index = lower capability).
    static const char* kOrder[] = {"sse42", "avx2", "avx3", "avx3_dl", "avx3_spr"};
    auto tier_idx = [](const char* name) -> int {
        for (int i = 0; i < 5; ++i)
            if (std::strcmp(kOrder[i], name) == 0) return i;
        return -1;
    };

    int env_idx    = tier_idx(env);
    int active_idx = tier_idx(active);

    ASSERT_GE(env_idx, 0)    << "Unrecognised JVECTOR_MAX_ISA value: " << env;
    ASSERT_GE(active_idx, 0) << "Unrecognised active ISA: " << active;

    // Active tier must be <= requested cap.
    EXPECT_LE(active_idx, env_idx)
        << "Active ISA (" << active << ") exceeds requested cap (" << env << ")";
}
