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

// Tests for int8 vector similarity kernels: dot_product_i8, euclidean_i8, cosine_i8.
//
// The kernels operate on signed int8_t vectors and return a float result:
//   dot_product_i8  — (float) sum(a[i] * b[i])
//   euclidean_i8    — (float) sum((a[i] - b[i])^2)   (squared L2 distance)
//   cosine_i8       — (float) dot(a,b) / sqrt(||a||^2 * ||b||^2)
//
// On AVX3_DL (Ice Lake+) these are overridden with VNNI (VPDPBUSD/VPDPWSSD)
// implementations; on all other tiers the generic Highway path is used.
//
// All tests are parametrised over kKernelTestParams (defined in test_helpers.cpp),
// which covers every ISA-tier loop-boundary for both f32 and i8 kernels, including
// the VNNI-specific 64/128/256-byte unroll boundaries added for the i8 suite.

#include "test_helpers.h"

// ---------------------------------------------------------------------------
// Reference scalar implementations
// ---------------------------------------------------------------------------

static float ref_dot_i8(const std::vector<int8_t>& a, const std::vector<int8_t>& b)
{
    int64_t s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return static_cast<float>(s);
}

static float ref_euclidean_i8(const std::vector<int8_t>& a, const std::vector<int8_t>& b)
{
    int64_t s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        s += d * d;
    }
    return static_cast<float>(s);
}

static float ref_cosine_i8(const std::vector<int8_t>& a, const std::vector<int8_t>& b)
{
    int64_t dot = 0, normA = 0, normB = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t ai = a[i], bi = b[i];
        dot   += static_cast<int64_t>(ai) * bi;
        normA += static_cast<int64_t>(ai) * ai;
        normB += static_cast<int64_t>(bi) * bi;
    }
    return static_cast<float>(dot / std::sqrt(static_cast<double>(normA)
                                              * static_cast<double>(normB)));
}

// ---------------------------------------------------------------------------
// Parametrised test fixture
// ---------------------------------------------------------------------------

class SimilarityI8Test : public ::testing::TestWithParam<KernelTestParam> {};

// ---------------------------------------------------------------------------
// dot_product_i8 — SIMD result must match the scalar reference
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, DotProduct)
{
    const size_t n = GetParam().length;
    auto a = make_vec_i8(n, 7);
    auto b = make_vec_i8(n, 11);

    const float want = ref_dot_i8(a, b);
    const float got  = dot_product_i8(a.data(), 0, b.data(), 0, n);

    // Integer accumulation with a single int64→float cast — result is exact.
    EXPECT_EQ(got, want);
}

// ---------------------------------------------------------------------------
// dot_product_i8 with non-zero offsets — exercises the aoffset/boffset path
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, DotProductWithOffset)
{
    const size_t n      = GetParam().length;
    const size_t prefix = 5; // arbitrary prefix that must be ignored

    std::vector<int8_t> a_pad(prefix + n, 0);
    std::vector<int8_t> b_pad(prefix + n, 0);
    auto a = make_vec_i8(n, 7);
    auto b = make_vec_i8(n, 11);
    std::copy(a.begin(), a.end(), a_pad.begin() + prefix);
    std::copy(b.begin(), b.end(), b_pad.begin() + prefix);

    const float want = ref_dot_i8(a, b);
    const float got  = dot_product_i8(a_pad.data(), prefix, b_pad.data(), prefix, n);

    EXPECT_EQ(got, want);
}

// ---------------------------------------------------------------------------
// dot_product_i8 — zero vector gives exactly 0.0
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, DotProductZeroVector)
{
    const size_t n = GetParam().length;
    auto a = make_vec_i8(n, 7);
    std::vector<int8_t> z(n, 0);

    EXPECT_EQ(dot_product_i8(a.data(), 0, z.data(), 0, n), 0.0f);
}

// ---------------------------------------------------------------------------
// euclidean_i8 — SIMD result must match the scalar reference
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, Euclidean)
{
    const size_t n = GetParam().length;
    auto a = make_vec_i8(n, 7);
    auto b = make_vec_i8(n, 11);

    const float want = ref_euclidean_i8(a, b);
    const float got  = euclidean_i8(a.data(), 0, b.data(), 0, n);

    // Integer accumulation with a single int64→float cast — result is exact.
    EXPECT_EQ(got, want);
}

// ---------------------------------------------------------------------------
// euclidean_i8 — identical vectors must give exactly 0
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, EuclideanSameVector)
{
    const size_t n = GetParam().length;
    auto a = make_vec_i8(n, 9);

    const float got = euclidean_i8(a.data(), 0, a.data(), 0, n);

    EXPECT_EQ(got, 0.0f);
}

// ---------------------------------------------------------------------------
// cosine_i8 — SIMD result must match the scalar reference
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, Cosine)
{
    const size_t n = GetParam().length;
    auto a = make_vec_i8(n, 7);
    auto b = make_vec_i8(n, 11);

    const float want = ref_cosine_i8(a, b);
    const float got  = cosine_i8(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, want, 1e-5f);
}

// ---------------------------------------------------------------------------
// cosine_i8 — parallel vectors (b = k*a, k > 0) should give similarity ≈ 1.0
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, CosineParallelVectors)
{
    const size_t n = GetParam().length;
    // Use small magnitudes so that 2*val stays within int8 range.
    auto a = make_vec_i8(n, 3);
    std::vector<int8_t> b(n);
    for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<int8_t>(std::max(-127, std::min(127, 2 * static_cast<int>(a[i]))));

    const float got = cosine_i8(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// cosine_i8 — orthogonal vectors should give similarity ≈ 0.0
//
// Same analytic construction as the f32 test: for even n,
//   a = [+1, +1, +1, ...]
//   b = [+1, -1, +1, -1, ...]  →  dot(a,b) = 0.
// Odd n: the odd last element is zeroed out on b (unchanged on a) so the
// dot product remains zero without affecting the norms materially.
// ---------------------------------------------------------------------------

TEST_P(SimilarityI8Test, CosineOrthogonalVectors)
{
    const size_t n = GetParam().length;
    if (n < 2) GTEST_SKIP() << "need at least 2 elements for orthogonality";

    const size_t even_n = n - (n % 2);

    std::vector<int8_t> a(n, 0), b(n, 0);
    for (size_t i = 0; i < even_n; ++i) {
        a[i] = 1;
        b[i] = (i % 2 == 0) ? 1 : -1;
    }

    const float got = cosine_i8(a.data(), 0, b.data(), 0, n);

    EXPECT_NEAR(got, 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Instantiation — named using the description field
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    SimilarityI8Test,
    ::testing::ValuesIn(kKernelTestParams),
    [](const ::testing::TestParamInfo<KernelTestParam>& info) {
        return info.param.description;
    });
