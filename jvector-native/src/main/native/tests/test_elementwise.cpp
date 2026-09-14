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

// Tests for element-wise in-place arithmetic kernels:
//   add_in_place_f32, add_scalar_in_place_f32,
//   sub_in_place_f32, sub_scalar_in_place_f32,
//   max_f32, min_in_place_f32.
//
// Vector sizes cover the same ISA boundary / tail matrix as test_similarity.cpp:
//   size  1        — tail only
//   size  3        — tail only
//   size  4        — SSE42 exact / AVX2+AVX512 capped path
//   size  7        — SSE42 1 full + 3-tail
//   size  8        — AVX2 exact / AVX512 capped
//   size  15       — AVX2 1 full + 7-tail
//   size  16       — SSE42 4× main exact / AVX512 1 full
//   size  17       — SSE42 4× main + 1-tail
//   size  32       — AVX2 4× main exact
//   size  37       — AVX2 4× main + 5-tail
//   size  64       — AVX512 4× main exact
//   size  71       — AVX512 4× main + 7-tail
//   size 100, 128, 255 — large mixed / power-of-2 / odd

#include "test_helpers.h"

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class ElementWiseTest : public ::testing::TestWithParam<KernelTestParam> {};

// ---------------------------------------------------------------------------
// add_in_place_f32:  v1[i] += v2[i]
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, AddInPlace)
{
    const size_t n = GetParam().length;
    auto v1   = make_vec(n, 1.1f);
    auto v2   = make_vec(n, 0.7f);
    auto want = v1;
    for (size_t i = 0; i < n; ++i) want[i] += v2[i];

    auto got = v1;
    add_in_place_f32(got.data(), v2.data(), n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "add_in_place_f32[" << i << "]";
}

// ---------------------------------------------------------------------------
// add_scalar_in_place_f32:  v1[i] += scalar
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, AddScalarInPlace)
{
    const size_t n      = GetParam().length;
    const float  scalar = 3.14f;
    auto v1   = make_vec(n, 1.1f);
    auto want = v1;
    for (size_t i = 0; i < n; ++i) want[i] += scalar;

    auto got = v1;
    add_scalar_in_place_f32(got.data(), scalar, n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "add_scalar_in_place_f32[" << i << "]";
}

// ---------------------------------------------------------------------------
// sub_in_place_f32:  v1[i] -= v2[i]
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, SubInPlace)
{
    const size_t n = GetParam().length;
    auto v1   = make_vec(n, 1.1f);
    auto v2   = make_vec(n, 0.7f);
    auto want = v1;
    for (size_t i = 0; i < n; ++i) want[i] -= v2[i];

    auto got = v1;
    sub_in_place_f32(got.data(), v2.data(), n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "sub_in_place_f32[" << i << "]";
}

// ---------------------------------------------------------------------------
// sub_scalar_in_place_f32:  v1[i] -= scalar
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, SubScalarInPlace)
{
    const size_t n      = GetParam().length;
    const float  scalar = 2.71f;
    auto v1   = make_vec(n, 1.1f);
    auto want = v1;
    for (size_t i = 0; i < n; ++i) want[i] -= scalar;

    auto got = v1;
    sub_scalar_in_place_f32(got.data(), scalar, n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "sub_scalar_in_place_f32[" << i << "]";
}

// ---------------------------------------------------------------------------
// max_f32: returns the maximum element
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, MaxF32)
{
    const size_t n = GetParam().length;
    auto v = make_vec(n, 0.9f);

    float want = *std::max_element(v.begin(), v.end());
    float got  = max_f32(v.data(), n);

    EXPECT_FLOAT_EQ(got, want);
}

// max_f32 on a vector with a known maximum at the last position (tail element)
TEST_P(ElementWiseTest, MaxF32TailElement)
{
    const size_t n = GetParam().length;
    auto v = make_vec(n, 0.5f);
    // Place the global maximum in the very last element — exercises tail path.
    v.back() = 1e6f;

    float got = max_f32(v.data(), n);

    EXPECT_FLOAT_EQ(got, 1e6f);
}

// ---------------------------------------------------------------------------
// min_in_place_f32:  v1[i] = min(v1[i], v2[i])
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, MinInPlace)
{
    const size_t n = GetParam().length;
    auto v1   = make_vec(n, 1.1f);
    auto v2   = make_vec(n, 0.7f);
    auto want = v1;
    for (size_t i = 0; i < n; ++i) want[i] = std::min(want[i], v2[i]);

    auto got = v1;
    min_in_place_f32(got.data(), v2.data(), n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "min_in_place_f32[" << i << "]";
}

// add then sub back — result must equal the original vector
TEST_P(ElementWiseTest, AddSubRoundTrip)
{
    const size_t n = GetParam().length;
    auto original = make_vec(n, 1.3f);
    auto delta    = make_vec(n, 0.4f);

    auto v = original;
    add_in_place_f32(v.data(), delta.data(), n);
    sub_in_place_f32(v.data(), delta.data(), n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(v[i], original[i], 1e-5f) << "add_sub_roundtrip[" << i << "]";
}

// ---------------------------------------------------------------------------
// Instantiation
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    ElementWiseTest,
    ::testing::ValuesIn(kKernelTestParams),
    [](const ::testing::TestParamInfo<KernelTestParam>& info) {
        return info.param.description;
    });
