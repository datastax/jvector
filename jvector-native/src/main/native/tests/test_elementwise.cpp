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
// sub_f32: res[i] = a[aoffset + i] - b[boffset + i]
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, SubF32)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n + 7, 1.1f);
    auto b = make_vec(n + 7, 0.7f);
    std::vector<float> res(n, 0.0f);
    const size_t aoffset = 3;
    const size_t boffset = 2;

    sub_f32(a.data(), aoffset, b.data(), boffset, res.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float want = a[aoffset + i] - b[boffset + i];
        EXPECT_NEAR(res[i], want, 1e-5f) << "sub_f32[" << i << "]";
    }
}

// ---------------------------------------------------------------------------
// sub_scalar_f32: res[i] = a[aoffset + i] - value
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, SubScalarF32)
{
    const size_t n = GetParam().length;
    auto a = make_vec(n + 5, 1.1f);
    std::vector<float> res(n, 0.0f);
    const size_t aoffset = 2;
    const float  scalar  = 1.618f;

    sub_scalar_f32(a.data(), aoffset, scalar, res.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float want = a[aoffset + i] - scalar;
        EXPECT_NEAR(res[i], want, 1e-5f) << "sub_scalar_f32[" << i << "]";
    }
}

// ---------------------------------------------------------------------------
// scale_f32:  v[i] *= multiplier
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, ScaleF32)
{
    const size_t n          = GetParam().length;
    const float  multiplier = 2.5f;
    auto v    = make_vec(n, 1.1f);
    auto want = v;
    for (size_t i = 0; i < n; ++i) want[i] *= multiplier;

    auto got = v;
    scale_f32(got.data(), multiplier, n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], want[i], 1e-5f) << "scale_f32[" << i << "]";
}

TEST(ScaleF32, AlignmentAndSpecialValues)
{
    const float values[] = {0.0f, -0.0f, 1.25f, -2.5f, INFINITY, -INFINITY, NAN};
    const size_t lengths[] = {0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129,
                              511, 512, 513, 767, 768, 769, 1536, 3072};
    alignas(64) float storage[3072 + 32];
    for (size_t offset = 0; offset < 16; ++offset) {
        float* v = storage + 16 + offset;
        for (size_t n : lengths) {
            for (float multiplier : values) {
                SCOPED_TRACE(::testing::Message() << "offset=" << offset << " length=" << n
                                                 << " multiplier=" << multiplier);
                std::fill_n(storage, 3072 + 32, 42.0f);
                for (size_t i = 0; i < n; ++i) v[i] = values[i % 7];
                scale_f32(v, multiplier, n);
                for (size_t i = 0; i < n; ++i) {
                    const float expected = values[i % 7] * multiplier;
                    if (std::isnan(expected)) {
                        ASSERT_TRUE(std::isnan(v[i])) << "index=" << i;
                    } else {
                        ASSERT_EQ(v[i], expected) << "index=" << i;
                        ASSERT_EQ(std::signbit(v[i]), std::signbit(expected)) << "index=" << i;
                    }
                }
                EXPECT_EQ(v[-1], 42.0f);
                EXPECT_EQ(v[n], 42.0f);
            }
        }
    }
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
// min_f32: returns the minimum element
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, MinF32)
{
    const size_t n = GetParam().length;
    auto v = make_vec(n, 0.9f);

    float want = *std::min_element(v.begin(), v.end());
    float got  = min_f32(v.data(), n);

    EXPECT_FLOAT_EQ(got, want);
}

// min_f32 on a vector with a known minimum at the last position (tail element)
TEST_P(ElementWiseTest, MinF32TailElement)
{
    const size_t n = GetParam().length;
    auto v = make_vec(n, 0.5f);
    // Place the global minimum in the very last element — exercises tail path.
    v.back() = -1e6f;

    float got = min_f32(v.data(), n);

    EXPECT_FLOAT_EQ(got, -1e6f);
}

// A NaN must survive later finite values in the same SIMD lane as well as tails.
TEST_P(ElementWiseTest, MinF32NaNAtEveryPosition)
{
    const size_t n = GetParam().length;
    for (size_t pos = 0; pos < n; ++pos) {
        std::vector<float> v(n, 2.0f);
        v[pos] = NAN;
        EXPECT_TRUE(std::isnan(min_f32(v.data(), n))) << "NaN at " << pos;
    }
}

TEST_P(ElementWiseTest, MinF32SignedZeroAtEveryPosition)
{
    const size_t n = GetParam().length;
    std::vector<float> v(n, 0.0f);
    EXPECT_FALSE(std::signbit(min_f32(v.data(), n)));
    for (size_t pos = 0; pos < n; ++pos) {
        v.assign(n, 0.0f);
        v[pos] = -0.0f;
        float result = min_f32(v.data(), n);
        EXPECT_EQ(result, 0.0f);
        EXPECT_TRUE(std::signbit(result)) << "negative zero at " << pos;
        // A negative finite minimum still takes precedence over negative zero.
        v[(pos + 1) % n] = -1.0f;
        EXPECT_EQ(min_f32(v.data(), n), -1.0f);
    }
}

// ---------------------------------------------------------------------------
// sum_f32: returns the sum of elements
// ---------------------------------------------------------------------------

TEST_P(ElementWiseTest, SumF32)
{
    const size_t n = GetParam().length;
    auto v = make_vec(n, 0.8f);

    float want = 0.0f;
    for (size_t i = 0; i < n; ++i) want += v[i];
    float got  = sum_f32(v.data(), n);

    EXPECT_NEAR(got, want, 1e-4f * std::max(1.0f, std::abs(want)));
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

// ---------------------------------------------------------------------------
// ISA-tier sanity test: active ISA must be a recognised tier name
// ---------------------------------------------------------------------------

TEST(IsaDispatch, ActiveIsaIsKnownTier)
{
    const char* active = jvector_simd_get_active_isa();
    ASSERT_NE(active, nullptr) << "jvector_simd_get_active_isa() returned null";

#if defined(__aarch64__)
    static const char* kOrder[] = {"neon", "sve", "sve2"};
    constexpr int kOrderLen = 3;
#else
    static const char* kOrder[] = {"sse42", "avx2", "avx3", "avx3_dl", "avx3_spr"};
    constexpr int kOrderLen = 5;
#endif
    bool found = false;
    for (int i = 0; i < kOrderLen; ++i)
        if (std::strcmp(kOrder[i], active) == 0) { found = true; break; }

    EXPECT_TRUE(found) << "Active ISA '" << active << "' is not a recognised tier";
}
