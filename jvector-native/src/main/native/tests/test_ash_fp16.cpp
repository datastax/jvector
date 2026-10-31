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
#include "jvector_ash_fp16.h"
#include <gtest/gtest.h>
#include <array>
#include <future>
#include <limits>
#include <memory>
#include <vector>

TEST(AshFp16, InvalidArgumentsDoNotCrossTheAbi) {
    const float matrix[] = {1};
    jvector_ash_fp16_result result{};
    EXPECT_EQ(jvector_ash_fp16_abi(), 1);
    EXPECT_EQ(jvector_ash_fp16_create(nullptr, 1, 1, &result), nullptr);
    EXPECT_EQ(result.status, JV_ASH_FP16_INVALID_ARGUMENT);
    EXPECT_EQ(jvector_ash_fp16_create(matrix, 0, 1, &result), nullptr);
    EXPECT_EQ(result.status, JV_ASH_FP16_INVALID_ARGUMENT);
    EXPECT_EQ(jvector_ash_fp16_create(matrix, 1, -1, &result), nullptr);
    EXPECT_EQ(jvector_ash_fp16_create(matrix, 1, 1, nullptr), nullptr);
    EXPECT_EQ(jvector_ash_fp16_project(nullptr, nullptr, nullptr, &result), JV_ASH_FP16_INVALID_ARGUMENT);
    EXPECT_EQ(jvector_ash_fp16_bytes(nullptr), 0);
    jvector_ash_fp16_destroy(nullptr);
}

TEST(AshFp16, RejectsMatrixRangeBeforeCreatingContext) {
    for (float value : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(), 65505.0f}) {
        jvector_ash_fp16_result result{};
        EXPECT_EQ(jvector_ash_fp16_create(&value, 1, 1, &result), nullptr);
        EXPECT_EQ(result.status, JV_ASH_FP16_RANGE);
    }
}

TEST(AshFp16, OddDimensionsConcurrentUseAndModelSwitching) {
    const float matrix[] = {.25f, -.5f, .75f, -.125f, .5f, .25f};
    jvector_ash_fp16_result result{};
    using Owner = std::unique_ptr<void, decltype(&jvector_ash_fp16_destroy)>;
    Owner a(jvector_ash_fp16_create(matrix, 2, 3, &result), &jvector_ash_fp16_destroy);
    if (result.status == JV_ASH_FP16_UNSUPPORTED) GTEST_SKIP() << result.message;
    ASSERT_EQ(result.status, JV_ASH_FP16_OK) << result.message;
    ASSERT_NE(a, nullptr);
    EXPECT_GT(jvector_ash_fp16_bytes(a.get()), 0);
    const float second[] = {2, -1, .5f};
    Owner b(jvector_ash_fp16_create(second, 1, 3, &result), &jvector_ash_fp16_destroy);
    ASSERT_EQ(result.status, JV_ASH_FP16_OK);
    std::vector<std::future<void>> workers;
    for (int thread = 0; thread < 4; ++thread) workers.push_back(std::async(std::launch::async, [&] {
        const float input[] = {1, 1, 1};
        float out[2]{};
        jvector_ash_fp16_result call_result{};
        for (int repeat = 0; repeat < 25; ++repeat) {
            EXPECT_EQ(jvector_ash_fp16_project(a.get(), input, out, &call_result), JV_ASH_FP16_OK);
            EXPECT_FLOAT_EQ(out[0], .5f);
            EXPECT_FLOAT_EQ(out[1], .625f);
            EXPECT_EQ(jvector_ash_fp16_project(b.get(), input, out, &call_result), JV_ASH_FP16_OK);
            EXPECT_FLOAT_EQ(out[0], 1.5f);
        }
    }));
    for (auto& worker : workers) worker.get();
    float out[2];
    const float input[] = {1, 1, 1};
    EXPECT_EQ(jvector_ash_fp16_project(a.get(), nullptr, out, &result), JV_ASH_FP16_INVALID_ARGUMENT);
    EXPECT_EQ(jvector_ash_fp16_project(a.get(), input, nullptr, &result), JV_ASH_FP16_INVALID_ARGUMENT);
    EXPECT_EQ(jvector_ash_fp16_project(a.get(), out, out, &result), JV_ASH_FP16_INVALID_ARGUMENT);
    for (float value : {std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN(), 65505.0f}) {
        float bad[] = {value, 1, 1};
        EXPECT_EQ(jvector_ash_fp16_project(a.get(), bad, out, &result), JV_ASH_FP16_RANGE);
    }
}
