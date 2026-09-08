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

// Google Benchmark micro-benchmarks for the int8 vector similarity kernels:
//   dot_product_i8, euclidean_i8, cosine_i8
//
// Parameterised over the realistic embedding dimensions used in production:
//   128, 256, 512, 1024, 1536, 3072
//
// Build (requires google-benchmark installed or available via pkg-config):
//   meson setup build && ninja -C build bench_simd_kernels
//
// Run:
//   ./build/bench_simd_kernels [--benchmark_filter=<pattern>]

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

#include "jvector_simd.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Deterministic, non-zero int8 vector: values cycle through a signed range to
// avoid degenerate all-zero inputs while staying within [-128, 127].
static std::vector<int8_t> make_i8_vec(size_t n, int8_t seed)
{
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        int val = seed + static_cast<int>(i % 127);
        if (i % 3 == 0) val = -val;
        // clamp to [-127, 127] to keep vectors non-degenerate for cosine
        if (val >  127) val =  127;
        if (val < -127) val = -127;
        v[i] = static_cast<int8_t>(val);
    }
    return v;
}

// Benchmark sizes matching production embedding dimensions.
static const std::vector<int64_t> kBenchSizes = {128, 256, 512, 1024, 1536, 3072};

// ---------------------------------------------------------------------------
// dot_product_i8
// ---------------------------------------------------------------------------

static void BM_dot_product_i8(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_i8_vec(n, 7);
    auto b = make_i8_vec(n, 13);

    for (auto _ : state) {
        float result = dot_product_i8(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(int8_t));
}
BENCHMARK(BM_dot_product_i8)->ArgsProduct({kBenchSizes});

// ---------------------------------------------------------------------------
// euclidean_i8
// ---------------------------------------------------------------------------

static void BM_euclidean_i8(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_i8_vec(n, 7);
    auto b = make_i8_vec(n, 13);

    for (auto _ : state) {
        float result = euclidean_i8(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(int8_t));
}
BENCHMARK(BM_euclidean_i8)->ArgsProduct({kBenchSizes});

// ---------------------------------------------------------------------------
// cosine_i8
// ---------------------------------------------------------------------------

static void BM_cosine_i8(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_i8_vec(n, 7);
    auto b = make_i8_vec(n, 13);

    for (auto _ : state) {
        float result = cosine_i8(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(int8_t));
}
BENCHMARK(BM_cosine_i8)->ArgsProduct({kBenchSizes});

