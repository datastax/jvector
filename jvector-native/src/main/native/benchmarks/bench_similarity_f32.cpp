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

// Google Benchmark micro-benchmarks for the fp32 vector similarity kernels:
//   cosine_f32, dot_product_f32, euclidean_f32
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
#include <cmath>
#include <vector>

#include "jvector_simd.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Deterministic, non-zero float vector: avoids degenerate cosine=NaN cases.
static std::vector<float> make_vec(size_t n, float seed)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = seed * (1.0f + static_cast<float>(i % 7) * 0.13f);
        if (i % 3 == 0) v[i] = -v[i];
        v[i] += 0.5f;
    }
    return v;
}

// Benchmark sizes matching production embedding dimensions.
static const std::vector<int64_t> kBenchSizes = {128, 256, 512, 1024, 1536, 3072};

// ---------------------------------------------------------------------------
// dot_product_f32
// ---------------------------------------------------------------------------

static void BM_dot_product_f32(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    for (auto _ : state) {
        float result = dot_product_f32(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(float));
}
BENCHMARK(BM_dot_product_f32)->ArgsProduct({kBenchSizes});

// ---------------------------------------------------------------------------
// euclidean_f32
// ---------------------------------------------------------------------------

static void BM_euclidean_f32(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    for (auto _ : state) {
        float result = euclidean_f32(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(float));
}
BENCHMARK(BM_euclidean_f32)->ArgsProduct({kBenchSizes});

// ---------------------------------------------------------------------------
// cosine_f32
// ---------------------------------------------------------------------------

static void BM_cosine_f32(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = make_vec(n, 0.7f);
    auto b = make_vec(n, 1.3f);

    for (auto _ : state) {
        float result = cosine_f32(a.data(), 0, b.data(), 0, n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(float));
}
BENCHMARK(BM_cosine_f32)->ArgsProduct({kBenchSizes});

// ---------------------------------------------------------------------------
// Entry point — benchmark::Initialize parses --benchmark_* flags.
// ---------------------------------------------------------------------------

BENCHMARK_MAIN();
