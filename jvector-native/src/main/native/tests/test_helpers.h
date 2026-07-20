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

// Shared helpers for the test_simd_kernels test binary.
// Included by each test .cpp file; defined in test_helpers.cpp.

#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "jvector_simd.h"

// ---------------------------------------------------------------------------
// Deterministic test vectors.
// make_vec(n, seed) produces n floats with a mix of signs and magnitudes
// so that no element is exactly zero (important for cosine tests).
// ---------------------------------------------------------------------------

std::vector<float>   make_vec(size_t n, float seed);

// make_vec_i8(n, seed) produces n int8_t values with a mix of signs
// suitable for testing the i8 similarity kernels.
std::vector<int8_t>  make_vec_i8(size_t n, int8_t seed);

// ---------------------------------------------------------------------------
// Shared test parameter — vector length + human-readable path description.
// Used by every parametrised test suite in the binary so the same set of
// sizes exercises each kernel.
// ---------------------------------------------------------------------------

struct KernelTestParam {
    size_t      length;
    std::string description;
};

// The canonical set of sizes that hits every code path across ISA tiers.
// See the top-of-file comment in test_similarity.cpp for the full breakdown.
extern const std::vector<KernelTestParam> kKernelTestParams;
