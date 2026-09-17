// Copyright DataStax, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Raw FFI bindings for `libjvector.so`.
//!
//! Every symbol in this module maps 1-to-1 onto an exported C function from
//! `jvector_simd.h` / `jvector_simd_kernel_list.h`.  All functions are
//! `unsafe`; callers are responsible for passing valid pointers and lengths.
//!
//! # Build
//!
//! `build.rs` locates `libjvector.so` and emits the necessary
//! `cargo:rustc-link-*` directives.  By default it looks for the library in
//! the `builddir/` directory relative to the native source tree.  Set
//! `JVECTOR_LIB_DIR` to override:
//!
//! ```text
//! JVECTOR_LIB_DIR=/custom/path cargo build
//! ```

use std::os::raw::{c_char, c_float, c_int, c_uchar};

// i64 is the Rust equivalent of C int64_t on all platforms we target.
#[allow(non_camel_case_types)]
type int64_t = i64;

#[link(name = "jvector")]
extern "C" {
    // -------------------------------------------------------------------------
    // Vector similarity
    // -------------------------------------------------------------------------

    /// Cosine similarity between two f32 slices (with byte offsets).
    pub fn cosine_f32(
        a: *const c_float,
        aoffset: usize,
        b: *const c_float,
        boffset: usize,
        length: usize,
    ) -> c_float;

    /// Dot product of two f32 slices (with byte offsets).
    pub fn dot_product_f32(
        a: *const c_float,
        aoffset: usize,
        b: *const c_float,
        boffset: usize,
        length: usize,
    ) -> c_float;

    /// Squared Euclidean distance between two f32 slices (with byte offsets).
    pub fn euclidean_f32(
        a: *const c_float,
        aoffset: usize,
        b: *const c_float,
        boffset: usize,
        length: usize,
    ) -> c_float;

    // -------------------------------------------------------------------------
    // Element-wise in-place arithmetic
    // -------------------------------------------------------------------------

    /// v1[i] += v2[i]  for i in 0..length
    pub fn add_in_place_f32(v1: *mut c_float, v2: *const c_float, length: usize);

    /// v1[i] += value  for i in 0..length
    pub fn add_scalar_in_place_f32(v1: *mut c_float, value: c_float, length: usize);

    /// v1[i] -= v2[i]  for i in 0..length
    pub fn sub_in_place_f32(v1: *mut c_float, v2: *const c_float, length: usize);

    /// v1[i] -= value  for i in 0..length
    pub fn sub_scalar_in_place_f32(v1: *mut c_float, value: c_float, length: usize);

    /// Returns the maximum element in v[0..length].
    pub fn max_f32(v: *const c_float, length: usize) -> c_float;

    /// v1[i] = min(v1[i], v2[i])  for i in 0..length
    pub fn min_in_place_f32(v1: *mut c_float, v2: *const c_float, length: usize);

    // -------------------------------------------------------------------------
    // PQ (Product Quantization) kernels
    // -------------------------------------------------------------------------

    pub fn assemble_and_sum_f32(
        data: *const c_float,
        data_base: c_int,
        base_offsets: *const c_uchar,
        base_offsets_offset: c_int,
        base_offsets_length: usize,
    ) -> c_float;

    pub fn assemble_and_sum_pq_f32(
        data: *const c_float,
        subspace_count: usize,
        base_offsets1: *const c_uchar,
        base_offsets_offset1: c_int,
        base_offsets2: *const c_uchar,
        base_offsets_offset2: c_int,
        cluster_count: c_int,
    ) -> c_float;

    pub fn pq_decoded_cosine_similarity_f32(
        base_offsets: *const c_uchar,
        base_offsets_offset: c_int,
        base_offsets_length: usize,
        cluster_count: c_int,
        partial_sums: *const c_float,
        a_magnitude: *const c_float,
        b_magnitude: c_float,
    ) -> c_float;

    pub fn calculate_partial_sums_dot_f32(
        codebook: *const c_float,
        codebook_index: c_int,
        size: usize,
        cluster_count: c_int,
        query: *const c_float,
        query_offset: c_int,
        partial_sums: *mut c_float,
    );

    pub fn calculate_partial_sums_euclidean_f32(
        codebook: *const c_float,
        codebook_index: c_int,
        size: usize,
        cluster_count: c_int,
        query: *const c_float,
        query_offset: c_int,
        partial_sums: *mut c_float,
    );

    pub fn calculate_partial_sums_self_magnitude_f32(
        codebook: *const c_float,
        codebook_index: c_int,
        size: usize,
        cluster_count: c_int,
        partial_sums: *mut c_float,
    );

    // -------------------------------------------------------------------------
    // NVQ (Non-uniform Vector Quantization) kernels
    // -------------------------------------------------------------------------

    pub fn nvq_quantize_8bit(
        vector: *const c_float,
        length: usize,
        alpha: c_float,
        x0: c_float,
        min_value: c_float,
        max_value: c_float,
        destination: *mut c_uchar,
    );

    pub fn nvq_loss(
        vector: *const c_float,
        length: usize,
        alpha: c_float,
        x0: c_float,
        min_value: c_float,
        max_value: c_float,
        n_bits: c_int,
    ) -> c_float;

    pub fn nvq_uniform_loss(
        vector: *const c_float,
        length: usize,
        min_value: c_float,
        max_value: c_float,
        n_bits: c_int,
    ) -> c_float;

    pub fn nvq_square_l2_distance_8bit(
        vector: *const c_float,
        quantized: *const c_uchar,
        length: usize,
        alpha: c_float,
        x0: c_float,
        min_value: c_float,
        max_value: c_float,
    ) -> c_float;

    pub fn nvq_dot_product_8bit(
        vector: *const c_float,
        quantized: *const c_uchar,
        length: usize,
        alpha: c_float,
        x0: c_float,
        min_value: c_float,
        max_value: c_float,
    ) -> c_float;

    pub fn nvq_cosine_8bit_packed(
        vector: *const c_float,
        quantized: *const c_uchar,
        length: usize,
        alpha: c_float,
        x0: c_float,
        min_value: c_float,
        max_value: c_float,
        centroid: *const c_float,
    ) -> int64_t;

    pub fn nvq_shuffle_query_in_place_8bit(vector: *mut c_float, length: usize);

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /// Returns the ISA tier selected at library init time, e.g. `"avx3"`.
    /// The pointer is a string literal owned by the library; do not free it.
    pub fn jvector_simd_get_active_isa() -> *const c_char;

    /// Returns the value of `JVECTOR_MAX_ISA` read at init time, or NULL.
    /// The pointer (when non-null) is a string literal; do not free it.
    pub fn jvector_simd_get_max_isa_env() -> *const c_char;
}
