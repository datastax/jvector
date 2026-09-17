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

//! Demonstrates calling the three vector similarity kernels in libjvector.so:
//!   cosine_f32, dot_product_f32, euclidean_f32
//!
//! Run from the rust/ directory:
//!   cargo run --example similarity_demo

use std::ffi::CStr;

use jvector::{cosine_f32, dot_product_f32, euclidean_f32, jvector_simd_get_active_isa};

fn main() {
    // Print which ISA tier was selected at library init time.
    let isa = unsafe { CStr::from_ptr(jvector_simd_get_active_isa()) }
        .to_str()
        .unwrap_or("unknown");
    println!("Active ISA : {isa}");
    println!();

    // Two simple 4-dimensional vectors.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0];

    // All three kernels accept raw pointers, element offsets (in elements, not
    // bytes), and a length.  Passing offset=0 means "start from the beginning".
    let dot = unsafe { dot_product_f32(a.as_ptr(), 0, b.as_ptr(), 0, a.len()) };
    let cosine = unsafe { cosine_f32(a.as_ptr(), 0, b.as_ptr(), 0, a.len()) };
    let euclidean = unsafe { euclidean_f32(a.as_ptr(), 0, b.as_ptr(), 0, a.len()) };

    println!("a          = {:?}", a);
    println!("b          = {:?}", b);
    println!();
    println!("dot product  = {dot:.6}");
    println!("cosine sim   = {cosine:.6}");
    println!("euclidean    = {euclidean:.6}   (squared L2 distance)");
    println!();

    // Demonstrate using offsets: treat the second half of a larger slice as the
    // input vector, skipping the first `offset` elements.
    let padded_a: Vec<f32> = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]; // real data starts at index 2
    let padded_b: Vec<f32> = vec![0.0, 0.0, 4.0, 3.0, 2.0, 1.0];
    let offset = 2_usize;
    let length = 4_usize;

    let dot_offset = unsafe {
        dot_product_f32(
            padded_a.as_ptr(),
            offset,
            padded_b.as_ptr(),
            offset,
            length,
        )
    };
    println!("dot product (with offset={offset}) = {dot_offset:.6}  (same result: {dot:.6})");
}
