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

// AVX3_DL (Ice Lake / Icelake-SP) tier: ONLY kernels that require ICX-specific
// instructions unavailable in the AVX3 (-march=skylake-avx512) compilation
// belong here.  Generic kernels that Highway auto-vectorises identically under
// both marches must go in jvector_simd_kernels.cpp instead — that file is
// compiled once for AVX3 and its function pointers are reused by this tier and
// AVX3_SPR via vtable inheritance, avoiding any duplication in .text.
//
// ICX adds over AVX3 (HWY_TARGET_STR_AVX3_DL):
//   VNNI, VBMI, VBMI2, IFMA, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ
//
// Compiled with -march=icelake-server.
//
// This file uses raw Intel AVX-512 intrinsics directly — NO Google Highway —
// so we get exactly the instructions we intend with zero abstraction overhead.

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>        // AVX-512 + VNNI intrinsics
#include "jvector_simd.h"

// =============================================================================
// Register naming convention
//   zmm  = 512-bit  (16 × int32, 32 × int16, 64 × int8)
//   ymm  = 256-bit  (32 × int8,  16 × int16)
//   xmm  = 128-bit  (16 × int8,  8 × int16)
//
// VNNI instructions used
// ─────────────────────────────────────────────────────────────────────────────
// VPDPBUSD  zmm_acc, zmm_a, zmm_b
//   For each group of 4 adjacent lanes (i×4 .. i×4+3):
//     acc[i] += (u8)a[i×4+0] * (i8)b[i×4+0]
//             + (u8)a[i×4+1] * (i8)b[i×4+1]
//             + (u8)a[i×4+2] * (i8)b[i×4+2]
//             + (u8)a[i×4+3] * (i8)b[i×4+3]
//   → 16 i32 accumulations, 64 int8 products per zmm register per cycle.
//   Latency: 3 cycles.  Throughput: 1/cycle (two ports on Ice Lake).
//
// VPDPWSSD  zmm_acc, zmm_a, zmm_b
//   For each group of 2 adjacent i16 lanes (i×2, i×2+1):
//     acc[i] += (i16)a[i×2+0] * (i16)b[i×2+0]
//             + (i16)a[i×2+1] * (i16)b[i×2+1]
//   → 16 i32 accumulations, 32 int16 products per zmm per cycle.
//   Latency: 3 cycles.  Throughput: 1/cycle.
//
// Signed i8 × signed i8 using VPDPBUSD
// ─────────────────────────────────────────────────────────────────────────────
// VPDPBUSD requires operand A to be unsigned.  For signed inputs we apply the
// standard bias trick:
//   (a + 128) is always non-negative, so we use it as the unsigned operand.
//   (a+128) * b = a*b + 128*b  →  a*b = VPDPBUSD(a+128, b) - 128 * sum(b)
//
// The bias (128*sum(b)) is constant per zmm load of b, computed as:
//   _mm512_dpwssd_epi32(zero, b, set1_epi16(128))   [reuse VPDPWSSD]
// and subtracted once per iteration from the accumulator.
//
// This adds one VPDPWSSD + one VPADDD per iteration, which is negligible
// compared to the main VPDPBUSD throughput.
//
// Unrolling strategy
// ─────────────────────────────────────────────────────────────────────────────
// With 3-cycle VPDPBUSD latency and 1/cycle throughput (ports 0+5), we need
// at least 4 independent accumulator chains to keep the ports saturated:
//   issued cycle 0: port 0 ← acc0
//   issued cycle 1: port 5 ← acc1
//   issued cycle 2: port 0 ← acc2
//   issued cycle 3: port 5 ← acc3   (acc0 writeback done, cycle 3)
// 4× unrolling fully hides the 3-cycle latency.
// =============================================================================

namespace AVX3_DL {

// ---------------------------------------------------------------------------
// Horizontal reduce: sum all 16 int32 lanes of a zmm register.
// _mm512_reduce_add_epi32 emits the optimal fold-down sequence; the compiler
// schedules it across surrounding instructions better than manual shuffles.
// ---------------------------------------------------------------------------
static inline int32_t hsum_epi32(__m512i v)
{
    return _mm512_reduce_add_epi32(v);
}

// ---------------------------------------------------------------------------
// dot_product_i8  —  VPDPBUSD with bias correction for signed i8 × signed i8
// ---------------------------------------------------------------------------
//
// Algorithm
//   acc  = VPDPBUSD(acc,  a_u8, b_i8)   where a_u8 = a + 128
//   bias = VPDPWSSD(bias, b_i8, 128)    accumulates 128 * sum(b)
//   result = hsum(acc) - hsum(bias)
//
// 4× unrolled (256 bytes/iteration) to saturate both ICX VNNI ports and
// fully hide the 3-cycle VPDPBUSD latency.
float dot_product_i8(const int8_t * __restrict__ a, size_t aoffset,
                     const int8_t * __restrict__ b, size_t boffset,
                     size_t length)
{
    a += aoffset;
    b += boffset;

    const __m512i bias128 = _mm512_set1_epi16(128);

    __m512i acc0  = _mm512_setzero_si512(), acc1  = _mm512_setzero_si512();
    __m512i acc2  = _mm512_setzero_si512(), acc3  = _mm512_setzero_si512();
    __m512i bias0 = _mm512_setzero_si512(), bias1 = _mm512_setzero_si512();
    __m512i bias2 = _mm512_setzero_si512(), bias3 = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 256 <= length; i += 256) {
        __m512i va0 = _mm512_loadu_si512(a + i +   0);
        __m512i va1 = _mm512_loadu_si512(a + i +  64);
        __m512i va2 = _mm512_loadu_si512(a + i + 128);
        __m512i va3 = _mm512_loadu_si512(a + i + 192);
        __m512i vb0 = _mm512_loadu_si512(b + i +   0);
        __m512i vb1 = _mm512_loadu_si512(b + i +  64);
        __m512i vb2 = _mm512_loadu_si512(b + i + 128);
        __m512i vb3 = _mm512_loadu_si512(b + i + 192);

        // Flip sign bit: maps signed [-128,127] → unsigned [0,255].
        const __m512i flip = _mm512_set1_epi8(-128);
        __m512i au0 = _mm512_add_epi8(va0, flip);
        __m512i au1 = _mm512_add_epi8(va1, flip);
        __m512i au2 = _mm512_add_epi8(va2, flip);
        __m512i au3 = _mm512_add_epi8(va3, flip);

        // VPDPBUSD: acc[i] += (u8)au[4i+k] * (i8)vb[4i+k], k=0..3
        acc0 = _mm512_dpbusd_epi32(acc0, au0, vb0);
        acc1 = _mm512_dpbusd_epi32(acc1, au1, vb1);
        acc2 = _mm512_dpbusd_epi32(acc2, au2, vb2);
        acc3 = _mm512_dpbusd_epi32(acc3, au3, vb3);

        // Bias: promote vb to i16 then compute 128 * sum(vb) using VPDPWSSD.
        // Each 64-byte zmm of int8 is split into two 512-bit i16 vectors.
        __m512i vb0_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i +   0)));
        __m512i vb0_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i +  32)));
        __m512i vb1_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i +  64)));
        __m512i vb1_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i +  96)));
        __m512i vb2_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 128)));
        __m512i vb2_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 160)));
        __m512i vb3_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 192)));
        __m512i vb3_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 224)));

        bias0 = _mm512_dpwssd_epi32(bias0, vb0_lo, bias128);
        bias0 = _mm512_dpwssd_epi32(bias0, vb0_hi, bias128);
        bias1 = _mm512_dpwssd_epi32(bias1, vb1_lo, bias128);
        bias1 = _mm512_dpwssd_epi32(bias1, vb1_hi, bias128);
        bias2 = _mm512_dpwssd_epi32(bias2, vb2_lo, bias128);
        bias2 = _mm512_dpwssd_epi32(bias2, vb2_hi, bias128);
        bias3 = _mm512_dpwssd_epi32(bias3, vb3_lo, bias128);
        bias3 = _mm512_dpwssd_epi32(bias3, vb3_hi, bias128);
    }
    __m512i acc  = _mm512_add_epi32(_mm512_add_epi32(acc0,  acc1),
                                    _mm512_add_epi32(acc2,  acc3));
    __m512i bias = _mm512_add_epi32(_mm512_add_epi32(bias0, bias1),
                                    _mm512_add_epi32(bias2, bias3));

    // Single-zmm tail (residual 64-byte blocks).
    for (; i + 64 <= length; i += 64) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        __m512i au = _mm512_add_epi8(va, _mm512_set1_epi8(-128));
        acc  = _mm512_dpbusd_epi32(acc,  au, vb);
        
        // Promote vb to i16 for bias calculation
        __m512i vb_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
        __m512i vb_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32)));
        bias = _mm512_dpwssd_epi32(bias, vb_lo, bias128);
        bias = _mm512_dpwssd_epi32(bias, vb_hi, bias128);
    }

    int32_t result = hsum_epi32(acc) - hsum_epi32(bias);

    // Scalar tail.
    for (; i < length; i++)
        result += (int32_t)a[i] * (int32_t)b[i];

    return (float)result;
}

// ---------------------------------------------------------------------------
// euclidean_i8  —  VPMOVSXBW sign-extend + VPDPWSSD squared differences
// ---------------------------------------------------------------------------
//
// Each 64-byte zmm block is processed as two 32-byte halves:
//   _mm512_cvtepi8_epi16(__m256i) = VPMOVSXBW: sign-extends 32×i8 → 32×i16
//   diff = da - db  (i16 subtraction, no overflow since range is [-255,255])
//   acc  = VPDPWSSD(acc, diff, diff)
//
// 4× unrolled (256 bytes/iteration).
float euclidean_i8(const int8_t * __restrict__ a, size_t aoffset,
                   const int8_t * __restrict__ b, size_t boffset,
                   size_t length)
{
    a += aoffset;
    b += boffset;

    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 256 <= length; i += 256) {
#define EUCL_BLOCK(off, acc_var)                                                          \
        {                                                                                 \
            const int8_t *ap = a + i + (off), *bp = b + i + (off);                      \
            __m512i da_lo = _mm512_cvtepi8_epi16(                                        \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap)));       \
            __m512i db_lo = _mm512_cvtepi8_epi16(                                        \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp)));       \
            __m512i da_hi = _mm512_cvtepi8_epi16(                                        \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap + 32)));  \
            __m512i db_hi = _mm512_cvtepi8_epi16(                                        \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + 32)));  \
            __m512i diff_lo = _mm512_sub_epi16(da_lo, db_lo);                            \
            __m512i diff_hi = _mm512_sub_epi16(da_hi, db_hi);                            \
            acc_var = _mm512_dpwssd_epi32(acc_var, diff_lo, diff_lo);                    \
            acc_var = _mm512_dpwssd_epi32(acc_var, diff_hi, diff_hi);                    \
        }
        EUCL_BLOCK(  0, acc0)
        EUCL_BLOCK( 64, acc1)
        EUCL_BLOCK(128, acc2)
        EUCL_BLOCK(192, acc3)
#undef EUCL_BLOCK
    }
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                   _mm512_add_epi32(acc2, acc3));

    // Single 64-byte tail blocks.
    for (; i + 64 <= length; i += 64) {
        __m512i da_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
        __m512i db_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
        __m512i da_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32)));
        __m512i db_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32)));
        acc = _mm512_dpwssd_epi32(acc, _mm512_sub_epi16(da_lo, db_lo), _mm512_sub_epi16(da_lo, db_lo));
        acc = _mm512_dpwssd_epi32(acc, _mm512_sub_epi16(da_hi, db_hi), _mm512_sub_epi16(da_hi, db_hi));
    }

    // 32-byte tail (one ymm → one 512-bit i16 vector).
    if (i + 32 <= length) {
        __m512i da = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
        __m512i db = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
        __m512i diff = _mm512_sub_epi16(da, db);
        acc = _mm512_dpwssd_epi32(acc, diff, diff);
        i += 32;
    }

    int32_t result = hsum_epi32(acc);

    // Scalar tail.
    for (; i < length; i++) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        result += d * d;
    }
    return (float)result;
}

// ---------------------------------------------------------------------------
// cosine_i8  —  three parallel VPDPBUSD chains with bias correction
// ---------------------------------------------------------------------------
//
// Computes dot(a,b), ||a||², ||b||² in a single pass using VPDPBUSD.
// Bias trick: a_u = a+128 (unsigned), then subtract 128*sum(b) and 128*sum(a).
//
//   dot(a,b)  = hsum(VPDPBUSD(acc_dot,  a_u, b))  - 128*sum(b)
//   ||a||²    = hsum(VPDPBUSD(acc_normA, a_u, a))  - 128*sum(a)
//   ||b||²    = hsum(VPDPBUSD(acc_normB, b_u, b))  - 128*sum(b)
//
// normB reuses the same biasAB accumulator as dot (both need 128*sum(b)).
// 2× unrolled (128 bytes/iteration) with 6 VPDPBUSD + 4 VPDPWSSD per iter.
float cosine_i8(const int8_t * __restrict__ a, size_t aoffset,
                const int8_t * __restrict__ b, size_t boffset,
                size_t length)
{
    a += aoffset;
    b += boffset;

    const __m512i bias128 = _mm512_set1_epi16(128);
    const __m512i flip    = _mm512_set1_epi8(-128);

    __m512i dot0  = _mm512_setzero_si512(), dot1  = _mm512_setzero_si512();
    __m512i normA0 = _mm512_setzero_si512(), normA1 = _mm512_setzero_si512();
    __m512i normB0 = _mm512_setzero_si512(), normB1 = _mm512_setzero_si512();
    __m512i biasAB0 = _mm512_setzero_si512(), biasAB1 = _mm512_setzero_si512();
    __m512i biasA0  = _mm512_setzero_si512(), biasA1  = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 128 <= length; i += 128) {
        __m512i va0 = _mm512_loadu_si512(a + i);
        __m512i vb0 = _mm512_loadu_si512(b + i);
        __m512i va1 = _mm512_loadu_si512(a + i + 64);
        __m512i vb1 = _mm512_loadu_si512(b + i + 64);
        __m512i au0 = _mm512_add_epi8(va0, flip);
        __m512i bu0 = _mm512_add_epi8(vb0, flip);
        __m512i au1 = _mm512_add_epi8(va1, flip);
        __m512i bu1 = _mm512_add_epi8(vb1, flip);

        dot0  = _mm512_dpbusd_epi32(dot0,  au0, vb0);
        dot1  = _mm512_dpbusd_epi32(dot1,  au1, vb1);
        normA0 = _mm512_dpbusd_epi32(normA0, au0, va0);
        normA1 = _mm512_dpbusd_epi32(normA1, au1, va1);
        normB0 = _mm512_dpbusd_epi32(normB0, bu0, vb0);
        normB1 = _mm512_dpbusd_epi32(normB1, bu1, vb1);
        
        // Promote to i16 for bias calculations
        __m512i va0_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
        __m512i va0_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32)));
        __m512i vb0_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
        __m512i vb0_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32)));
        __m512i va1_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 64)));
        __m512i va1_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 96)));
        __m512i vb1_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 64)));
        __m512i vb1_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 96)));
        
        biasAB0 = _mm512_dpwssd_epi32(biasAB0, vb0_lo, bias128);
        biasAB0 = _mm512_dpwssd_epi32(biasAB0, vb0_hi, bias128);
        biasAB1 = _mm512_dpwssd_epi32(biasAB1, vb1_lo, bias128);
        biasAB1 = _mm512_dpwssd_epi32(biasAB1, vb1_hi, bias128);
        biasA0  = _mm512_dpwssd_epi32(biasA0,  va0_lo, bias128);
        biasA0  = _mm512_dpwssd_epi32(biasA0,  va0_hi, bias128);
        biasA1  = _mm512_dpwssd_epi32(biasA1,  va1_lo, bias128);
        biasA1  = _mm512_dpwssd_epi32(biasA1,  va1_hi, bias128);
    }
    __m512i dot    = _mm512_add_epi32(dot0,    dot1);
    __m512i normA  = _mm512_add_epi32(normA0,  normA1);
    __m512i normB  = _mm512_add_epi32(normB0,  normB1);
    __m512i biasAB = _mm512_add_epi32(biasAB0, biasAB1);
    __m512i biasA  = _mm512_add_epi32(biasA0,  biasA1);

    // Single-zmm tail.
    for (; i + 64 <= length; i += 64) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        __m512i au = _mm512_add_epi8(va, flip);
        __m512i bu = _mm512_add_epi8(vb, flip);
        dot    = _mm512_dpbusd_epi32(dot,   au, vb);
        normA  = _mm512_dpbusd_epi32(normA, au, va);
        normB  = _mm512_dpbusd_epi32(normB, bu, vb);
        
        // Promote to i16 for bias calculations
        __m512i va_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
        __m512i va_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32)));
        __m512i vb_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
        __m512i vb_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32)));
        
        biasAB = _mm512_dpwssd_epi32(biasAB, vb_lo, bias128);
        biasAB = _mm512_dpwssd_epi32(biasAB, vb_hi, bias128);
        biasA  = _mm512_dpwssd_epi32(biasA,  va_lo, bias128);
        biasA  = _mm512_dpwssd_epi32(biasA,  va_hi, bias128);
    }

    // Apply bias corrections before scalar tail.
    int64_t dotResult   = (int64_t)hsum_epi32(dot)   - (int64_t)hsum_epi32(biasAB);
    int64_t normAResult = (int64_t)hsum_epi32(normA) - (int64_t)hsum_epi32(biasA);
    int64_t normBResult = (int64_t)hsum_epi32(normB) - (int64_t)hsum_epi32(biasAB);

    // Scalar tail.
    for (; i < length; i++) {
        int32_t ai = a[i], bi = b[i];
        dotResult   += (int64_t)ai * bi;
        normAResult += (int64_t)ai * ai;
        normBResult += (int64_t)bi * bi;
    }

    return (float)(dotResult / sqrt((double)normAResult * (double)normBResult));
}

} // namespace AVX3_DL
