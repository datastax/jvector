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

#include <inttypes.h>
#include <float.h>
#include <math.h>
#include <cstring>
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assertHwyTargets.h"

// =============================================================================
// Highway macro usage in this file
// =============================================================================
//
// HWY_INLINE
//   Expands to `inline __attribute__((always_inline))` on GCC/Clang.
//   Used on every helper that participates in a hot SIMD loop to prevent the
//   compiler from ever emitting a real call and to keep register pressure
//   visible across the inlined body.  Prefer this over plain `inline` for any
//   function that contains SIMD intrinsics.
//
// HWY_FLATTEN
//   Expands to `__attribute__((flatten))`, which asks the compiler to inline
//   *all* callees into the annotated function.  Used on the public entry-points
//   (assemble_and_sum_f32, pq_decoded_cosine_similarity_f32, the three
//   distance wrappers) so that the multi-target Highway dispatch stub sees a
//   single monolithic body with no residual call overhead.
//
// HWY_RESTRICT
//   Portable spelling of `__restrict__` / `restrict`.  Tells the compiler that
//   a pointer does not alias any other pointer in scope, so loads through it
//   remain valid across stores made through a different pointer.  Applied to:
//     - Load helper parameters (e.g. LoadDup256) — ensures the loaded value is
//       treated as loop-invariant even when the caller stores to an accumulator.
//     - calculate_partial_sums_f32 inputs (codebook, query) and output
//       (partialSums) — prevents the compiler from reloading read-only inputs
//       after each write to partialSums.
//   Not needed when inputs and output already have different types (e.g.
//   float* vs unsigned char*), because C++ strict-aliasing rules already
//   guarantee they cannot alias.
//
// =============================================================================
// Highway API tutorial — intrinsics used in this file
// =============================================================================
//
// --- Tags (describe vector type and width) ---
//
// ScalableTag<T>
//   Represents the full native SIMD width for type T.
//   e.g. ScalableTag<float> is 8 lanes on AVX2, 16 lanes on AVX-512.
//   Used in the main loop bodies where we want the widest available vector.
//
// CappedTag<T, N>  /  HWY_CAPPED(T, N)
//   A tag capped to at most N lanes, even on wider ISAs.
//   Used in the small-vector fast paths (e.g. size==4, size==8) so that we
//   avoid wasting the extra lanes of a wide register on tiny inputs.
//
// Half<D>
//   Produces a tag whose lane count is half that of D.
//   Used in LoadDup256 to load 8 floats into the lower half of a 512-bit
//   register before Combine duplicates them into the upper half.
//
// Rebind<NewT, Tag>  /  RebindToSigned<Tag>
//   Produce a new tag of the same width but a different element type.
//   Used in assemble_and_sum_f32 and pq_decoded_cosine_similarity to
//   reinterpret the float-width register as uint8/uint16/int32 during the
//   index promotion pipeline.
//
// Lanes(tag)   — runtime lane count for the given tag.
// MaxLanes(tag) — compile-time upper bound on lane count (used in static_assert
//                 and constexpr branches).
//
// --- Vector type ---
//
// Vec<Tag>
//   The SIMD vector type corresponding to a tag.
//   All arithmetic and load/store operations return or accept Vec<Tag>.
//
// --- Initialisation ---
//
// Zero(tag)         — vector of all zeros; used to initialise accumulators.
// Set(tag, scalar)  — broadcast a scalar to every lane.
// Iota(tag, start)  — fill lanes with start, start+1, start+2, …
//                     Used to build the running index vector for GatherIndex.
//
// --- Loads ---
//
// LoadU(tag, ptr)        — unaligned load of Lanes(tag) elements from ptr.
// LoadN(tag, ptr, n)     — load n elements; remaining lanes are zero-padded.
//                          Used for loop tails without a branch per element.
// LoadDup128(tag, ptr)   — load 128 bits and broadcast across the full vector.
//                          Used for size==2 and size==4 query vectors so the
//                          same query chunk lines up with every centroid chunk.
// MaskedLoad(mask, tag, ptr)
//                        — load only the lanes where mask is set; others zero.
//                          Used in CosineDistance tail handling.
//
// --- Store ---
//
// StoreU(vec, tag, ptr)  — unaligned store of Lanes(tag) elements to ptr.
//
// --- Arithmetic ---
//
// Add(a, b)              — lane-wise addition.
// Sub(a, b)              — lane-wise subtraction.
// Mul(a, b)              — lane-wise multiplication.
// MulAdd(a, b, c)        — fused multiply-add: (a * b) + c.
//                          Preferred over separate Mul+Add for FMA throughput.
//
// --- Type promotion ---
//
// PromoteTo(narrower_tag, vec)
//   Zero-extends each element to the wider type.
//   Used twice in the gather pipeline: u8 → u16 → i32, so that byte offsets
//   become 32-bit gather indices without sign-extension artefacts.
//
// --- Gather ---
//
// GatherIndex(tag, base_ptr, index_vec)
//   Loads one element per lane using per-lane 32-bit indices (in elements,
//   not bytes).  Used to collect PQ lookup-table entries and codebook floats
//   whose positions are determined at runtime by the encoded offsets.
//
// --- Reductions ---
//
// ReduceSum(tag, vec)    — horizontal sum of all lanes; returns a scalar.
//
// --- Horizontal-reduction shuffles (used in calculate_partial_sums_f32) ---
//
// Shuffle2301(vec)
//   Swaps adjacent 32-bit lane pairs within each 64-bit group: [0,1,2,3] → [1,0,3,2].
//   (Name "2301" encodes source indices from lane 3 down to lane 0 per _MM_SHUFFLE
//   convention: lane3←2, lane2←3, lane1←0, lane0←1.)
//   Adding a vector with its Shuffle2301 partner sums adjacent lane pairs:
//   result[0]=result[1]=(0+1) and result[2]=result[3]=(2+3).
//
// Shuffle1032(vec)
//   Swaps the two 64-bit halves within each 128-bit block: [0,1,2,3] → [2,3,0,1].
//   Used as the second reduction step: adding those pairwise sums together
//   yields the full 4-lane horizontal sum in every lane.
//
// SwapAdjacentBlocks(vec)
//   Swaps the two 128-bit halves of each 256-bit block.
//   On AVX-512 (512-bit vector = four 128-bit blocks) this is used as the
//   first step of the size==8 horizontal reduction before Shuffle1032/2301.
//
// --- Masks ---
//
// FirstN(tag, n)
//   Returns a mask with the first n lanes set and the rest clear.
//   Used with MaskedLoad to handle the tail of a vector that doesn't fill
//   a full register.
//
// --- Combine ---
//
// Combine(d, hi, lo)
//   Concatenates two half-width vectors into one full-width vector.
//   Used in LoadDup256 to duplicate 8 floats across both halves of a
//   512-bit register (lo = hi = the same 256-bit load).
//

namespace hn = hwy::HWY_NAMESPACE;

// Loads 8 floats from ptr and broadcasts them to fill the full vector D.
// On ISAs where D is exactly 8 lanes (e.g. AVX2) this is a plain LoadU.
// On wider ISAs (e.g. AVX-512, 16 lanes) the 8 floats are loaded into the
// 256-bit half-tag and then Combine'd to duplicate them into both halves.
// NOTE: not designed for ISAs wider than 512-bit (would need additional Combine levels).
// HWY_RESTRICT tells the compiler that ptr does not alias any accumulator or
// other pointer visible at the call site, allowing it to treat all loads from
// ptr as invariant across iterations and hoist them freely.
template <class D>
HWY_INLINE hn::Vec<D> LoadDup256(D d, const float *HWY_RESTRICT ptr)
{
    static_assert(hn::MaxLanes(d) <= 16,
                  "LoadDup256 is not implemented for ISAs wider than 512-bit");
    if constexpr (hn::MaxLanes(d) > 8) {
        const hn::Half<D> dh;
        const auto half = hn::LoadU(dh, ptr);
        return hn::Combine(d, half, half);
    }
    else {
        return hn::LoadU(d, ptr);
    }
}
// =============================================================================
// Base Fp32 kernels
// =============================================================================

// Dot product kernel templated on the Highway tag type, shared by the
// full-width ScalableTag path and the HWY_CAPPED fast paths.
// The 4x unrolled loop hides FMA latency; for capped tags the loop body
// is never entered for small sizes and the single-vector path handles them.
template <class Tag>
HWY_INLINE float DotProductImpl(Tag tag, const float *a, const float *b, size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto acc0 = hn::Zero(tag), acc1 = hn::Zero(tag);
    auto acc2 = hn::Zero(tag), acc3 = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        acc0 = hn::MulAdd(hn::LoadU(tag, a + ii + 0*lanes), hn::LoadU(tag, b + ii + 0*lanes), acc0);
        acc1 = hn::MulAdd(hn::LoadU(tag, a + ii + 1*lanes), hn::LoadU(tag, b + ii + 1*lanes), acc1);
        acc2 = hn::MulAdd(hn::LoadU(tag, a + ii + 2*lanes), hn::LoadU(tag, b + ii + 2*lanes), acc2);
        acc3 = hn::MulAdd(hn::LoadU(tag, a + ii + 3*lanes), hn::LoadU(tag, b + ii + 3*lanes), acc3);
    }
    auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
    for (; ii + lanes <= size; ii += lanes) {
        acc = hn::MulAdd(hn::LoadU(tag, a + ii), hn::LoadU(tag, b + ii), acc);
    }
    if (ii < size) {
        acc = hn::MulAdd(hn::LoadN(tag, a + ii, size - ii),
                         hn::LoadN(tag, b + ii, size - ii), acc);
    }
    return hn::ReduceSum(tag, acc);
}

// L2 square distance kernel templated on the Highway tag type, shared by
// the full-width ScalableTag path and the HWY_CAPPED fast paths.
template <class Tag>
HWY_INLINE float L2SquareDistanceImpl(Tag tag, const float *a, const float *b, size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto acc0 = hn::Zero(tag), acc1 = hn::Zero(tag);
    auto acc2 = hn::Zero(tag), acc3 = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        auto d0 = hn::LoadU(tag, a + ii + 0*lanes) - hn::LoadU(tag, b + ii + 0*lanes);
        auto d1 = hn::LoadU(tag, a + ii + 1*lanes) - hn::LoadU(tag, b + ii + 1*lanes);
        auto d2 = hn::LoadU(tag, a + ii + 2*lanes) - hn::LoadU(tag, b + ii + 2*lanes);
        auto d3 = hn::LoadU(tag, a + ii + 3*lanes) - hn::LoadU(tag, b + ii + 3*lanes);
        acc0 = hn::MulAdd(d0, d0, acc0);
        acc1 = hn::MulAdd(d1, d1, acc1);
        acc2 = hn::MulAdd(d2, d2, acc2);
        acc3 = hn::MulAdd(d3, d3, acc3);
    }
    auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
    for (; ii + lanes <= size; ii += lanes) {
        auto d = hn::LoadU(tag, a + ii) - hn::LoadU(tag, b + ii);
        acc = hn::MulAdd(d, d, acc);
    }
    if (ii < size) {
        auto d = hn::LoadN(tag, a + ii, size - ii) - hn::LoadN(tag, b + ii, size - ii);
        acc = hn::MulAdd(d, d, acc);
    }
    return hn::ReduceSum(tag, acc);
}

// Cosine distance kernel templated on the Highway tag type, shared by
// the full-width ScalableTag path and the HWY_CAPPED fast paths.
template <class Tag>
HWY_INLINE float CosineDistanceImpl(Tag tag, const float *a, const float *b, size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto sum_ab = hn::Zero(tag), sum_aa = hn::Zero(tag), sum_bb = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + lanes <= size; ii += lanes) {
        auto va = hn::LoadU(tag, a + ii);
        auto vb = hn::LoadU(tag, b + ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    if (ii < size) {
        auto va = hn::LoadN(tag, a + ii, size - ii);
        auto vb = hn::LoadN(tag, b + ii, size - ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    return hn::ReduceSum(tag, sum_ab)
           / sqrtf(hn::ReduceSum(tag, sum_aa) * hn::ReduceSum(tag, sum_bb));
}

// Returns the dot product sum(a[ii] * b[ii]).
//
// Short-vector fast paths: when the register width is wider than the vector
// (e.g. a 4-element input on AVX-512), using the full register wastes lanes
// and can hurt latency.  Capped tags keep execution in narrow registers.
HWY_INLINE float DotProduct(const float *a,
                            size_t aoffset,
                            const float *b,
                            size_t boffset,
                            size_t length)
{
    a += aoffset;
    b += boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return DotProductImpl(HWY_CAPPED(float, 4){}, a, b, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return DotProductImpl(HWY_CAPPED(float, 8){}, a, b, length); }
#endif
#endif
    return DotProductImpl(hn::ScalableTag<float>{}, a, b, length);
}

HWY_INLINE float CosineDistance(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    const float *ap = a + aoffset;
    const float *bp = b + boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return CosineDistanceImpl(HWY_CAPPED(float, 4){}, ap, bp, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return CosineDistanceImpl(HWY_CAPPED(float, 8){}, ap, bp, length); }
#endif
#endif
    return CosineDistanceImpl(hn::ScalableTag<float>{}, ap, bp, length);
}

HWY_INLINE float L2SquareDistance(const float *a,
                                  size_t aoffset,
                                  const float *b,
                                  size_t boffset,
                                  size_t length)
{
    a += aoffset;
    b += boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return L2SquareDistanceImpl(HWY_CAPPED(float, 4){}, a, b, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return L2SquareDistanceImpl(HWY_CAPPED(float, 8){}, a, b, length); }
#endif
#endif
    return L2SquareDistanceImpl(hn::ScalableTag<float>{}, a, b, length);
}

namespace JV_ISA {

HWY_FLATTEN float cosine_f32(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return CosineDistance(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float dot_product_f32(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return DotProduct(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float euclidean_f32(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return L2SquareDistance(a, aoffset, b, boffset, length);
}

// =============================================================================
// Element-wise in-place arithmetic and reduction kernels
// =============================================================================
//
// rename-registers: extra GCC register-renaming pass that breaks false WAR/WAW
//   hazards between short-lived zmm values, enabling more ILP in the SIMD loops.
// #pragma GCC unroll 4: unroll by 4 to hide the 4-cycle FMA latency and keep
//   both AVX-512 FMA ports saturated across independent load–op–store chains.
//
__attribute__((optimize("rename-registers")))
HWY_FLATTEN void add_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Add(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Add(a, b), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void add_scalar_in_place_f32(float *HWY_RESTRICT v1,
                                          float value,
                                          size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto vval = hn::Set(d, value);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        hn::StoreU(hn::Add(a, vval), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        hn::StoreN(hn::Add(a, vval), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void sub_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Sub(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Sub(a, b), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void sub_scalar_in_place_f32(float *HWY_RESTRICT v1,
                                          float value,
                                          size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto vval = hn::Set(d, value);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        hn::StoreU(hn::Sub(a, vval), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        hn::StoreN(hn::Sub(a, vval), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN float max_f32(const float *HWY_RESTRICT v, size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto accum = hn::Set(d, -FLT_MAX);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        accum = hn::Max(accum, hn::LoadU(d, v + i));
    }
    float result = hn::ReduceMax(d, accum);
    for (; i < length; i++) {
        if (v[i] > result) result = v[i];
    }
    return result;
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void min_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Min(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Min(a, b), d, v1 + i, rem);
    }
}

// =============================================================================
// PQ kernels
// =============================================================================

enum class DistanceType { DotProduct, Euclidean };

// Computes the per-element score vector for a single SIMD register pair (c, qq).
// For DotProduct: score[ii] = c[ii] * qq[ii]
// For Euclidean:  score[ii] = (c[ii] - qq[ii])^2
template <DistanceType DT, class D>
HWY_INLINE hn::Vec<D> partial_sum_score(const hn::Vec<D> &c,
                                        const hn::Vec<D> &qq)
{
    if constexpr (DT == DistanceType::DotProduct) { return hn::Mul(c, qq); }
    else if constexpr (DT == DistanceType::Euclidean) {
        const hn::Vec<D> diff = hn::Sub(c, qq);
        return hn::Mul(diff, diff);
    }
    else {
        static_assert(DT == DistanceType::DotProduct
                              || DT == DistanceType::Euclidean,
                      "Unsupported DistanceType");
        // Unreachable, but silences compiler warnings about missing return.
        return hn::Zero(c);
    }
}

// Scalar fallback: returns dot_product_f32 or euclidean_f32 depending on DT.
template <DistanceType DT>
HWY_INLINE float distance_func(const float *codebook,
                               int clusterOffset,
                               const float *query,
                               int queryOffset,
                               size_t size)
{
    if constexpr (DT == DistanceType::DotProduct)
        return DotProduct(codebook, clusterOffset, query, queryOffset, size);
    else
        return L2SquareDistance(
                codebook, clusterOffset, query, queryOffset, size);
}

template <DistanceType DT>
// HWY_RESTRICT on codebook and query informs the compiler that writes to
// partialSums cannot alias either read-only input, so it need not reload
// codebook/query values after each store to partialSums.
HWY_INLINE void calculate_partial_sums_f32(const float *HWY_RESTRICT codebook,
                                           int codebookIndex,
                                           size_t size,
                                           int clusterCount,
                                           const float *HWY_RESTRICT query,
                                           int queryOffset,
                                           float *HWY_RESTRICT partialSums)
{
    int codebookBase = codebookIndex * clusterCount;
    using FloatTag = hn::ScalableTag<float>;
    FloatTag tag;
    constexpr size_t kLanes = hn::MaxLanes(tag);
    alignas(64) float tmp[kLanes];
    int ii = 0;

    if constexpr (kLanes >= 2) {
        if (size == 2) {
            float qtmp[4] = {query[queryOffset],
                             query[queryOffset + 1],
                             query[queryOffset],
                             query[queryOffset + 1]};
            hn::Vec<FloatTag> queryVec = hn::LoadDup128(tag, qtmp);

            constexpr size_t kBlock = 2;
            constexpr int centroids_per_iter = kLanes / kBlock;

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * 2;
                hn::Vec<FloatTag> centroidVec = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> score = partial_sum_score<DT, FloatTag>(
                        centroidVec, queryVec);
                hn::Vec<FloatTag> swapped = hn::Shuffle2301(score);
                hn::Vec<FloatTag> sum = score + swapped;
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 8
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 2];
                }
            }
        }
    }
    if constexpr (kLanes >= 4) {
        if (size == 4) {
            constexpr int centroids_per_iter = static_cast<int>(kLanes / 4);
            hn::Vec<FloatTag> queryVec
                    = hn::LoadDup128(tag, query + queryOffset);

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * size;
                hn::Vec<FloatTag> centroidVec = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> sum = partial_sum_score<DT, FloatTag>(
                        centroidVec, queryVec);
                hn::Vec<FloatTag> temp = hn::Shuffle2301(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle1032(sum);
                sum = hn::Add(sum, temp);
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 4
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 4];
                }
            }
        }
    }
    if constexpr (kLanes >= 8) {
        if (size == 8) {
            hn::Vec<FloatTag> queryVec = LoadDup256(tag, query + queryOffset);
            constexpr int centroids_per_iter = static_cast<int>(kLanes / 8);

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * size;
                hn::Vec<FloatTag> centroidVec = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> sum = partial_sum_score<DT, FloatTag>(
                        centroidVec, queryVec);
                hn::Vec<FloatTag> temp = hn::SwapAdjacentBlocks(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle1032(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle2301(sum);
                sum = hn::Add(sum, temp);
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 2
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 8];
                }
            }
        }
    }
    if constexpr (kLanes == 16) {
        // Don't have to worry about making this work on 1024-bit lanes just yet
        if (size == 16) {
            const hn::Vec<FloatTag> queryVec
                    = hn::LoadU(tag, query + queryOffset);
            for (; ii < clusterCount; ++ii) {
                const hn::Vec<FloatTag> centroidVec
                        = hn::LoadU(tag, codebook + ii * size);
                partialSums[codebookBase + ii] = hn::ReduceSum(
                        tag,
                        partial_sum_score<DT, FloatTag>(centroidVec, queryVec));
            }
        }
    }
    for (; ii < clusterCount; ii++) {
        partialSums[codebookBase + ii] = distance_func<DT>(
                codebook, ii * size, query, queryOffset, size);
    }
}

// HWY_RESTRICT is not needed here: `data` (float*) and `baseOffsets` (unsigned char*)
// have different types, so the compiler already treats them as non-aliasing under
// C++ strict-aliasing rules. Neither pointer is written through, so there are no
// stores that could force a reload of the other.

// Inner kernel templated on the Highway tag type so it works for both the
// full-width ScalableTag path and the capped HWY_CAPPED fast paths.
// gatherIndices[k] = (ii + k) * dataBase + baseOffsets[ii + k]
template <class FloatTag>
HWY_INLINE float AssembleAndSumImpl(
        const float         *HWY_RESTRICT data,
        int                  dataBase,
        const uint8_t       *HWY_RESTRICT baseOffsets,
        size_t               baseOffsetsLength,
        FloatTag             floatTag)
{
    const hn::RebindToSigned<FloatTag>                  int32Tag;
    const hn::Rebind<uint16_t, hn::RebindToSigned<FloatTag>> uint16Tag;
    const hn::Rebind<uint8_t,  hn::RebindToSigned<FloatTag>> uint8Tag;
    const size_t lanes = hn::Lanes(floatTag);

    // Precompute scaleVec = [0, db, 2*db, ..., (lanes-1)*db] once.
    // This eliminates a 512-bit VPMULLD on every iteration; the per-iteration
    // base (ii*dataBase) is a scalar that the compiler strength-reduces to an add.
    const auto scaleVec = hn::Mul(hn::Iota(int32Tag, 0),
                                  hn::Set(int32Tag, dataBase));
    auto sumVec = hn::Zero(floatTag);

    size_t ii = 0;
    for (; ii + lanes <= baseOffsetsLength; ii += lanes) {
        // Load `lanes` bytes and zero-extend to i32 via u8→u16→i32.
        const auto offsetVec = hn::PromoteTo(int32Tag,
                                   hn::PromoteTo(uint16Tag,
                                       hn::LoadU(uint8Tag, baseOffsets + ii)));
        const auto base = hn::Set(int32Tag,
                                  static_cast<int32_t>(ii * static_cast<size_t>(dataBase)));
        sumVec = hn::Add(sumVec, hn::GatherIndex(floatTag, data,
                             hn::Add(hn::Add(base, scaleVec), offsetVec)));
    }

    float res = hn::ReduceSum(floatTag, sumVec);
    for (; ii < baseOffsetsLength; ii++) {
        res += data[dataBase * ii + baseOffsets[ii]];
    }
    return res;
}

HWY_FLATTEN float assemble_and_sum_f32(const float *data,
                                           int dataBase,
                                           const unsigned char *baseOffsets,
                                           int baseOffsetsOffset,
                                           size_t baseOffsetsLength)
{
    baseOffsets += baseOffsetsOffset;

#if HWY_MAX_BYTES > 16
    if (baseOffsetsLength <= 4) {
        return AssembleAndSumImpl(data, dataBase, baseOffsets, baseOffsetsLength,
                                  HWY_CAPPED(float, 4){});
    }
#if HWY_MAX_BYTES > 32
    if (baseOffsetsLength <= 8) {
        return AssembleAndSumImpl(data, dataBase, baseOffsets, baseOffsetsLength,
                                  HWY_CAPPED(float, 8){});
    }
#endif
#endif

    return AssembleAndSumImpl(data, dataBase, baseOffsets, baseOffsetsLength,
                              hn::ScalableTag<float>{});
}

// Inner kernel for the triangular-table PQ gather, templated on the Highway tag
// type so it works for both the full-width ScalableTag path and HWY_CAPPED fast paths.
// gatherIndex[j] = laneIdx[j]*blockSize + triangularIndex(c1[j], c2[j])
template <class FloatTag>
HWY_INLINE float AssembleAndSumPQImpl(
        const float    *HWY_RESTRICT data,
        size_t          subspaceCount,
        const uint8_t  *HWY_RESTRICT baseOffsets1,
        const uint8_t  *HWY_RESTRICT baseOffsets2,
        int             k,
        int             blockSize,
        FloatTag        d_f)
{
    const hn::RebindToSigned<FloatTag>                       d_i;
    const hn::Rebind<uint16_t, hn::RebindToSigned<FloatTag>> d_u16;
    const hn::Rebind<uint8_t,  hn::RebindToSigned<FloatTag>> d_u8;
    const size_t lanes = hn::Lanes(d_f);

    const auto vk         = hn::Set(d_i, k);
    const auto vBlockSize = hn::Set(d_i, blockSize);
    auto sumVec  = hn::Zero(d_f);

    // Precompute laneIdxScaled = [0, bs, 2*bs, ..., (lanes-1)*bs] and advance
    // it by a fixed increment each iteration.  Replaces Mul(laneIdx, vBlockSize)
    // (VPMULLD, 3-cycle latency on critical path) with VPADDD (1-cycle latency).
    auto laneIdxScaled = hn::Mul(hn::Iota(d_i, 0), vBlockSize);
    const auto laneScaledInc = hn::Set(d_i, static_cast<int32_t>(lanes) * blockSize);

    size_t ii = 0;
    for (; ii + lanes <= subspaceCount; ii += lanes) {
        // Load `lanes` u8 ordinals and zero-extend to i32 via u8→u16→i32.
        const auto c1 = hn::PromoteTo(d_i, hn::PromoteTo(d_u16, hn::LoadU(d_u8, baseOffsets1 + ii)));
        const auto c2 = hn::PromoteTo(d_i, hn::PromoteTo(d_u16, hn::LoadU(d_u8, baseOffsets2 + ii)));
        const auto r  = hn::Min(c1, c2);
        const auto c  = hn::Max(c1, c2);
        // triangular = r*(r-1)/2; always even & non-negative so ShiftRight<1> is exact.
        const auto triangular = hn::ShiftRight<1>(hn::Mul(r, hn::Sub(r, hn::Set(d_i, 1))));
        const auto offsetRow  = hn::Sub(hn::Mul(r, vk), triangular);
        // gatherIndex = laneIdxScaled + offsetRow + (c - r)
        const auto gatherIdx = hn::Add(laneIdxScaled,
                                       hn::Add(offsetRow, hn::Sub(c, r)));
        sumVec       = hn::Add(sumVec, hn::GatherIndex(d_f, data, gatherIdx));
        laneIdxScaled = hn::Add(laneIdxScaled, laneScaledInc);
    }

    float res = hn::ReduceSum(d_f, sumVec);
    for (; ii < subspaceCount; ii++) {
        int c1v = baseOffsets1[ii], c2v = baseOffsets2[ii];
        int r   = c1v < c2v ? c1v : c2v;
        int cv  = c1v > c2v ? c1v : c2v;
        res += data[ii * blockSize + r * k - r * (r - 1) / 2 + (cv - r)];
    }
    return res;
}

// For each of the M subspaces, looks up data[i*blockSize + triangularIndex(c1[i], c2[i])]
// where blockSize = k*(k+1)/2 and triangularIndex(r,c) = r*k - r*(r-1)/2 + (c-r),
// r = min(c1,c2), c = max(c1,c2).  Vectorised via a gather over i32 indices built
// from integer min/max and an arithmetic right-shift for the triangular number.
// On ISAs with >128-bit registers, capped fast-paths are used when subspaceCount
// fits in 4 or 8 lanes to avoid wasting the extra lanes of a wide register.
HWY_FLATTEN float assemble_and_sum_pq_f32(
        const float    *HWY_RESTRICT data,
        size_t          subspaceCount,
        const uint8_t  *HWY_RESTRICT baseOffsets1, int baseOffsetsOffset1,
        const uint8_t  *HWY_RESTRICT baseOffsets2, int baseOffsetsOffset2,
        int             clusterCount)
{
    baseOffsets1 += baseOffsetsOffset1;
    baseOffsets2 += baseOffsetsOffset2;

    const int k         = clusterCount;
    const int blockSize = k * (k + 1) / 2;

#if HWY_MAX_BYTES > 16
    if (subspaceCount <= 4) {
        return AssembleAndSumPQImpl(data, subspaceCount, baseOffsets1, baseOffsets2,
                                    k, blockSize, HWY_CAPPED(float, 4){});
    }
#if HWY_MAX_BYTES > 32
    if (subspaceCount <= 8) {
        return AssembleAndSumPQImpl(data, subspaceCount, baseOffsets1, baseOffsets2,
                                    k, blockSize, HWY_CAPPED(float, 8){});
    }
#endif
#endif

    return AssembleAndSumPQImpl(data, subspaceCount, baseOffsets1, baseOffsets2,
                                k, blockSize, hn::ScalableTag<float>{});
}

// HWY_RESTRICT is not needed here: `baseOffsets` (unsigned char*) is a different type
// from `partialSums` and `aMagnitude` (float*), so strict-aliasing already guarantees
// the compiler that writes through one cannot affect loads from the other. All three
// pointers are read-only within the loop, so there are no stores to reason about anyway.
HWY_FLATTEN float
pq_decoded_cosine_similarity_f32(const unsigned char *baseOffsets,
                                     int baseOffsetsOffset,
                                     size_t baseOffsetsLength,
                                     int clusterCount,
                                     const float *partialSums,
                                     const float *aMagnitude,
                                     float bMagnitude)
{
    using FloatTag = hn::ScalableTag<float>;
    using Int32Tag = hn::RebindToSigned<FloatTag>;
    using Uint16Tag = hn::Rebind<uint16_t, Int32Tag>;
    using Uint8Tag = hn::Rebind<uint8_t, Int32Tag>;

    const FloatTag floatTag;
    const Int32Tag int32Tag;
    const Uint16Tag uint16Tag;
    const Uint8Tag uint8Tag;
    const size_t kLanes = hn::Lanes(floatTag);

    baseOffsets += baseOffsetsOffset;

    auto sumVec = hn::Zero(floatTag);
    auto magnitudeVec = hn::Zero(floatTag);

    // Precompute scaleVec = [0, cc, 2*cc, ..., (kLanes-1)*cc] once.
    // Eliminates a 512-bit VPMULLD on every iteration; the per-iteration scalar
    // base (ii*clusterCount) is strength-reduced to an add by the compiler.
    const auto scaleVec = hn::Mul(hn::Iota(int32Tag, 0),
                                  hn::Set(int32Tag, clusterCount));

    size_t ii = 0;
    for (; ii + kLanes <= baseOffsetsLength; ii += kLanes) {
        // Load kLanes bytes and zero-extend to int32 via two PromoteTo steps (u8→u16→i32)
        const auto u8Vec     = hn::LoadU(uint8Tag, baseOffsets + ii);
        const auto u16Vec    = hn::PromoteTo(uint16Tag, u8Vec);
        const auto offsetVec = hn::PromoteTo(int32Tag, u16Vec);

        // gatherIndices[k] = (ii + k) * clusterCount + baseOffsets[ii + k]
        const auto base = hn::Set(int32Tag,
                                  static_cast<int32_t>(ii * static_cast<size_t>(clusterCount)));
        const auto gatherIndices = hn::Add(hn::Add(base, scaleVec), offsetVec);

        sumVec       = hn::Add(sumVec,       hn::GatherIndex(floatTag, partialSums, gatherIndices));
        magnitudeVec = hn::Add(magnitudeVec, hn::GatherIndex(floatTag, aMagnitude,  gatherIndices));
    }

    float sumResult = hn::ReduceSum(floatTag, sumVec);
    float aMagnitudeResult = hn::ReduceSum(floatTag, magnitudeVec);

    // Handle the remaining elements
    for (; ii < baseOffsetsLength; ii++) {
        int offset = clusterCount * static_cast<int>(ii) + baseOffsets[ii];
        sumResult += partialSums[offset];
        aMagnitudeResult += aMagnitude[offset];
    }

    return sumResult / sqrtf(aMagnitudeResult * bMagnitude);
}

HWY_FLATTEN void calculate_partial_sums_dot_f32(const float *codebook,
                                                    int codebookIndex,
                                                    size_t size,
                                                    int clusterCount,
                                                    const float *query,
                                                    int queryOffset,
                                                    float *partialSums)
{
    calculate_partial_sums_f32<DistanceType::DotProduct>(codebook,
                                                         codebookIndex,
                                                         size,
                                                         clusterCount,
                                                         query,
                                                         queryOffset,
                                                         partialSums);
}

HWY_FLATTEN void calculate_partial_sums_euclidean_f32(const float *codebook,
                                                          int codebookIndex,
                                                          size_t size,
                                                          int clusterCount,
                                                          const float *query,
                                                          int queryOffset,
                                                          float *partialSums)
{
    calculate_partial_sums_f32<DistanceType::Euclidean>(codebook,
                                                        codebookIndex,
                                                        size,
                                                        clusterCount,
                                                        query,
                                                        queryOffset,
                                                        partialSums);
}

// Computes partialSums[codebookIndex * clusterCount + i] = dot(centroid_i, centroid_i)
// i.e. the squared L2 norm of each codebook centroid. This is the per-subspace
// contribution to aMagnitude in the PQ cosine similarity formula.
//
// Fast paths for size == 2/4/8/16 pack multiple centroids into one SIMD register
// (kLanes/size per iteration), eliminating the per-centroid ReduceSum overhead.
// They mirror the structure of calculate_partial_sums_f32 but replace the
// centroid×query multiply with centroid×centroid (self-square).
HWY_FLATTEN void calculate_partial_sums_self_magnitude_f32(
        const float *HWY_RESTRICT codebook,
        int codebookIndex,
        size_t size,
        int clusterCount,
        float *HWY_RESTRICT partialSums)
{
    const int codebookBase = codebookIndex * clusterCount;
    using FloatTag = hn::ScalableTag<float>;
    FloatTag tag;
    constexpr size_t kLanes = hn::MaxLanes(tag);
    alignas(64) float tmp[kLanes];
    int ii = 0;

    if constexpr (kLanes >= 2) {
        if (size == 2) {
            constexpr size_t kBlock = 2;
            constexpr int centroids_per_iter = kLanes / kBlock;

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * 2;
                const hn::Vec<FloatTag> cv = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> sum = hn::Mul(cv, cv);
                hn::Vec<FloatTag> swapped = hn::Shuffle2301(sum);
                sum = hn::Add(sum, swapped);
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 8
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 2];
                }
            }
        }
    }
    if constexpr (kLanes >= 4) {
        if (size == 4) {
            constexpr int centroids_per_iter = static_cast<int>(kLanes / 4);

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * size;
                const hn::Vec<FloatTag> cv = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> sum = hn::Mul(cv, cv);
                hn::Vec<FloatTag> temp = hn::Shuffle2301(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle1032(sum);
                sum = hn::Add(sum, temp);
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 4
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 4];
                }
            }
        }
    }
    if constexpr (kLanes >= 8) {
        if (size == 8) {
            constexpr int centroids_per_iter = static_cast<int>(kLanes / 8);

            for (; ii + centroids_per_iter <= clusterCount;
                 ii += centroids_per_iter) {
                const float *cptr = codebook + ii * size;
                const hn::Vec<FloatTag> cv = hn::LoadU(tag, cptr);
                hn::Vec<FloatTag> sum = hn::Mul(cv, cv);
                hn::Vec<FloatTag> temp = hn::SwapAdjacentBlocks(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle1032(sum);
                sum = hn::Add(sum, temp);
                temp = hn::Shuffle2301(sum);
                sum = hn::Add(sum, temp);
                hn::StoreU(sum, tag, tmp);
#pragma GCC unroll 2
                for (int jj = 0; jj < centroids_per_iter; ++jj) {
                    partialSums[codebookBase + ii + jj] = tmp[jj * 8];
                }
            }
        }
    }
    if constexpr (kLanes == 16) {
        // AVX-512 only: one full register holds exactly one size==16 centroid.
        if (size == 16) {
            for (; ii < clusterCount; ++ii) {
                const hn::Vec<FloatTag> cv
                        = hn::LoadU(tag, codebook + ii * size);
                partialSums[codebookBase + ii]
                        = hn::ReduceSum(tag, hn::Mul(cv, cv));
            }
        }
    }
    // General fallback: one centroid at a time, vector-accumulate then reduce.
    for (; ii < clusterCount; ii++) {
        const float *cptr = codebook + ii * size;
        auto accVec = hn::Zero(tag);
        size_t j = 0;
        for (; j + kLanes <= size; j += kLanes) {
            const auto v = hn::LoadU(tag, cptr + j);
            accVec = hn::MulAdd(v, v, accVec);
        }
        float sum = hn::ReduceSum(tag, accVec);
        for (; j < size; j++) {
            sum += cptr[j] * cptr[j];
        }
        partialSums[codebookBase + ii] = sum;
    }
}

// =============================================================================
// NVQ kernels
// =============================================================================
//
// Bit-manipulation helpers used by all NVQ public kernels:
//
// logisticNQT — approximate sigmoid via IEEE 754 bit tricks (2^x approximation).
// logitNQT    — inverse: fast log2 via exponent extraction.
// Both exploit the float bit layout to avoid transcendental instructions.
//

// logisticNQT: approximate sigmoid using integer bit manipulation.
// Computes an approximation of the logistic function:
//   result ≈ 1 / (1 + exp(-alpha * (v - x0)))
// using the identity sigmoid(x) = 2^x / (2^x + 1) and a fast bit-hack for 2^x:
//   given x = p + f where p is integer and f ∈ [0,1):
//     2^x ≈ reinterpret_float(bits_of((f*0.5+1.0) << 23  +  p << 23))
template <class D>
HWY_INLINE hn::Vec<D> logisticNQT(D d, hn::Vec<D> v, float alpha, float x0)
{
    const hn::RebindToSigned<D> di;

    // temp = alpha * v - alpha * x0
    auto temp = hn::MulAdd(v, hn::Set(d, alpha), hn::Set(d, -alpha * x0));

    // p = (int)(temp + 1) where temp >= 0, else (int)(temp)
    // Mirrors Java: p = (int) floor(temp + 1); truncation == floor for temp >= 0.
    const auto isPositive = hn::Not(hn::IsNegative(temp));
    auto selected = hn::IfThenElse(isPositive,
                                   hn::Add(temp, hn::Set(d, 1.0f)),
                                   temp);
    auto p = hn::ConvertTo(di, selected);  // truncate towards zero

    // e = (float) p
    auto e = hn::ConvertTo(d, p);

    // m = reinterpret_bits((temp - e) * 0.5 + 1.0)
    // (temp - e) is in (-1, 1), so the result is in (0.5, 1.5) — a mantissa value.
    auto m = hn::BitCast(di,
                         hn::MulAdd(hn::Sub(temp, e),
                                    hn::Set(d, 0.5f),
                                    hn::Set(d, 1.0f)));

    // Reconstruct: (m_bits + (p << 23)) reinterpreted as float  =  m_mantissa * 2^p
    auto result = hn::BitCast(d, hn::Add(m, hn::ShiftLeft<23>(p)));

    // Sigmoid: result / (result + 1)
    return hn::Div(result, hn::Add(result, hn::Set(d, 1.0f)));
}

// logitNQT: inverse of logisticNQT — fast log2 via IEEE 754 exponent extraction.
// Computes approximately:
//   inverseAlpha * (log2(v / (1-v)) - 1) + x0
// The "-1" offset comes from subtracting 128 instead of 127 from the biased exponent,
// matching the Java implementation exactly.
template <class D>
HWY_INLINE hn::Vec<D> logitNQT(D d, hn::Vec<D> v, float inverseAlpha, float x0)
{
    const hn::RebindToSigned<D> di;

    // z = v / (1 - v)
    auto z = hn::Div(v, hn::Sub(hn::Set(d, 1.0f), v));

    // Reinterpret float bits as int32 to extract exponent and mantissa fields.
    auto temp = hn::BitCast(di, z);

    // p = (biased_exponent >> 23) - 128
    // Masking with 0x7f800000 isolates the 8 exponent bits; shifting by 23 places
    // them in the low byte. Subtracting 128 (vs. the standard 127 bias) is intentional
    // and matches the Java source.
    auto p = hn::Sub(hn::ShiftRight<23>(hn::And(temp, hn::Set(di, 0x7f800000))),
                     hn::Set(di, 128));

    // m = reinterpret as float: set exponent to 127 (i.e., 2^0) and keep mantissa
    // → value in [1.0, 2.0)
    auto m = hn::BitCast(d, hn::Add(hn::And(temp, hn::Set(di, 0x007fffff)),
                                    hn::Set(di, 0x3f800000)));

    // return (m + (float)p) * inverseAlpha + x0
    return hn::MulAdd(hn::Add(m, hn::ConvertTo(d, p)),
                      hn::Set(d, inverseAlpha),
                      hn::Set(d, x0));
}

// Single-element wrappers used only for the two setup constants (logisticBias,
// logisticScale) computed at the start of each NVQ kernel.  They delegate to
// the vector templates via CappedTag<float,1> so there is no duplication of
// the bit-manipulation logic.  They are NOT called in any hot loop — all tail
// elements are handled by LoadN + FirstN-masked vector operations below.
static HWY_INLINE float logisticNQT_scalar(float value, float alpha, float x0)
{
    const hn::CappedTag<float, 1> d1;
    return hn::GetLane(logisticNQT(d1, hn::Set(d1, value), alpha, x0));
}

// Public kernels — called from Java via FFI.
//
// All six functions mirror the @Override methods in PanamaVectorUtilSupport.
// They share the same mathematical logic; the Highway vector loops replace the
// Panama FloatVector loops and the scalar tails are identical to the Java ones.
//
// Byte↔float conversion pipeline (mirrors Java nvqDequantize8bit):
//   LoadU(uint8Tag, ptr)   — fill 4N-lane u8 vector from N bytes at ptr
//   PromoteTo(uint16Tag)   — lower N u8s → N u16s (2N-lane vector)
//   PromoteTo(int32Tag)    — lower N u16s → N i32s (N-lane vector)
//   ConvertTo(floatTag)    — i32 → float
//   MulAdd(scale, bias)    — byte * logisticScale + logisticBias
//   logitNQT(...)          — inverse logistic
//
// Float→byte pipeline (nvq_quantize_8bit):
//   logisticNQT(...)       — forward logistic
//   scale and shift
//   ConvertTo(int32) after +0.5  — round toward nearest for positive values
//   StoreU to tmp[] + scalar clamp and byte-cast
//
// Cosine packing: nvq_cosine_8bit_packed returns an int64_t whose low 32 bits
// are the IEEE-754 bits of `sum` and whose high 32 bits are `bMagnitude`, so
// the caller can unpack with Float.intBitsToFloat without any heap allocation.
// =============================================================================

HWY_FLATTEN void nvq_quantize_8bit(const float *HWY_RESTRICT vector,
                                   size_t length,
                                   float alpha, float x0,
                                   float minValue, float maxValue,
                                   uint8_t *HWY_RESTRICT destination)
{
    using FloatTag = hn::ScalableTag<float>;
    using Int32Tag = hn::RebindToSigned<FloatTag>;
    FloatTag d_f;
    Int32Tag d_i;
    constexpr size_t kLanes = hn::MaxLanes(d_f);
    alignas(64) int32_t tmp[kLanes];

    float delta            = maxValue - minValue;
    float scaledAlpha      = alpha / delta;
    float scaledX0         = x0 * delta;
    float logisticBias     = logisticNQT_scalar(minValue, scaledAlpha, scaledX0);
    float invLogisticScale = 255.0f / (logisticNQT_scalar(maxValue, scaledAlpha, scaledX0) - logisticBias);

    size_t i = 0;
    for (; i + kLanes <= length; i += kLanes) {
        auto arr = hn::LoadU(d_f, vector + i);
        arr = logisticNQT(d_f, arr, scaledAlpha, scaledX0);
        arr = hn::Add(hn::Mul(hn::Sub(arr, hn::Set(d_f, logisticBias)),
                              hn::Set(d_f, invLogisticScale)),
                      hn::Set(d_f, 0.5f));
        auto fi = hn::ConvertTo(d_i, arr);
        hn::StoreU(fi, d_i, tmp);
        for (size_t j = 0; j < kLanes; j++) {
            int v = tmp[j];
            destination[i + j] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    // Tail: LoadN zero-pads lanes beyond `remaining`; only write the first
    // `remaining` bytes from tmp[] so the padding lanes are never observed.
    const size_t remaining = length - i;
    if (remaining > 0) {
        auto arr = hn::LoadN(d_f, vector + i, remaining);
        arr = logisticNQT(d_f, arr, scaledAlpha, scaledX0);
        arr = hn::Add(hn::Mul(hn::Sub(arr, hn::Set(d_f, logisticBias)),
                              hn::Set(d_f, invLogisticScale)),
                      hn::Set(d_f, 0.5f));
        hn::StoreU(hn::ConvertTo(d_i, arr), d_i, tmp);
        for (size_t j = 0; j < remaining; j++) {
            int v = tmp[j];
            destination[i + j] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}

HWY_FLATTEN float nvq_loss(const float *HWY_RESTRICT vector,
                           size_t length,
                           float alpha, float x0,
                           float minValue, float maxValue,
                           int nBits)
{
    using FloatTag = hn::ScalableTag<float>;
    using Int32Tag = hn::RebindToSigned<FloatTag>;
    FloatTag d_f;
    Int32Tag d_i;
    constexpr size_t kLanes = hn::MaxLanes(d_f);

    int   constant        = (1 << nBits) - 1;
    float delta           = maxValue - minValue;
    float scaledAlpha     = alpha / delta;
    float invScaledAlpha  = delta / alpha;   // 1 / scaledAlpha
    float scaledX0        = x0 * delta;
    float logisticBias    = logisticNQT_scalar(minValue, scaledAlpha, scaledX0);
    float logisticScale   = (logisticNQT_scalar(maxValue, scaledAlpha, scaledX0) - logisticBias) / (float)constant;
    float invLogisticScale = 1.0f / logisticScale;

    auto squaredSum = hn::Zero(d_f);

    size_t i = 0;
    for (; i + kLanes <= length; i += kLanes) {
        auto arr    = hn::LoadU(d_f, vector + i);
        auto recArr = logisticNQT(d_f, arr, scaledAlpha, scaledX0);
        recArr = hn::Mul(hn::Sub(recArr, hn::Set(d_f, logisticBias)),
                         hn::Set(d_f, invLogisticScale));
        // Round to nearest integer (add 0.5, truncate toward zero)
        auto recInt = hn::ConvertTo(d_i, hn::Add(recArr, hn::Set(d_f, 0.5f)));
        recArr = hn::ConvertTo(d_f, recInt);
        recArr = hn::MulAdd(recArr, hn::Set(d_f, logisticScale), hn::Set(d_f, logisticBias));
        recArr = logitNQT(d_f, recArr, invScaledAlpha, scaledX0);
        auto diff = hn::Sub(arr, recArr);
        squaredSum = hn::MulAdd(diff, diff, squaredSum);
    }

    float result = hn::ReduceSum(d_f, squaredSum);

    // Tail: LoadN zero-pads; mask the diff so padding lanes don't contribute.
    const size_t remaining = length - i;
    if (remaining > 0) {
        const auto mask = hn::FirstN(d_f, remaining);
        auto arr    = hn::LoadN(d_f, vector + i, remaining);
        auto recArr = logisticNQT(d_f, arr, scaledAlpha, scaledX0);
        recArr = hn::Mul(hn::Sub(recArr, hn::Set(d_f, logisticBias)),
                         hn::Set(d_f, invLogisticScale));
        auto recInt = hn::ConvertTo(d_i, hn::Add(recArr, hn::Set(d_f, 0.5f)));
        recArr = hn::ConvertTo(d_f, recInt);
        recArr = hn::MulAdd(recArr, hn::Set(d_f, logisticScale), hn::Set(d_f, logisticBias));
        recArr = logitNQT(d_f, recArr, invScaledAlpha, scaledX0);
        auto diff = hn::IfThenElseZero(mask, hn::Sub(arr, recArr));
        result += hn::ReduceSum(d_f, hn::Mul(diff, diff));
    }

    return result;
}

HWY_FLATTEN float nvq_uniform_loss(const float *HWY_RESTRICT vector,
                                   size_t length,
                                   float minValue, float maxValue,
                                   int nBits)
{
    using FloatTag = hn::ScalableTag<float>;
    using Int32Tag = hn::RebindToSigned<FloatTag>;
    FloatTag d_f;
    Int32Tag d_i;
    constexpr size_t kLanes = hn::MaxLanes(d_f);

    float constant = (float)((1 << nBits) - 1);
    float delta    = maxValue - minValue;

    auto squaredSum = hn::Zero(d_f);

    size_t i = 0;
    for (; i + kLanes <= length; i += kLanes) {
        auto arr    = hn::LoadU(d_f, vector + i);
        auto recArr = hn::Mul(hn::Sub(arr, hn::Set(d_f, minValue)),
                              hn::Set(d_f, constant / delta));
        auto recInt = hn::ConvertTo(d_i, hn::Add(recArr, hn::Set(d_f, 0.5f)));
        recArr = hn::ConvertTo(d_f, recInt);
        recArr = hn::MulAdd(recArr, hn::Set(d_f, delta / constant), hn::Set(d_f, minValue));
        auto diff = hn::Sub(arr, recArr);
        squaredSum = hn::MulAdd(diff, diff, squaredSum);
    }

    float result = hn::ReduceSum(d_f, squaredSum);

    // Tail: LoadN zero-pads; mask the diff so padding lanes don't contribute.
    const size_t remaining = length - i;
    if (remaining > 0) {
        const auto mask = hn::FirstN(d_f, remaining);
        auto arr    = hn::LoadN(d_f, vector + i, remaining);
        auto recArr = hn::Mul(hn::Sub(arr, hn::Set(d_f, minValue)),
                              hn::Set(d_f, constant / delta));
        auto recInt = hn::ConvertTo(d_i, hn::Add(recArr, hn::Set(d_f, 0.5f)));
        recArr = hn::ConvertTo(d_f, recInt);
        recArr = hn::MulAdd(recArr, hn::Set(d_f, delta / constant), hn::Set(d_f, minValue));
        auto diff = hn::IfThenElseZero(mask, hn::Sub(arr, recArr));
        result += hn::ReduceSum(d_f, hn::Mul(diff, diff));
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Two dequantization helpers — used by the three NVQ 8-bit scoring kernels.
//
// dequantize_bytes  (tail / non-hot path)
//   Loads kLanes uint8 values via a narrow u8 tag, widens to int32 through
//   two PromoteTo calls, converts to float, then applies scale+bias+logitNQT.
//   Used only for the unaligned tail elements after the main FastLanes loop;
//   those elements' query floats are in their original (un-shuffled) order.
//
// dequantize_bytes_fastlanes  (hot path)
//   Accepts a Vec<Int32Tag> that is a full-width byte vector BitCast-ed to
//   kLanes int32 lanes (i.e., 4*kLanes bytes loaded in one 512-bit register).
//   Extracts the `part`-th byte from each int32 lane via shift-right + AND,
//   converts to float, and applies scale+bias+logitNQT.  All operations run
//   at the native SIMD width — no widening, no register-width changes.
//   Mirrors the Panama FastLanes strategy:
//   https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf
// ─────────────────────────────────────────────────────────────────────────────
template <class FloatTag,
          class Int32Tag  = hn::RebindToSigned<FloatTag>,
          class Uint16Tag = hn::Rebind<uint16_t, Int32Tag>,
          class Uint8Tag  = hn::Rebind<uint8_t,  Int32Tag>>
HWY_INLINE hn::Vec<FloatTag>
dequantize_bytes(FloatTag d_f, Int32Tag d_i, Uint16Tag d_u16, Uint8Tag d_u8,
                 const uint8_t *HWY_RESTRICT quantized, size_t i,
                 float logisticScale, float logisticBias,
                 float invScaledAlpha, float scaledX0)
{
    const auto b_u8  = hn::LoadU(d_u8,  quantized + i);
    const auto b_u16 = hn::PromoteTo(d_u16, b_u8);
    const auto b_i32 = hn::PromoteTo(d_i,   b_u16);
    auto vb = hn::ConvertTo(d_f, b_i32);
    vb = hn::MulAdd(vb, hn::Set(d_f, logisticScale), hn::Set(d_f, logisticBias));
    return logitNQT(d_f, vb, invScaledAlpha, scaledX0);
}

template <class FloatTag,
          class Int32Tag = hn::RebindToSigned<FloatTag>>
HWY_INLINE hn::Vec<FloatTag>
dequantize_bytes_fastlanes(FloatTag d_f, Int32Tag d_i,
                           hn::Vec<Int32Tag> as_ints, int part,
                           float logisticScale, float logisticBias,
                           float invScaledAlpha, float scaledX0)
{
    // Extract the `part`-th byte from each int32 lane, then convert to float.
    // ShiftRightSame on a signed tag does arithmetic shift, but the AND with
    // 0xFF zeroes the sign-extended upper bits, giving an unsigned 0-255 value.
    auto shifted = hn::ShiftRightSame(as_ints, 8 * part);
    auto masked  = hn::And(shifted, hn::Set(d_i, 0xFF));
    auto vb      = hn::ConvertTo(d_f, masked);
    vb = hn::MulAdd(vb, hn::Set(d_f, logisticScale), hn::Set(d_f, logisticBias));
    return logitNQT(d_f, vb, invScaledAlpha, scaledX0);
}

HWY_FLATTEN float nvq_square_l2_distance_8bit(const float    *HWY_RESTRICT vector,
                                              const uint8_t  *HWY_RESTRICT quantized,
                                              size_t length,
                                              float alpha, float x0,
                                              float minValue, float maxValue)
{
    using FloatTag   = hn::ScalableTag<float>;
    using Int32Tag   = hn::RebindToSigned<FloatTag>;
    using Uint8x4Tag = hn::ScalableTag<uint8_t>;   // 4*kLanes lanes — same total width as FloatTag
    // Tail-path tags (narrower, used only outside the hot loop)
    using Uint16Tag  = hn::Rebind<uint16_t, Int32Tag>;
    using Uint8Tag   = hn::Rebind<uint8_t,  Int32Tag>;
    FloatTag   d_f;
    Int32Tag   d_i;
    Uint8x4Tag d_b;
    Uint16Tag  d_u16;
    Uint8Tag   d_u8;
    constexpr size_t kLanes = hn::MaxLanes(d_f);

    float delta          = maxValue - minValue;
    float scaledAlpha    = alpha / delta;
    float invScaledAlpha = delta / alpha;
    float scaledX0       = x0 * delta;
    float logisticBias   = logisticNQT_scalar(minValue, scaledAlpha, scaledX0);
    float logisticScale  = (logisticNQT_scalar(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255.0f;

    auto squaredSum = hn::Zero(d_f);

    // FastLanes main loop: load 4*kLanes bytes per iteration (full native width),
    // reinterpret as kLanes int32 values, and extract one byte per int32 via
    // shift+mask for each of the 4 parts.  Requires the query vector to have
    // been pre-shuffled by nvq_shuffle_query_in_place_8bit.
    size_t i = 0;
    for (; i + 4 * kLanes <= length; i += 4 * kLanes) {
        auto bytes   = hn::LoadU(d_b, quantized + i);
        auto as_ints = hn::BitCast(d_i, bytes);
        for (int part = 0; part < 4; ++part) {
            auto va   = hn::LoadU(d_f, vector + i + part * kLanes);
            auto vb   = dequantize_bytes_fastlanes(d_f, d_i, as_ints, part,
                                                   logisticScale, logisticBias,
                                                   invScaledAlpha, scaledX0);
            auto diff = hn::Sub(va, vb);
            squaredSum = hn::MulAdd(diff, diff, squaredSum);
        }
    }

    float result = hn::ReduceSum(d_f, squaredSum);

    // kLanes-aligned tail: query floats are un-shuffled here, so use the
    // sequential PromoteTo path which reads bytes in natural order.
    for (; i + kLanes <= length; i += kLanes) {
        auto va   = hn::LoadU(d_f, vector + i);
        auto vb   = dequantize_bytes(d_f, d_i, d_u16, d_u8, quantized, i,
                                     logisticScale, logisticBias, invScaledAlpha, scaledX0);
        auto diff = hn::Sub(va, vb);
        result += hn::ReduceSum(d_f, hn::Mul(diff, diff));
    }

    // Sub-kLanes tail: LoadN zero-pads; mask diff to exclude padding lanes.
    const size_t remaining = length - i;
    if (remaining > 0) {
        const auto mask  = hn::FirstN(d_f, remaining);
        auto va          = hn::LoadN(d_f,  vector    + i, remaining);
        const auto b_u8  = hn::LoadN(d_u8, quantized + i, remaining);
        const auto b_u16 = hn::PromoteTo(d_u16, b_u8);
        const auto b_i32 = hn::PromoteTo(d_i,   b_u16);
        auto vb          = hn::MulAdd(hn::ConvertTo(d_f, b_i32),
                                      hn::Set(d_f, logisticScale),
                                      hn::Set(d_f, logisticBias));
        vb = logitNQT(d_f, vb, invScaledAlpha, scaledX0);
        auto diff = hn::IfThenElseZero(mask, hn::Sub(va, vb));
        result += hn::ReduceSum(d_f, hn::Mul(diff, diff));
    }

    return result;
}

HWY_FLATTEN float nvq_dot_product_8bit(const float   *HWY_RESTRICT vector,
                                       const uint8_t *HWY_RESTRICT quantized,
                                       size_t length,
                                       float alpha, float x0,
                                       float minValue, float maxValue)
{
    using FloatTag   = hn::ScalableTag<float>;
    using Int32Tag   = hn::RebindToSigned<FloatTag>;
    using Uint8x4Tag = hn::ScalableTag<uint8_t>;   // 4*kLanes lanes — same total width as FloatTag
    using Uint16Tag  = hn::Rebind<uint16_t, Int32Tag>;
    using Uint8Tag   = hn::Rebind<uint8_t,  Int32Tag>;
    FloatTag   d_f;
    Int32Tag   d_i;
    Uint8x4Tag d_b;
    Uint16Tag  d_u16;
    Uint8Tag   d_u8;
    constexpr size_t kLanes = hn::MaxLanes(d_f);

    float delta          = maxValue - minValue;
    float scaledAlpha    = alpha / delta;
    float invScaledAlpha = delta / alpha;
    float scaledX0       = x0 * delta;
    float logisticBias   = logisticNQT_scalar(minValue, scaledAlpha, scaledX0);
    float logisticScale  = (logisticNQT_scalar(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255.0f;

    auto dotProd = hn::Zero(d_f);

    // FastLanes main loop: full-width byte load, shift+mask extraction.
    size_t i = 0;
    for (; i + 4 * kLanes <= length; i += 4 * kLanes) {
        auto bytes   = hn::LoadU(d_b, quantized + i);
        auto as_ints = hn::BitCast(d_i, bytes);
        for (int part = 0; part < 4; ++part) {
            auto va = hn::LoadU(d_f, vector + i + part * kLanes);
            auto vb = dequantize_bytes_fastlanes(d_f, d_i, as_ints, part,
                                                 logisticScale, logisticBias,
                                                 invScaledAlpha, scaledX0);
            dotProd = hn::MulAdd(va, vb, dotProd);
        }
    }

    float result = hn::ReduceSum(d_f, dotProd);

    // kLanes-aligned tail: un-shuffled query, sequential byte access.
    for (; i + kLanes <= length; i += kLanes) {
        auto va = hn::LoadU(d_f, vector + i);
        auto vb = dequantize_bytes(d_f, d_i, d_u16, d_u8, quantized, i,
                                   logisticScale, logisticBias, invScaledAlpha, scaledX0);
        result += hn::ReduceSum(d_f, hn::Mul(va, vb));
    }

    // Sub-kLanes tail: LoadN zero-pads va; 0 * vb = 0 for padding lanes.
    const size_t remaining = length - i;
    if (remaining > 0) {
        auto va          = hn::LoadN(d_f,  vector    + i, remaining);
        const auto b_u8  = hn::LoadN(d_u8, quantized + i, remaining);
        const auto b_u16 = hn::PromoteTo(d_u16, b_u8);
        const auto b_i32 = hn::PromoteTo(d_i,   b_u16);
        auto vb          = hn::MulAdd(hn::ConvertTo(d_f, b_i32),
                                      hn::Set(d_f, logisticScale),
                                      hn::Set(d_f, logisticBias));
        vb = logitNQT(d_f, vb, invScaledAlpha, scaledX0);
        result += hn::ReduceSum(d_f, hn::Mul(va, vb));
    }

    return result;
}

// nvq_shuffle_query_in_place_8bit
//
// Pre-processes the float query vector in-place so that nvq_cosine_8bit_packed,
// nvq_dot_product_8bit, and nvq_square_l2_distance_8bit can use the FastLanes
// byte-extraction strategy.  Only complete blocks of 4*kLanes floats are
// transposed; the remaining tail elements are left in their original order and
// processed by the sequential tail path inside each scoring kernel.
//
// The permutation applied is an in-place matrix transpose of 4×kLanes blocks:
// after the shuffle, shuffled[j*kLanes + k] == original[k*4 + j], so that
// when the scoring kernel extracts byte `j` from int32 slot `k` (via
// ShiftRight(8*j) & 0xFF), it lines up with query float at position j*kLanes+k.
HWY_FLATTEN void nvq_shuffle_query_in_place_8bit(float *HWY_RESTRICT vector,
                                                  size_t length)
{
    using FloatTag = hn::ScalableTag<float>;
    FloatTag d_f;
    const size_t kLanes = hn::Lanes(d_f);
    const size_t step   = 4 * kLanes;   // block size = number of bytes in a full-width vector
    const size_t mn1    = step - 1;

    // Maximum step across all ISAs compiled for: 4 * 16 = 64 (AVX-512).
    // Declared outside the loop so the stack frame is allocated once.
    constexpr size_t kMaxStep = 4 * hn::MaxLanes(FloatTag{});
    bool visited[kMaxStep];

    size_t offset = 0;
    while (offset + step <= length) {
        float *arr = vector + offset;
        memset(visited, 0, step);  // only zero the portion we will inspect

        // In-place cyclic transposition: for each unvisited cycle, rotate
        // elements along the cycle defined by a -> (kLanes * a) % mn1.
        // This maps shuffled[p] = original[(p % kLanes) * 4 + (p / kLanes)],
        // which is the inverse of the FastLanes interleaving.
        for (size_t cycle = 1; cycle < step; ++cycle) {
            if (visited[cycle]) continue;
            size_t a = cycle;
            do {
                a = (a == mn1) ? mn1 : (kLanes * a) % mn1;
                float temp = arr[a];
                arr[a]     = arr[cycle];
                arr[cycle] = temp;
                visited[a] = true;
            } while (a != cycle);
        }
        offset += step;
    }
}

// Returns sum and bMagnitude packed into a single int64_t:
//   bits [31:0]  = IEEE-754 bits of sum
//   bits [63:32] = IEEE-754 bits of bMagnitude
// The Java caller unpacks with Float.intBitsToFloat — no heap allocation needed.
HWY_FLATTEN int64_t nvq_cosine_8bit_packed(const float   *HWY_RESTRICT vector,
                                           const uint8_t *HWY_RESTRICT quantized,
                                           size_t length,
                                           float alpha, float x0,
                                           float minValue, float maxValue,
                                           const float   *HWY_RESTRICT centroid)
{
    using FloatTag   = hn::ScalableTag<float>;
    using Int32Tag   = hn::RebindToSigned<FloatTag>;
    using Uint8x4Tag = hn::ScalableTag<uint8_t>;   // 4*kLanes lanes — same total width as FloatTag
    using Uint16Tag  = hn::Rebind<uint16_t, Int32Tag>;
    using Uint8Tag   = hn::Rebind<uint8_t,  Int32Tag>;
    FloatTag   d_f;
    Int32Tag   d_i;
    Uint8x4Tag d_b;
    Uint16Tag  d_u16;
    Uint8Tag   d_u8;
    constexpr size_t kLanes = hn::MaxLanes(d_f);

    float delta          = maxValue - minValue;
    float scaledAlpha    = alpha / delta;
    float invScaledAlpha = delta / alpha;
    float scaledX0       = x0 * delta;
    float logisticBias   = logisticNQT_scalar(minValue, scaledAlpha, scaledX0);
    float logisticScale  = (logisticNQT_scalar(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255.0f;

    auto sumVec  = hn::Zero(d_f);
    auto bMagVec = hn::Zero(d_f);

    // FastLanes main loop: full-width byte load, shift+mask extraction.
    size_t i = 0;
    for (; i + 4 * kLanes <= length; i += 4 * kLanes) {
        auto bytes   = hn::LoadU(d_b, quantized + i);
        auto as_ints = hn::BitCast(d_i, bytes);
        for (int part = 0; part < 4; ++part) {
            auto va = hn::LoadU(d_f, vector   + i + part * kLanes);
            auto vc = hn::LoadU(d_f, centroid + i + part * kLanes);
            auto vb = dequantize_bytes_fastlanes(d_f, d_i, as_ints, part,
                                                 logisticScale, logisticBias,
                                                 invScaledAlpha, scaledX0);
            vb = hn::Add(vb, vc);
            sumVec  = hn::MulAdd(va, vb, sumVec);
            bMagVec = hn::MulAdd(vb, vb, bMagVec);
        }
    }

    float sum  = hn::ReduceSum(d_f, sumVec);
    float bMag = hn::ReduceSum(d_f, bMagVec);

    // kLanes-aligned tail: un-shuffled query and centroid, sequential bytes.
    for (; i + kLanes <= length; i += kLanes) {
        auto va = hn::LoadU(d_f, vector   + i);
        auto vc = hn::LoadU(d_f, centroid + i);
        auto vb = dequantize_bytes(d_f, d_i, d_u16, d_u8, quantized, i,
                                   logisticScale, logisticBias, invScaledAlpha, scaledX0);
        vb = hn::Add(vb, vc);
        sum  += hn::ReduceSum(d_f, hn::Mul(va, vb));
        bMag += hn::ReduceSum(d_f, hn::Mul(vb, vb));
    }

    // Sub-kLanes tail: LoadN zero-pads va and vc.  Mask vb before adding the
    // centroid so padding lanes contribute 0 to both sum and bMagnitude.
    const size_t remaining = length - i;
    if (remaining > 0) {
        const auto mask  = hn::FirstN(d_f, remaining);
        auto va          = hn::LoadN(d_f,  vector    + i, remaining);
        auto vc          = hn::LoadN(d_f,  centroid  + i, remaining);
        const auto b_u8  = hn::LoadN(d_u8, quantized + i, remaining);
        const auto b_u16 = hn::PromoteTo(d_u16, b_u8);
        const auto b_i32 = hn::PromoteTo(d_i,   b_u16);
        auto vb          = hn::MulAdd(hn::ConvertTo(d_f, b_i32),
                                      hn::Set(d_f, logisticScale),
                                      hn::Set(d_f, logisticBias));
        vb = logitNQT(d_f, vb, invScaledAlpha, scaledX0);
        auto vb_c = hn::Add(hn::IfThenElseZero(mask, vb), vc);
        sum  += hn::ReduceSum(d_f, hn::Mul(va, vb_c));
        bMag += hn::ReduceSum(d_f, hn::Mul(vb_c, vb_c));
    }

    int32_t sum_bits, bmag_bits;
    memcpy(&sum_bits,  &sum,  sizeof(float));
    memcpy(&bmag_bits, &bMag, sizeof(float));
    return ((int64_t)bmag_bits << 32) | (int64_t)(uint32_t)sum_bits;
}

}  // namespace JV_ISA
