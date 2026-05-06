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
#include <math.h>
#include "jvector_simd.h"
#include "hwy/highway.h"

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
//   (assemble_and_sum_f32_512, pq_decoded_cosine_similarity_f32_512, the three
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
//   Used in assemble_and_sum_f32_512 and pq_decoded_cosine_similarity to
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
//   Permutes pairs of adjacent lanes: [0,1,2,3] → [2,3,0,1].
//   Adding a vector with its Shuffle2301 partner sums adjacent pairs.
//
// Shuffle1032(vec)
//   Permutes 32-bit elements within each 128-bit lane: [0,1,2,3] → [1,0,3,2].
//   Used as a second reduction step to sum the results of Shuffle2301.
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
// Dot product specialised for short vectors (size <= N).
//
// HWY_CAPPED(float, N) caps the SIMD vector width to at most N lanes.  For a
// 4-element vector on AVX-512 this avoids allocating a full 512-bit register
// for work that fits in 128 bits, reducing register pressure and potentially
// improving throughput on CPUs where narrow operations are cheaper.
template <int N>
HWY_INLINE float DotProductCapped(const float *a, const float *b, size_t size)
{
    // capped_tag limits the vector width to at most N floats.
    HWY_CAPPED(float, N) capped_tag;
    auto dot_acc = hn::Zero(capped_tag);
    size_t ii = 0;
    for (; ii + hn::Lanes(capped_tag) <= size; ii += hn::Lanes(capped_tag)) {
        auto va = hn::LoadU(capped_tag, a + ii);
        auto vb = hn::LoadU(capped_tag, b + ii);
        dot_acc = hn::MulAdd(va, vb, dot_acc);
    }
    // Handle any remaining elements that don't fill a full vector.
    if (ii < size) {
        auto va_tail = hn::LoadN(capped_tag, a + ii, size - ii);
        auto vb_tail = hn::LoadN(capped_tag, b + ii, size - ii);
        dot_acc = hn::MulAdd(va_tail, vb_tail, dot_acc);
    }
    return hn::ReduceSum(capped_tag, dot_acc);
}

// L2 distance specialised for short vectors (size <= N).
template <int N>
HWY_INLINE float L2DistanceCapped(const float *a, const float *b, size_t size)
{
    HWY_CAPPED(float, N) capped_tag;
    auto sq_diff_acc = hn::Zero(capped_tag);
    size_t ii = 0;
    for (; ii + hn::Lanes(capped_tag) <= size; ii += hn::Lanes(capped_tag)) {
        auto va = hn::LoadU(capped_tag, a + ii);
        auto vb = hn::LoadU(capped_tag, b + ii);
        auto diff = va - vb;
        sq_diff_acc = hn::MulAdd(diff, diff, sq_diff_acc);
    }
    if (ii < size) {
        auto va = hn::LoadN(capped_tag, a + ii, size - ii);
        auto vb = hn::LoadN(capped_tag, b + ii, size - ii);
        auto diff = va - vb;
        sq_diff_acc = hn::MulAdd(diff, diff, sq_diff_acc);
    }
    return hn::ReduceSum(capped_tag, sq_diff_acc);
}

// Cosine distance specialised for short vectors (size <= N).
template <int N>
HWY_INLINE float
CosineDistanceCapped(const float *a, const float *b, size_t size)
{
    HWY_CAPPED(float, N) capped_tag;
    auto sum_ab = hn::Zero(capped_tag);
    auto sum_aa = hn::Zero(capped_tag);
    auto sum_bb = hn::Zero(capped_tag);
    size_t ii = 0;
    for (; ii + hn::Lanes(capped_tag) <= size; ii += hn::Lanes(capped_tag)) {
        auto va = hn::LoadU(capped_tag, a + ii);
        auto vb = hn::LoadU(capped_tag, b + ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    if (ii < size) {
        auto va = hn::LoadN(capped_tag, a + ii, size - ii);
        auto vb = hn::LoadN(capped_tag, b + ii, size - ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    float dot = hn::ReduceSum(capped_tag, sum_ab);
    float norm_a = hn::ReduceSum(capped_tag, sum_aa);
    float norm_b = hn::ReduceSum(capped_tag, sum_bb);
    return dot / sqrtf(norm_a * norm_b);
}

// Returns the dot product sum(a[ii] * b[ii]).
//
// Short-vector fast paths: when the register width is wider than the vector
// (e.g. a 4-element input on AVX-512), using the full register wastes lanes
// and can hurt latency.  HWY_CAPPED paths in DotProductCapped cap
// the width to match the data, keeping execution in narrow registers.
HWY_INLINE float DotProduct(const float *a,
                            size_t aoffset,
                            const float *b,
                            size_t boffset,
                            size_t length)
{
    a += aoffset;
    b += boffset;
    const size_t size = length;
    // On ISAs with >128-bit registers, prefer a capped tag for tiny vectors
    // to avoid unnecessarily wide SIMD operations.
#if HWY_MAX_BYTES > 16
    if (size <= 4) { return DotProductCapped<4>(a, b, size); }
#if HWY_MAX_BYTES > 32
    if (size <= 8) { return DotProductCapped<8>(a, b, size); }
#endif
#endif // HWY_MAX_BYTES >= 16

    // General case: use the full native vector width.
    hn::ScalableTag<float> tag;
    const size_t lanes = hn::Lanes(tag);

    // 4x unrolled loop: four independent accumulators hide FMA latency.
    auto dot_acc0 = hn::Zero(tag);
    auto dot_acc1 = hn::Zero(tag);
    auto dot_acc2 = hn::Zero(tag);
    auto dot_acc3 = hn::Zero(tag);

    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        auto va0 = hn::LoadU(tag, a + ii + 0 * lanes);
        auto vb0 = hn::LoadU(tag, b + ii + 0 * lanes);
        auto va1 = hn::LoadU(tag, a + ii + 1 * lanes);
        auto vb1 = hn::LoadU(tag, b + ii + 1 * lanes);
        auto va2 = hn::LoadU(tag, a + ii + 2 * lanes);
        auto vb2 = hn::LoadU(tag, b + ii + 2 * lanes);
        auto va3 = hn::LoadU(tag, a + ii + 3 * lanes);
        auto vb3 = hn::LoadU(tag, b + ii + 3 * lanes);
        dot_acc0 = hn::MulAdd(va0, vb0, dot_acc0);
        dot_acc1 = hn::MulAdd(va1, vb1, dot_acc1);
        dot_acc2 = hn::MulAdd(va2, vb2, dot_acc2);
        dot_acc3 = hn::MulAdd(va3, vb3, dot_acc3);
    }
    // Fold the four accumulators into one.
    auto dot_acc
            = hn::Add(hn::Add(dot_acc0, dot_acc1), hn::Add(dot_acc2, dot_acc3));

    // Remaining full vectors.
    for (; ii + lanes <= size; ii += lanes) {
        auto va = hn::LoadU(tag, a + ii);
        auto vb = hn::LoadU(tag, b + ii);
        dot_acc = hn::MulAdd(va, vb, dot_acc);
    }

    // Tail: zero-padded masked load handles the last < lanes elements.
    if (ii < size) {
        auto va_tail = hn::LoadN(tag, a + ii, size - ii);
        auto vb_tail = hn::LoadN(tag, b + ii, size - ii);
        dot_acc = hn::MulAdd(va_tail, vb_tail, dot_acc);
    }

    return hn::ReduceSum(tag, dot_acc);
}

HWY_INLINE float CosineDistance(
        const float *a, int aoffset, const float *b, int boffset, int length)
{
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const float *ap = a + aoffset;
    const float *bp = b + boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return CosineDistanceCapped<4>(ap, bp, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return CosineDistanceCapped<8>(ap, bp, length); }
#endif
#endif

    auto dotSum = hn::Zero(d);
    auto normASum = hn::Zero(d);
    auto normBSum = hn::Zero(d);

    int ii = 0;
    for (; ii + N <= length; ii += N) {
        auto ai = hn::LoadU(d, ap + ii);
        auto bi = hn::LoadU(d, bp + ii);
        dotSum = hn::MulAdd(ai, bi, dotSum);
        normASum = hn::MulAdd(ai, ai, normASum);
        normBSum = hn::MulAdd(bi, bi, normBSum);
    }

    // Masked tail
    const int remaining = length - ii;
    if (remaining > 0) {
        const auto mask = hn::FirstN(d, remaining);
        auto ai = hn::MaskedLoad(mask, d, ap + ii);
        auto bi = hn::MaskedLoad(mask, d, bp + ii);
        dotSum = hn::MulAdd(ai, bi, dotSum);
        normASum = hn::MulAdd(ai, ai, normASum);
        normBSum = hn::MulAdd(bi, bi, normBSum);
    }

    float dot = hn::ReduceSum(d, dotSum);
    float normA = hn::ReduceSum(d, normASum);
    float normB = hn::ReduceSum(d, normBSum);

    return dot / sqrtf(normA * normB);
}

HWY_INLINE float L2SquareDistance(const float *a,
                                  size_t aoffset,
                                  const float *b,
                                  size_t boffset,
                                  size_t length)
{
    a += aoffset;
    b += boffset;
    const size_t size = length;
#if HWY_MAX_BYTES > 16
    if (size <= 4) { return L2DistanceCapped<4>(a, b, size); }
#if HWY_MAX_BYTES > 32
    if (size <= 8) { return L2DistanceCapped<8>(a, b, size); }
#endif
#endif

    hn::ScalableTag<float> tag;

    const size_t lanes = hn::Lanes(tag);

    // 4x unrolled: four independent accumulators hide FMA latency.
    auto sq_diff_acc0 = hn::Zero(tag);
    auto sq_diff_acc1 = hn::Zero(tag);
    auto sq_diff_acc2 = hn::Zero(tag);
    auto sq_diff_acc3 = hn::Zero(tag);

    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        auto va0 = hn::LoadU(tag, a + ii + 0 * lanes);
        auto vb0 = hn::LoadU(tag, b + ii + 0 * lanes);
        auto va1 = hn::LoadU(tag, a + ii + 1 * lanes);
        auto vb1 = hn::LoadU(tag, b + ii + 1 * lanes);
        auto va2 = hn::LoadU(tag, a + ii + 2 * lanes);
        auto vb2 = hn::LoadU(tag, b + ii + 2 * lanes);
        auto va3 = hn::LoadU(tag, a + ii + 3 * lanes);
        auto vb3 = hn::LoadU(tag, b + ii + 3 * lanes);
        auto diff0 = va0 - vb0;
        auto diff1 = va1 - vb1;
        auto diff2 = va2 - vb2;
        auto diff3 = va3 - vb3;
        sq_diff_acc0 = hn::MulAdd(diff0, diff0, sq_diff_acc0);
        sq_diff_acc1 = hn::MulAdd(diff1, diff1, sq_diff_acc1);
        sq_diff_acc2 = hn::MulAdd(diff2, diff2, sq_diff_acc2);
        sq_diff_acc3 = hn::MulAdd(diff3, diff3, sq_diff_acc3);
    }
    // Fold the four accumulators into one.
    auto sq_diff_acc = hn::Add(hn::Add(sq_diff_acc0, sq_diff_acc1),
                               hn::Add(sq_diff_acc2, sq_diff_acc3));

    // Remaining full vectors.
    for (; ii + lanes <= size; ii += lanes) {
        auto va = hn::LoadU(tag, a + ii);
        auto vb = hn::LoadU(tag, b + ii);
        auto diff = va - vb;
        sq_diff_acc = hn::MulAdd(diff, diff, sq_diff_acc);
    }

    // Tail: LoadN zero-pads elements beyond the end so MulAdd handles the remainder.
    if (ii < size) {
        auto va = hn::LoadN(tag, a + ii, size - ii);
        auto vb = hn::LoadN(tag, b + ii, size - ii);
        auto diff = va - vb;
        sq_diff_acc = hn::MulAdd(diff, diff, sq_diff_acc);
    }

    return hn::ReduceSum(tag, sq_diff_acc);
}

/* PQ related SIMD kernels */

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
                               int size)
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
                                           int size,
                                           int clusterCount,
                                           const float *HWY_RESTRICT query,
                                           int queryOffset,
                                           float *HWY_RESTRICT partialSums)
{
    int codebookBase = codebookIndex * clusterCount;
    using FloatTag = hn::ScalableTag<float>;
    FloatTag tag;
    constexpr size_t kLanes = Lanes(tag);
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
            constexpr size_t centroids_per_iter = kLanes / 4;
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
            constexpr size_t centroids_per_iter = kLanes / 8;

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

/* List API's exposed to JAVA via FFI here: Do not mark them static or online,
 * as they need to be visible to the dynamic linker and we may want to
 * benchmark them individually in C.
 */

// HWY_RESTRICT is not needed here: `data` (float*) and `baseOffsets` (unsigned char*)
// have different types, so the compiler already treats them as non-aliasing under
// C++ strict-aliasing rules. Neither pointer is written through, so there are no
// stores that could force a reload of the other.
HWY_FLATTEN float assemble_and_sum_f32_512(const float *data,
                                           int dataBase,
                                           const unsigned char *baseOffsets,
                                           int baseOffsetsOffset,
                                           int baseOffsetsLength)
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
    const auto dataBaseVec = hn::Set(int32Tag, dataBase);
    const auto laneIncrement = hn::Set(int32Tag, static_cast<int32_t>(kLanes));
    // indexVec[k] = ii + k, initialised to [0, 1, ..., kLanes-1]
    auto indexVec = hn::Iota(int32Tag, 0);

    int ii = 0;
    for (; ii + static_cast<int>(kLanes) <= baseOffsetsLength;
         ii += static_cast<int>(kLanes)) {
        // Load kLanes bytes and zero-extend to int32 via two PromoteTo steps (u8→u16→i32)
        const auto u8Vec = hn::LoadU(uint8Tag, baseOffsets + ii);
        const auto u16Vec = hn::PromoteTo(uint16Tag, u8Vec);
        const auto offsetVec = hn::PromoteTo(int32Tag, u16Vec);

        // gatherIndices[k] = (ii + k) * dataBase + baseOffsets[ii + k]
        const auto gatherIndices
                = hn::Add(hn::Mul(indexVec, dataBaseVec), offsetVec);

        sumVec = hn::Add(sumVec,
                         hn::GatherIndex(floatTag, data, gatherIndices));
        indexVec = hn::Add(indexVec, laneIncrement);
    }

    float res = hn::ReduceSum(floatTag, sumVec);
    for (; ii < baseOffsetsLength; ii++) {
        res += data[dataBase * ii + baseOffsets[ii]];
    }
    return res;
}

// HWY_RESTRICT is not needed here: `baseOffsets` (unsigned char*) is a different type
// from `partialSums` and `aMagnitude` (float*), so strict-aliasing already guarantees
// the compiler that writes through one cannot affect loads from the other. All three
// pointers are read-only within the loop, so there are no stores to reason about anyway.
HWY_FLATTEN float
pq_decoded_cosine_similarity_f32_512(const unsigned char *baseOffsets,
                                     int baseOffsetsOffset,
                                     int baseOffsetsLength,
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
    const auto clusterCountVec = hn::Set(int32Tag, clusterCount);
    const auto laneIncrement = hn::Set(int32Tag, static_cast<int32_t>(kLanes));
    auto indexVec = hn::Iota(int32Tag, 0);

    int ii = 0;
    for (; ii + static_cast<int>(kLanes) <= baseOffsetsLength;
         ii += static_cast<int>(kLanes)) {
        // Load kLanes bytes and zero-extend to int32 via two PromoteTo steps (u8→u16→i32)
        const auto u8Vec = hn::LoadU(uint8Tag, baseOffsets + ii);
        const auto u16Vec = hn::PromoteTo(uint16Tag, u8Vec);
        const auto offsetVec = hn::PromoteTo(int32Tag, u16Vec);

        // gatherIndices[k] = (ii + k) * clusterCount + baseOffsets[ii + k]
        const auto gatherIndices
                = hn::Add(hn::Mul(indexVec, clusterCountVec), offsetVec);

        sumVec = hn::Add(sumVec,
                         hn::GatherIndex(floatTag, partialSums, gatherIndices));
        magnitudeVec
                = hn::Add(magnitudeVec,
                          hn::GatherIndex(floatTag, aMagnitude, gatherIndices));
        indexVec = hn::Add(indexVec, laneIncrement);
    }

    float sumResult = hn::ReduceSum(floatTag, sumVec);
    float aMagnitudeResult = hn::ReduceSum(floatTag, magnitudeVec);

    // Handle the remaining elements
    for (; ii < baseOffsetsLength; ii++) {
        int offset = clusterCount * ii + baseOffsets[ii];
        sumResult += partialSums[offset];
        aMagnitudeResult += aMagnitude[offset];
    }

    return sumResult / sqrtf(aMagnitudeResult * bMagnitude);
}

HWY_FLATTEN float cosine_f32_512_native(
        const float *a, int aoffset, const float *b, int boffset, int length)
{
    return CosineDistance(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float dot_product_f32_512_native(
        const float *a, int aoffset, const float *b, int boffset, int length)
{
    return DotProduct(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float euclidean_f32_512_native(
        const float *a, int aoffset, const float *b, int boffset, int length)
{
    return L2SquareDistance(a, aoffset, b, boffset, length);
}

HWY_FLATTEN void calculate_partial_sums_dot_f32_512(const float *codebook,
                                                    int codebookIndex,
                                                    int size,
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

HWY_FLATTEN void calculate_partial_sums_euclidean_f32_512(const float *codebook,
                                                          int codebookIndex,
                                                          int size,
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
