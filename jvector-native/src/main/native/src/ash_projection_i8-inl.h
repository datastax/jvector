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

#ifndef JVECTOR_ASH_PROJECTION_I8_INL_H
#define JVECTOR_ASH_PROJECTION_I8_INL_H

// INT8 ASH query projection with exact u8 x s8 integer dot products (VNNI vpdpbusd).
// Included only by the VNNI tiers, inside their namespace, with JV_ASH_I8_WIDE set:
//  - 0 (AVX3_DL, Ice Lake class and AMD Zen 4/5; AVX3_VNNI, Cascade Lake and Cooper Lake):
//    256-bit registers. On Ice Lake, dense
//    512-bit integer dot products slowed the graph search that follows them by 9-15%;
//    the same tiles on 256-bit registers did not, and were the fastest.
//  - 1 (AVX3_SPR, Sapphire Rapids and later): 512-bit registers, 1-4% faster there.
// Other tiers keep the stub in jvector_simd_kernels.cpp, which reports the kernel unsupported.
//
// The caller owns the model: packed INT8 rows, one FP32 scale and one int32 sum per row,
// quantized with the same rounding as quantize() below. Packed layout:
// [row block of 16][column group of 4][16 rows][4 columns], zero padded to whole blocks
// and groups. Scratch holds 2 * 8 * paddedColumns query bytes, then 2 * 8 * paddedRows
// int32 dot products. Each query becomes two INT8 digits with one scale,
// query / scale ~ coarse + fine / 254; both dot products are exact in int32 for up to
// 65535 columns, so the output does not depend on the tier or the tile size.

#if !defined(JV_ASH_I8_WIDE)
#error "Define JV_ASH_I8_WIDE before including ash_projection_i8-inl.h"
#endif
// The u8 x s8 dot-product step; a tier whose Highway target emulates it may substitute vpdpbusd.
#if !defined(JV_ASH_I8_DOT)
#define JV_ASH_I8_DOT(d32, a, b, sum) hn::SumOfMulQuadAccumulate(d32, a, b, sum)
#endif

namespace {

constexpr int kI8Ok = 0, kI8InvalidShape = 2, kI8ScaleUnderflow = 3;
constexpr int kI8Tile = 8, kI8MaxColumns = 65535;

// Returns the scale, or NaN without writing any digit when an operand is not finite, so a
// non-finite query yields NaN outputs for that query only. Matches the Java model quantizer.
float i8_quantize(const float* input, int count, uint8_t* output, int32_t offset) {
    float maxabs = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) return std::numeric_limits<float>::quiet_NaN();
        maxabs = std::max(maxabs, std::fabs(input[i]));
    }
    float scale = maxabs == 0 ? 1.0f : maxabs / 127.0f;
    if (scale == 0) return 0.0f; // Rejected by the caller: a nonzero operand underflowed.
    for (int i = 0; i < count; ++i) {
        int q = std::clamp(int(std::nearbyint(input[i] / scale)), -127, 127);
        output[i] = uint8_t(q + offset);
    }
    return scale;
}

// The fine digit holds the coarse digit's rounding remainder, which accounts for nearly
// all INT8 projection error. Both digits are stored with the +128 offset.
float i8_quantize_two_digits(const float* input, int count, uint8_t* coarse, uint8_t* fine) {
    float scale = i8_quantize(input, count, coarse, 128);
    if (!(scale > 0)) return scale;
    for (int i = 0; i < count; ++i) {
        float remainder = input[i] / scale - float(int(coarse[i]) - 128);
        fine[i] = uint8_t(std::clamp(int(std::nearbyint(remainder * 254.0f)), -127, 127) + 128);
    }
    return scale;
}

// coarse * 254 + fine is combined exactly in integers, so the floating-point steps are one
// conversion and multiplications only: nothing a compiler can fuse into an FMA, and every
// build rounds exactly as the Java test oracle does. |coarse| < 65536 * 127 * 127.
inline float i8_combine(int32_t coarse, int32_t fine, float queryScale, float rowScale) {
    constexpr float kInv254 = 1.0f / 254.0f;
    return float(int64_t(coarse) * 254 + fine) * kInv254 * queryScale * rowScale;
}

#if JV_ASH_I8_WIDE
using I8Tag = hn::ScalableTag<int32_t>;   // 16 lanes: one register per 16-row block
#else
using I8Tag = hn::CappedTag<int32_t, 8>;  // 8 lanes: two registers per 16-row block
#endif
constexpr int kI8Lanes = 64 / (JV_ASH_I8_WIDE ? 1 : 2) / 4;
constexpr int kI8VecsPerBlock = 16 / kI8Lanes;

// NB blocks of 16 output rows for every query row of the tile: M * NB * kI8VecsPerBlock
// independent int32 accumulators hide the dot-product latency. Missing final blocks
// recompute block 0 and are not stored.
template <int M, int NB>
HWY_INLINE void i8_tile(const uint8_t* HWY_RESTRICT query, size_t stride, const int8_t* HWY_RESTRICT packed,
                        size_t groups, size_t blocks, int32_t* HWY_RESTRICT out, size_t outStride) {
    const I8Tag d32;
    const hn::Repartition<uint8_t, I8Tag> du8;
    const hn::Repartition<int8_t, I8Tag> di8;
    constexpr int kRow = NB * kI8VecsPerBlock;  // accumulators per query row
    static_assert(kI8Lanes * kI8VecsPerBlock == 16, "16 rows per block");
    for (size_t block = 0; block < blocks; block += NB) {
        const int8_t* w[NB];
        for (int j = 0; j < NB; ++j)
            w[j] = packed + (block + j < blocks ? block + j : block) * groups * 64;
        hn::Vec<I8Tag> acc[M * kRow];
        for (int k = 0; k < M * kRow; ++k) acc[k] = hn::Zero(d32);
        for (size_t g = 0; g < groups; ++g) {
            hn::Vec<decltype(di8)> b[kRow];
            for (int k = 0; k < kRow; ++k)
                b[k] = hn::LoadU(di8, w[k / kI8VecsPerBlock] + g * 64 + (k % kI8VecsPerBlock) * kI8Lanes * 4);
            for (int i = 0; i < M; ++i) {
                int32_t four;
                std::memcpy(&four, query + i * stride + g * 4, sizeof(four));
                const auto a = hn::BitCast(du8, hn::Set(d32, four));
                for (int k = 0; k < kRow; ++k)
                    acc[i * kRow + k] = JV_ASH_I8_DOT(d32, a, b[k], acc[i * kRow + k]);
            }
        }
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < NB && block + j < blocks; ++j)
                for (int v = 0; v < kI8VecsPerBlock; ++v)
                    hn::StoreU(acc[i * kRow + j * kI8VecsPerBlock + v], d32,
                               out + i * outStride + (block + j) * 16 + v * kI8Lanes);
    }
}

// GCC 12 at -O3: when interprocedural constant propagation specializes this function for
// its caller, the tiles copy every accumulator to another register around each dot product
// (two extra instructions per vpdpbusd), about 20% slower at 4-8 queries on Ice Lake.
// Keeping it out of interprocedural optimization leaves the accumulators in place.
#if defined(__GNUC__) && !defined(__clang__)
#define JV_ASH_I8_GEMM_ATTR __attribute__((noipa))
#else
#define JV_ASH_I8_GEMM_ATTR HWY_NOINLINE
#endif

// Tile shapes per query count (rows x 16-row blocks), tuned on Ice Lake and Granite Rapids.
JV_ASH_I8_GEMM_ATTR void i8_gemm(int count, const uint8_t* query, size_t stride, const int8_t* packed, size_t groups,
             size_t blocks, int32_t* out, size_t outStride) {
#if JV_ASH_I8_WIDE
    switch (count) {
        case 1: i8_tile<1, 4>(query, stride, packed, groups, blocks, out, outStride); break;
        case 2: i8_tile<2, 4>(query, stride, packed, groups, blocks, out, outStride); break;
        case 3: i8_tile<3, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 4: i8_tile<4, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 5: i8_tile<5, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 6: i8_tile<6, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 7: i8_tile<7, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        default: i8_tile<8, 2>(query, stride, packed, groups, blocks, out, outStride); break;
    }
#else
    switch (count) {
        case 1: i8_tile<1, 4>(query, stride, packed, groups, blocks, out, outStride); break;
        case 2: i8_tile<2, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 3: i8_tile<3, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 4: i8_tile<4, 2>(query, stride, packed, groups, blocks, out, outStride); break;
        case 5: i8_tile<5, 1>(query, stride, packed, groups, blocks, out, outStride); break;
        case 6: i8_tile<6, 1>(query, stride, packed, groups, blocks, out, outStride); break;
        case 7: i8_tile<7, 1>(query, stride, packed, groups, blocks, out, outStride); break;
        default: i8_tile<8, 1>(query, stride, packed, groups, blocks, out, outStride); break;
    }
#endif
}

#undef JV_ASH_I8_GEMM_ATTR

} // namespace

int ash_project_i8_available() { return 1; }

// Returns 0; 2 for an invalid shape or pointer, before anything is written; or 3 when a
// nonzero query's scale underflows FP32, after which outputs may be partially written.
// Outputs of a query whose values are not all finite are NaN; the rest of the batch is
// unaffected.
int ash_project_batch_i8(const int8_t *HWY_RESTRICT packed, const float *HWY_RESTRICT rowScales,
                         const int32_t *HWY_RESTRICT rowSums, int rows, int cols,
                         const float *HWY_RESTRICT inputs, int count, unsigned char *HWY_RESTRICT scratch,
                         float *HWY_RESTRICT out) {
    if (packed == nullptr || rowScales == nullptr || rowSums == nullptr || inputs == nullptr ||
        scratch == nullptr || out == nullptr || rows <= 0 || cols <= 0 || cols > kI8MaxColumns || count < 0) {
        return kI8InvalidShape;
    }
    const size_t columns = size_t(cols), paddedColumns = (columns + 3) & ~size_t{3};
    const size_t outputs = size_t(rows), paddedRows = (outputs + 15) & ~size_t{15};
    uint8_t* coarse = scratch;
    uint8_t* fine = coarse + kI8Tile * paddedColumns;
    int32_t* coarseDots = reinterpret_cast<int32_t*>(fine + kI8Tile * paddedColumns);
    int32_t* fineDots = coarseDots + kI8Tile * paddedRows;
    float scales[kI8Tile];
    for (int first = 0; first < count;) {
        const int size = std::min(kI8Tile, count - first);
        for (int i = 0; i < size; ++i) {
            scales[i] = i8_quantize_two_digits(inputs + size_t(first + i) * columns, cols,
                    coarse + i * paddedColumns, fine + i * paddedColumns);
            if (scales[i] == 0) return kI8ScaleUnderflow;
        }
        i8_gemm(size, coarse, paddedColumns, packed, paddedColumns / 4, paddedRows / 16, coarseDots, paddedRows);
        i8_gemm(size, fine, paddedColumns, packed, paddedColumns / 4, paddedRows / 16, fineDots, paddedRows);
        for (int i = 0; i < size; ++i)
            for (size_t row = 0; row < outputs; ++row) {
                const size_t at = i * paddedRows + row;
                const int32_t c = coarseDots[at] - 128 * rowSums[row];
                const int32_t f = fineDots[at] - 128 * rowSums[row];
                out[size_t(first + i) * outputs + row] = i8_combine(c, f, scales[i], rowScales[row]);
            }
        first += size;
    }
    return kI8Ok;
}

#endif
