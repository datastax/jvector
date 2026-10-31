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

#ifndef JVECTOR_ASH_PROJECTION_INL_H
#define JVECTOR_ASH_PROJECTION_INL_H

// Included within JV_ISA. The immutable matrix is contiguous row-major and is
// prepared once per model. A whole query crosses FFM once; no allocation,
// callback, synchronization or pointer retention occurs in this leaf kernel.
void ash_project_f32(const float *HWY_RESTRICT matrix, const float *HWY_RESTRICT query,
                     int rows, int cols, float *HWY_RESTRICT out) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const size_t columns = static_cast<size_t>(cols);
    int row = 0;
    // Four rows share each query load. Two chains per row hide FMA latency
    // without changing the stored matrix or batching independent graph queries.
    for (; rows - row >= 4; row += 4) {
        const float *a = matrix + static_cast<size_t>(row) * columns;
        auto s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
        auto t0 = hn::Zero(d), t1 = hn::Zero(d), t2 = hn::Zero(d), t3 = hn::Zero(d);
        size_t col = 0;
        for (; col + 2 * lanes <= columns; col += 2 * lanes) {
            const auto q0 = hn::LoadU(d, query + col);
            const auto q1 = hn::LoadU(d, query + col + lanes);
            s0 = hn::MulAdd(hn::LoadU(d, a + col), q0, s0);
            s1 = hn::MulAdd(hn::LoadU(d, a + columns + col), q0, s1);
            s2 = hn::MulAdd(hn::LoadU(d, a + 2 * columns + col), q0, s2);
            s3 = hn::MulAdd(hn::LoadU(d, a + 3 * columns + col), q0, s3);
            t0 = hn::MulAdd(hn::LoadU(d, a + col + lanes), q1, t0);
            t1 = hn::MulAdd(hn::LoadU(d, a + columns + col + lanes), q1, t1);
            t2 = hn::MulAdd(hn::LoadU(d, a + 2 * columns + col + lanes), q1, t2);
            t3 = hn::MulAdd(hn::LoadU(d, a + 3 * columns + col + lanes), q1, t3);
        }
        s0 = hn::Add(s0, t0); s1 = hn::Add(s1, t1);
        s2 = hn::Add(s2, t2); s3 = hn::Add(s3, t3);
        for (; col < columns; col += lanes) {
            const size_t n = std::min(lanes, columns - col);
            const auto q = hn::LoadN(d, query + col, n);
            s0 = hn::MulAdd(hn::LoadN(d, a + col, n), q, s0);
            s1 = hn::MulAdd(hn::LoadN(d, a + columns + col, n), q, s1);
            s2 = hn::MulAdd(hn::LoadN(d, a + 2 * columns + col, n), q, s2);
            s3 = hn::MulAdd(hn::LoadN(d, a + 3 * columns + col, n), q, s3);
        }
        out[row] = hn::ReduceSum(d, s0); out[row + 1] = hn::ReduceSum(d, s1);
        out[row + 2] = hn::ReduceSum(d, s2); out[row + 3] = hn::ReduceSum(d, s3);
    }
    for (; row < rows; ++row) {
        const float *a = matrix + static_cast<size_t>(row) * columns;
        auto sum = hn::Zero(d);
        for (size_t col = 0; col < columns; col += lanes) {
            const size_t n = std::min(lanes, columns - col);
            sum = hn::MulAdd(hn::LoadN(d, a + col, n), hn::LoadN(d, query + col, n), sum);
        }
        out[row] = hn::ReduceSum(d, sum);
    }
}

// A packed matrix consists of 32-output tiles. Within each tile the original
// dimensions are contiguous rows of up to 32 weights (no tail padding). The
// caller prepares this once; each tile is reused across the whole input batch.
// All operands, FMA accumulators and results are FP32.
template <bool FourQueries, bool TwoVectors>
HWY_INLINE void ash_project_batch_tile(const float* HWY_RESTRICT weights,
                                      const float* HWY_RESTRICT inputs,
                                      int rows, int cols, size_t width, size_t firstLane,
                                      size_t n0, size_t n1,
                                      float* HWY_RESTRICT out) {
    const hn::CappedTag<float, 16> d;
    const size_t lanes = hn::Lanes(d);
    const size_t columns = static_cast<size_t>(cols);
    const size_t stride = static_cast<size_t>(rows);
    auto s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
    auto t0 = hn::Zero(d), t1 = hn::Zero(d), t2 = hn::Zero(d), t3 = hn::Zero(d);
    for (size_t col = 0; col < columns; ++col) {
        const float* w = weights + col * width + firstLane;
        const auto a0 = hn::LoadN(d, w, n0);
        const auto q0 = hn::Set(d, inputs[col]);
        s0 = hn::MulAdd(a0, q0, s0);
        if constexpr (TwoVectors) {
            const auto a1 = hn::LoadN(d, w + lanes, n1);
            t0 = hn::MulAdd(a1, q0, t0);
            if constexpr (FourQueries) {
                t1 = hn::MulAdd(a1, hn::Set(d, inputs[columns + col]), t1);
                t2 = hn::MulAdd(a1, hn::Set(d, inputs[2 * columns + col]), t2);
                t3 = hn::MulAdd(a1, hn::Set(d, inputs[3 * columns + col]), t3);
            }
        }
        if constexpr (FourQueries) {
            s1 = hn::MulAdd(a0, hn::Set(d, inputs[columns + col]), s1);
            s2 = hn::MulAdd(a0, hn::Set(d, inputs[2 * columns + col]), s2);
            s3 = hn::MulAdd(a0, hn::Set(d, inputs[3 * columns + col]), s3);
        }
    }
    hn::StoreN(s0, d, out, n0);
    if constexpr (TwoVectors) hn::StoreN(t0, d, out + lanes, n1);
    if constexpr (FourQueries) {
        hn::StoreN(s1, d, out + stride, n0);
        hn::StoreN(s2, d, out + 2 * stride, n0);
        hn::StoreN(s3, d, out + 3 * stride, n0);
        if constexpr (TwoVectors) {
            hn::StoreN(t1, d, out + stride + lanes, n1);
            hn::StoreN(t2, d, out + 2 * stride + lanes, n1);
            hn::StoreN(t3, d, out + 3 * stride + lanes, n1);
        }
    }
}

void ash_project_batch_f32(const float* HWY_RESTRICT matrix,
                           const float* HWY_RESTRICT inputs,
                           int rows, int cols, int count,
                           float* HWY_RESTRICT out) {
    const hn::CappedTag<float, 16> d;
    const size_t lanes = hn::Lanes(d);
    for (size_t row = 0; row < static_cast<size_t>(rows); row += 32) {
        const size_t tileRows = std::min(size_t{32}, static_cast<size_t>(rows) - row);
        const float* weights = matrix + row * static_cast<size_t>(cols);
        for (size_t offset = 0; offset < tileRows; offset += 2 * lanes) {
            const size_t n0 = std::min(lanes, tileRows - offset);
            const size_t n1 = tileRows - offset > lanes
                    ? std::min(lanes, tileRows - offset - lanes) : 0;
            int b = 0;
            for (; count - b >= 4; b += 4) {
                const float* x = inputs + static_cast<size_t>(b) * cols;
                float* y = out + static_cast<size_t>(b) * rows + row + offset;
                if (n1 != 0)
                    ash_project_batch_tile<true, true>(weights, x, rows, cols, tileRows, offset, n0, n1, y);
                else
                    ash_project_batch_tile<true, false>(weights, x, rows, cols, tileRows, offset, n0, n1, y);
            }
            for (; b < count; ++b) {
                const float* x = inputs + static_cast<size_t>(b) * cols;
                float* y = out + static_cast<size_t>(b) * rows + row + offset;
                if (n1 != 0)
                    ash_project_batch_tile<false, true>(weights, x, rows, cols, tileRows, offset, n0, n1, y);
                else
                    ash_project_batch_tile<false, false>(weights, x, rows, cols, tileRows, offset, n0, n1, y);
            }
        }
    }
}
#endif
