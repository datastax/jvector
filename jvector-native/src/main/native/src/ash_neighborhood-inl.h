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

#ifndef JVECTOR_ASH_NEIGHBORHOOD_INL_H
#define JVECTOR_ASH_NEIGHBORHOOD_INL_H

// Included inside JV_ISA after Highway headers. The helpers are private to each
// ISA translation unit. The public entry is registered by the kernel X-macro.
namespace {
constexpr int kAshSuccess = 0;
constexpr int kAshInvalidLandmark = 1;
constexpr int kAshInvalidShape = 2;
constexpr size_t kAshHeaderBytesPerLane = 5;

template <class D, class I> HWY_INLINE auto AshLookup(D d, const float *lut, I indices) {
    const hn::Rebind<int32_t, D> di;
    if constexpr (hn::MaxLanes(d) == 16) {
        return hn::TableLookupLanes(hn::LoadU(d, lut), hn::IndicesFromVec(d, hn::BitCast(di, indices)));
    } else {
        // AVX2/scalar do not have a register-wide 16-float permute.
        return hn::GatherIndex(d, lut, hn::BitCast(di, indices));
    }
}

template <class D, class V> HWY_INLINE auto AshPromoteHalf(D d, V bits) {
    const hn::Rebind<hwy::float16_t, D> df16;
    auto value = hn::PromoteTo(d, hn::BitCast(df16, bits));
    if constexpr (hn::MaxLanes(d) < 8) {
        // Highway's SSE software conversion assumes finite half values. Repair
        // exponent 31 explicitly; AVX2/AVX-512 use hardware F16 conversion.
        const hn::Rebind<uint32_t, D> du;
        const auto wide = hn::PromoteTo(du, bits);
        const auto exponent = hn::And(wide, hn::Set(du, 0x7c00));
        const auto special = hn::Eq(exponent, hn::Set(du, 0x7c00));
        const auto sign = hn::ShiftLeft<16>(hn::And(wide, hn::Set(du, 0x8000)));
        const auto mantissa = hn::ShiftLeft<13>(hn::And(wide, hn::Set(du, 0x03ff)));
        const auto ieee = hn::BitCast(d, hn::Or(hn::Or(sign, mantissa), hn::Set(du, 0x7f800000)));
        value = hn::IfThenElse(hn::RebindMask(d, special), ieee, value);
    }
    return value;
}

template <class D>
HWY_INLINE int AshFinish(D d, const unsigned char *header, int blockSize, int lane, int count, float landmarkDot,
                         hn::VFromD<D> dot, float *out) {
    const hn::Rebind<uint8_t, D> db;
    const hn::Rebind<uint16_t, D> dh;
    const auto landmarks = hn::LoadN(db, header + 4 * blockSize + lane, count);
    if (!hn::AllTrue(db, hn::Eq(landmarks, hn::Zero(db)))) {
        return kAshInvalidLandmark;
    }
    // Header arrays are [scale: B x f16][offset: B x f16][landmark: B x u8].
    // Disk f16 values are big-endian; Highway's x86 loads interpret native byte order.
    auto scaleBits = hn::LoadN(dh, reinterpret_cast<const uint16_t *>(header) + lane, count);
    auto offsetBits = hn::LoadN(dh, reinterpret_cast<const uint16_t *>(header + 2 * blockSize) + lane, count);
    scaleBits = hn::Or(hn::ShiftLeft<8>(scaleBits), hn::ShiftRight<8>(scaleBits));
    offsetBits = hn::Or(hn::ShiftLeft<8>(offsetBits), hn::ShiftRight<8>(offsetBits));
    const auto scale = AshPromoteHalf(d, scaleBits);
    const auto offset = AshPromoteHalf(d, offsetBits);
    const auto raw = hn::Add(hn::Add(hn::Mul(scale, dot), hn::Set(d, landmarkDot)), offset);
    const auto similarity = hn::Mul(hn::Add(hn::Set(d, 1.0f), raw), hn::Set(d, 0.5f));
    const auto result = hn::IfThenElse(hn::IsNaN(similarity), similarity, hn::Max(hn::Zero(d), similarity));
    hn::StoreN(result, d, out, count);
    return kAshSuccess;
}

template <class D>
HWY_INLINE int AshBlock32(D d, const unsigned char *codes, const float *lut, int groups, float landmarkDot,
                          float *out) {
    const hn::Rebind<uint8_t, D> db;
    const hn::Rebind<uint32_t, D> du;
    const auto mask = hn::Set(du, 15);
    // Both halves of B32 share the loaded lookup tables. Four independent
    // accumulators hide addition latency and avoid repeated LUT loads.
    auto lo0 = hn::Zero(d), hi0 = hn::Zero(d);
    auto lo1 = hn::Zero(d), hi1 = hn::Zero(d);
    const auto *row = codes;
    const auto *table = lut;
    const auto *end = lut + static_cast<size_t>(groups / 2) * 32;
    for (; table < end; table += 32, row += 32) {
        const auto c0 = hn::PromoteTo(du, hn::LoadU(db, row));
        const auto c1 = hn::PromoteTo(du, hn::LoadU(db, row + 16));
        const auto lowTable = hn::LoadU(d, table);
        const auto highTable = hn::LoadU(d, table + 16);
        const hn::Rebind<int32_t, D> di;
        lo0 = hn::Add(lo0, hn::TableLookupLanes(lowTable, hn::IndicesFromVec(d, hn::BitCast(di, hn::And(c0, mask)))));
        hi0 = hn::Add(hi0,
                      hn::TableLookupLanes(highTable, hn::IndicesFromVec(d, hn::BitCast(di, hn::ShiftRight<4>(c0)))));
        lo1 = hn::Add(lo1, hn::TableLookupLanes(lowTable, hn::IndicesFromVec(d, hn::BitCast(di, hn::And(c1, mask)))));
        hi1 = hn::Add(hi1,
                      hn::TableLookupLanes(highTable, hn::IndicesFromVec(d, hn::BitCast(di, hn::ShiftRight<4>(c1)))));
    }
    if (groups & 1) {
        const auto c0 = hn::PromoteTo(du, hn::LoadU(db, row));
        const auto c1 = hn::PromoteTo(du, hn::LoadU(db, row + 16));
        lo0 = hn::Add(lo0, AshLookup(d, table, hn::And(c0, mask)));
        lo1 = hn::Add(lo1, AshLookup(d, table, hn::And(c1, mask)));
    }
    const auto *header = codes + ((static_cast<size_t>(groups) + 1) / 2) * 32;
    int status = AshFinish(d, header, 32, 0, 16, landmarkDot, hn::Add(lo0, hi0), out);
    return status ? status : AshFinish(d, header, 32, 16, 16, landmarkDot, hn::Add(lo1, hi1), out + 16);
}

} // namespace

// Synchronous, allocation-free leaf. Buffer sizes and non-aliasing are checked by
// NativeASHKernels before FFM. The ordinary FFM path uses the same entry for larger
// legal inputs. FP32 reductions may differ in order from the Java scalar oracle.
int ash_score_neighborhood_c1_f32(const unsigned char *HWY_RESTRICT codes, const float *HWY_RESTRICT lut,
                                  float landmarkDot, int groups, int blockSize, int capacity, float *HWY_RESTRICT out) {
    if (codes == nullptr || lut == nullptr || out == nullptr || groups <= 0 || capacity <= 0 ||
        (blockSize != 8 && blockSize != 16 && blockSize != 32)) {
        return kAshInvalidShape;
    }
    const hn::CappedTag<float, 16> d;
    const hn::Rebind<uint8_t, decltype(d)> db;
    const hn::Rebind<uint32_t, decltype(d)> du;
    const auto mask = hn::Set(du, 15);
    const int lanes = static_cast<int>(hn::Lanes(d));
    const size_t bodyBytes = ((static_cast<size_t>(groups) + 1) / 2) * blockSize;
    const size_t blockBytes = bodyBytes + kAshHeaderBytesPerLane * blockSize;
    // Widen traversal arithmetic so the final base increment cannot overflow int.
    for (size_t base = 0; base < static_cast<size_t>(capacity); base += blockSize, codes += blockBytes) {
        const int count = std::min(blockSize, capacity - static_cast<int>(base));
        if constexpr (hn::MaxLanes(d) == 16) {
            if (blockSize == 32 && count == 32) {
                int status = AshBlock32(d, codes, lut, groups, landmarkDot, out + base);
                if (status)
                    return status;
                continue;
            }
        }
        for (int lane = 0; lane < count; lane += lanes) {
            const int active = std::min(lanes, count - lane);
            auto lo = hn::Zero(d), hi = hn::Zero(d);
            int g = 0;
            for (; g + 1 < groups; g += 2) {
                const auto c =
                    hn::PromoteTo(du, hn::LoadN(db, codes + static_cast<size_t>(g / 2) * blockSize + lane, active));
                lo = hn::Add(lo, AshLookup(d, lut + static_cast<size_t>(g) * 16, hn::And(c, mask)));
                hi = hn::Add(hi, AshLookup(d, lut + static_cast<size_t>(g + 1) * 16, hn::ShiftRight<4>(c)));
            }
            if (g < groups) {
                const auto c =
                    hn::PromoteTo(du, hn::LoadN(db, codes + static_cast<size_t>(g / 2) * blockSize + lane, active));
                lo = hn::Add(lo, AshLookup(d, lut + static_cast<size_t>(g) * 16, hn::And(c, mask)));
            }
            int status = AshFinish(d, codes + bodyBytes, blockSize, lane, active, landmarkDot, hn::Add(lo, hi),
                                   out + base + lane);
            if (status)
                return status;
        }
    }
    return kAshSuccess;
}

#endif // JVECTOR_ASH_NEIGHBORHOOD_INL_H
