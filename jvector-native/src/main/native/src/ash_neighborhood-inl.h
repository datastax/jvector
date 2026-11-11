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

template <class D>
HWY_INLINE hn::VFromD<D> AshLookup(D d, const float *lut, hn::VFromD<hn::Rebind<uint32_t, D>> indices) {
    const hn::Rebind<int32_t, D> di;
#if HWY_TARGET <= HWY_AVX3
    // A register-wide 16-float permute is an AVX-512 instruction.
    if constexpr (hn::MaxLanes(d) == 16) {
        return hn::TableLookupLanes(hn::LoadU(d, lut), hn::IndicesFromVec(d, hn::BitCast(di, indices)));
    } else
#endif
    {
        // AVX2, NEON, SVE and scalar do not have a register-wide 16-float permute.
        return hn::GatherIndex(d, lut, hn::BitCast(di, indices));
    }
}

template <class D>
HWY_INLINE hn::VFromD<D> AshPromoteHalf(D d, hn::VFromD<hn::Rebind<uint16_t, D>> bits) {
    const hn::Rebind<hwy::float16_t, D> df16;
    auto value = hn::PromoteTo(d, hn::BitCast(df16, bits));
#if HWY_TARGET <= HWY_SSE4
    if constexpr (hn::MaxLanes(d) < 8) {
        // Highway's SSE software conversion assumes finite half values. Repair
        // exponent 31 explicitly; AVX2/AVX-512 use hardware F16 conversion and
        // NEON converts infinities and NaNs correctly.
        const hn::Rebind<uint32_t, D> du;
        const auto wide = hn::PromoteTo(du, bits);
        const auto exponent = hn::And(wide, hn::Set(du, 0x7c00));
        const auto special = hn::Eq(exponent, hn::Set(du, 0x7c00));
        const auto sign = hn::ShiftLeft<16>(hn::And(wide, hn::Set(du, 0x8000)));
        const auto mantissa = hn::ShiftLeft<13>(hn::And(wide, hn::Set(du, 0x03ff)));
        const auto ieee = hn::BitCast(d, hn::Or(hn::Or(sign, mantissa), hn::Set(du, 0x7f800000)));
        value = hn::IfThenElse(hn::RebindMask(d, special), ieee, value);
    }
#endif
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
    scaleBits = hn::ReverseLaneBytes(scaleBits);
    offsetBits = hn::ReverseLaneBytes(offsetBits);
    const auto scale = AshPromoteHalf(d, scaleBits);
    const auto offset = AshPromoteHalf(d, offsetBits);
    const auto raw = hn::Add(hn::MulAdd(scale, dot, hn::Set(d, landmarkDot)), offset);
    const auto half = hn::Set(d, 0.5f);
    const auto similarity = hn::MulAdd(raw, half, half);
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

#ifndef JVECTOR_ASH_INT8_NEIGHBORHOOD_ONLY
// Canonical vectors pack consecutive groups into low/high nibbles of each byte.
// Gather from the same query LUT as the neighborhood kernel, without gathering
// projected query coordinates or repeatedly decoding signed-magnitude components.
float ash_canonical_lut_dot_f32(const unsigned char *HWY_RESTRICT code,
                               const float *HWY_RESTRICT lut, int groups) {
    if (code == nullptr || lut == nullptr || groups <= 0) return NAN;
    const hn::CappedTag<float, 16> d;
    const hn::Rebind<uint8_t, decltype(d)> db;
    const hn::Rebind<int32_t, decltype(d)> di;
    const auto offsets = hn::Mul(hn::Iota(di, 0), hn::Set(di, 32));
    const auto mask = hn::Set(di, 15);
    const size_t lanes = hn::Lanes(d);
    const size_t pairs = static_cast<size_t>(groups) / 2;
    auto low = hn::Zero(d), high = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= pairs; i += lanes) {
        const auto packed = hn::PromoteTo(di, hn::LoadU(db, code + i));
        const auto lowIndex = hn::Add(offsets, hn::And(packed, mask));
        const auto highIndex = hn::Add(offsets, hn::Add(hn::Set(di, 16), hn::ShiftRight<4>(packed)));
        low = hn::Add(low, hn::GatherIndex(d, lut + i * 32, lowIndex));
        high = hn::Add(high, hn::GatherIndex(d, lut + i * 32, highIndex));
    }
    float sum = hn::ReduceSum(d, hn::Add(low, high));
    for (; i < pairs; i++) {
        sum += lut[i * 32 + (code[i] & 15)];
        sum += lut[i * 32 + 16 + (code[i] >> 4)];
    }
    if (groups & 1) sum += lut[pairs * 32 + (code[pairs] & 15)];
    return sum;
}

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
#if HWY_TARGET <= HWY_AVX3
        if constexpr (hn::MaxLanes(d) == 16) {
            if (blockSize == 32 && count == 32) {
                int status = AshBlock32(d, codes, lut, groups, landmarkDot, out + base);
                if (status)
                    return status;
                continue;
            }
        }
#endif
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

#endif // JVECTOR_ASH_INT8_NEIGHBORHOOD_ONLY

namespace {
// Offset-binary neighborhoods (3 and 5 to 9 bits). The body is [unit][lane][bits bytes]: one unit is
// eight dimensions, so each lane's eight fields fill exactly `bits` bytes. A vector group holds one
// 64-bit word per lane. Bytes are gathered into each lane's word, every field is rotated into its own
// byte, and products with an INT8 query digit accumulate four dimensions per 32-bit lane.
constexpr int kAshInt8MaxBits = 9;

// Eight consecutive query digits as one 64-bit word, for broadcasting to every lane's word.
HWY_INLINE uint64_t AshQueryWord(const int8_t *digits) {
    uint64_t word;
    hwy::CopyBytes<8>(digits, &word);
    return word;
}

// Per-lane exact sums: each lane's two 32-bit halves hold dimensions 0-3 and 4-7 of every unit.
template <class D64, class D32>
HWY_INLINE hn::VFromD<hn::Rebind<float, D64>> AshLaneSums(D64 d64, D32 /*d32*/, hn::VFromD<D32> acc) {
    const hn::RebindToSigned<D64> di64;
    const hn::Rebind<int32_t, D64> di32;
    const auto wide = hn::BitCast(di64, acc);
    const auto sum = hn::Add(hn::ShiftRight<32>(hn::ShiftLeft<32>(wide)), hn::ShiftRight<32>(wide));
    return hn::ConvertTo(hn::Rebind<float, D64>(), hn::DemoteTo(di32, sum));
}

// Unpacks one lane group's fields for one unit and accumulates their products with the query word.
// Nine-bit fields add their top bits to `top`, which the caller scales by 256.
template <class D64, class D8, class D32, class VI8, class Idx>
HWY_INLINE void AshAccumulateFields(D64 d64, D8 d8, D32 d32, int bits, size_t lanes, const unsigned char *src,
                                    VI8 qv, Idx idxA, Idx idxB, hn::VFromD<D8> rotLo,
                                    hn::VFromD<D8> rotHi, hn::VFromD<D8> lowMask, hn::MFromD<D8> lastField,
                                    hn::VFromD<D32> &acc, hn::VFromD<D32> &top) {
    if (bits <= 8) {
        // A group needs lanes * bits <= Lanes(d8) bytes; later bytes are ignored, and the header after the
        // body keeps the full-width load inside the block.
        auto fields = hn::LoadU(d8, src);
        if (bits < 8) {
            fields = hn::TableLookupLanes(fields, idxA);
            fields = hn::And(hn::BitCast(d8, hn::MultiRotateRight(hn::BitCast(d64, fields), rotLo)), lowMask);
        }
        acc = hn::SumOfMulQuadAccumulate(d32, fields, qv, acc);
        return;
    }
    // lanes * 9 bytes; the second table may extend past the block, so load it bounded.
    const size_t span = lanes * static_cast<size_t>(bits);
    const auto t0 = hn::LoadN(d8, src, std::min(span, hn::Lanes(d8)));
    const auto t1 = hn::LoadN(d8, src + hn::Lanes(d8), span > hn::Lanes(d8) ? span - hn::Lanes(d8) : 0);
    const auto viewA = hn::BitCast(d64, hn::TwoTablesLookupLanes(d8, t0, t1, idxA));
    const auto viewB = hn::BitCast(d64, hn::TwoTablesLookupLanes(d8, t0, t1, idxB));
    const auto low = hn::IfThenElse(lastField, hn::BitCast(d8, hn::MultiRotateRight(viewB, rotLo)),
                                    hn::BitCast(d8, hn::MultiRotateRight(viewA, rotLo)));
    const auto high = hn::And(hn::IfThenElse(lastField, hn::BitCast(d8, hn::MultiRotateRight(viewB, rotHi)),
                                             hn::BitCast(d8, hn::MultiRotateRight(viewA, rotHi))),
                              hn::Set(d8, uint8_t{1}));
    acc = hn::SumOfMulQuadAccumulate(d32, low, qv, acc);
    top = hn::SumOfMulQuadAccumulate(d32, high, qv, top);
}
} // namespace

// Whole-neighborhood C=1 ASH for offset-binary codes with an INT8 query: per lane,
// dot = (<q, u> - center) * invScale, then the header terms and similarity. q holds units * 8 entries
// (zero past the projected dimension). Returns kAshSuccess, kAshInvalidLandmark or kAshInvalidShape.
int ash_score_neighborhood_i8_c1(const unsigned char *HWY_RESTRICT codes, const int8_t *HWY_RESTRICT q,
                                 float invScale, float center, float landmarkDot, int bits, int units,
                                 int blockSize, int capacity, float *HWY_RESTRICT out) {
    if (codes == nullptr || q == nullptr || out == nullptr || units <= 0 || capacity <= 0 || bits < 3 ||
        bits > kAshInt8MaxBits || bits == 4 || (blockSize != 8 && blockSize != 16 && blockSize != 32)) {
        return kAshInvalidShape;
    }
    // At most eight 64-bit lanes per group, so groups divide every block size.
    const hn::CappedTag<uint64_t, 8> d64;
    const hn::Repartition<uint8_t, decltype(d64)> d8;
    const hn::Repartition<int8_t, decltype(d64)> di8;
    const hn::Repartition<int32_t, decltype(d64)> d32;
    const hn::Rebind<float, decltype(d64)> df;
    const size_t lanes = hn::Lanes(d64);
    const size_t b = static_cast<size_t>(bits);
    const size_t rowBytes = b * static_cast<size_t>(blockSize);
    const size_t bodyBytes = static_cast<size_t>(units) * rowBytes;
    const size_t blockBytes = bodyBytes + kAshHeaderBytesPerLane * blockSize;

    // Gather and rotation controls. Lane l's bytes start at l * bits; field k starts at bit k * bits.
    // Nine-bit fields span nine bytes per lane: two eight-byte views (bytes 0-7 and 1-8) supply the low
    // byte and top bit of fields 0-6 and of field 7 respectively.
    HWY_ALIGN uint8_t gatherA[64], gatherB[64], rotateLo[64], rotateHi[64];
    for (size_t l = 0; l < lanes; l++) {
        for (size_t k = 0; k < 8; k++) {
            const size_t i = 8 * l + k;
            if (bits <= 8) {
                gatherA[i] = static_cast<uint8_t>(l * b + std::min(k, b - 1));
                gatherB[i] = gatherA[i];
                rotateLo[i] = static_cast<uint8_t>(k * b);
                rotateHi[i] = 0;
            } else {
                gatherA[i] = static_cast<uint8_t>(l * b + k);
                gatherB[i] = static_cast<uint8_t>(l * b + 1 + k);
                rotateLo[i] = static_cast<uint8_t>(k < 7 ? k * b : 55);
                rotateHi[i] = static_cast<uint8_t>(k < 7 ? k * b + 8 : 63);
            }
        }
    }
    const auto lowMask = hn::Set(d8, static_cast<uint8_t>(bits >= 8 ? 0xFF : (1u << bits) - 1));
    const auto rotLo = hn::Load(d8, rotateLo), rotHi = hn::Load(d8, rotateHi);
    // Byte 7 of every lane's word takes field 7 from the shifted view.
    const auto lastField = hn::Eq(hn::And(hn::Iota(d8, 0), hn::Set(d8, uint8_t{7})), hn::Set(d8, uint8_t{7}));
    const auto idxA = hn::IndicesFromVec(d8, hn::Load(d8, gatherA));
    const auto idxB = hn::IndicesFromVec(d8, hn::Load(d8, gatherB));

    // Up to four lane groups share each pass over a block: every unit row is read once, contiguously, and
    // its query word is broadcast once. With eight 64-bit lanes, one pass covers a 32-lane block.
    const size_t groupBytes = lanes * b;
    for (size_t base = 0; base < static_cast<size_t>(capacity); base += blockSize, codes += blockBytes) {
        const int count = std::min(blockSize, capacity - static_cast<int>(base));
        const size_t groups = (static_cast<size_t>(count) + lanes - 1) / lanes;
        for (size_t first = 0; first < groups; first += 4) {
            const size_t n = std::min<size_t>(4, groups - first);
            auto a0 = hn::Zero(d32), a1 = hn::Zero(d32), a2 = hn::Zero(d32), a3 = hn::Zero(d32);
            auto t0 = hn::Zero(d32), t1 = hn::Zero(d32), t2 = hn::Zero(d32), t3 = hn::Zero(d32);
            const unsigned char *row = codes + first * groupBytes;
            for (int u = 0; u < units; u++, row += rowBytes) {
                const auto qv = hn::BitCast(di8, hn::Set(d64, AshQueryWord(q + 8 * u)));
                AshAccumulateFields(d64, d8, d32, bits, lanes, row, qv, idxA, idxB, rotLo, rotHi, lowMask, lastField,
                                    a0, t0);
                if (n > 1)
                    AshAccumulateFields(d64, d8, d32, bits, lanes, row + groupBytes, qv, idxA, idxB, rotLo, rotHi,
                                        lowMask, lastField, a1, t1);
                if (n > 2)
                    AshAccumulateFields(d64, d8, d32, bits, lanes, row + 2 * groupBytes, qv, idxA, idxB, rotLo,
                                        rotHi, lowMask, lastField, a2, t2);
                if (n > 3)
                    AshAccumulateFields(d64, d8, d32, bits, lanes, row + 3 * groupBytes, qv, idxA, idxB, rotLo,
                                        rotHi, lowMask, lastField, a3, t3);
            }
            // Vectors are passed by value (no arrays), which scalable-vector targets require.
            const auto finish = [&](size_t i, hn::VFromD<decltype(d32)> sum, hn::VFromD<decltype(d32)> top) -> int {
                const auto acc = bits > 8 ? hn::Add(sum, hn::ShiftLeft<8>(top)) : sum;
                const auto dot = hn::Mul(hn::Sub(AshLaneSums(d64, d32, acc), hn::Set(df, center)), hn::Set(df, invScale));
                const int lane = static_cast<int>((first + i) * lanes);
                const int active = std::min(static_cast<int>(lanes), count - lane);
                return AshFinish(df, codes + bodyBytes, blockSize, lane, active, landmarkDot, dot, out + base + lane);
            };
            int status = finish(0, a0, t0);
            if (!status && n > 1) status = finish(1, a1, t1);
            if (!status && n > 2) status = finish(2, a2, t2);
            if (!status && n > 3) status = finish(3, a3, t3);
            if (status) return status;
        }
    }
    return kAshSuccess;
}

#ifndef JVECTOR_ASH_INT8_NEIGHBORHOOD_ONLY
// A 4-bit nibble represents one coordinate, so a full query LUT is 16 times
// larger than the projection. Decode consecutive coordinates in registers and
// keep the projection access contiguous instead of gathering a large table.
float ash_projection_dot_4bit_f32(const float *HWY_RESTRICT query,
                                  const unsigned char *HWY_RESTRICT code, int dimensions) {
    if (query == nullptr || code == nullptr || dimensions < 0) return NAN;
    const hn::CappedTag<float, 16> d;
    const hn::Rebind<uint8_t, decltype(d)> db;
    const hn::Rebind<uint32_t, decltype(d)> du;
    const size_t lanes = hn::Lanes(d);
    const size_t count = static_cast<size_t>(dimensions);
#if HWY_TARGET <= HWY_AVX3
    // Exact signed-magnitude levels: keep them in one register instead of
    // converting magnitudes and constructing sign masks for every vector.
    alignas(64) static constexpr float levels[16] = {
        -.5f, -1.5f, -2.5f, -3.5f, -4.5f, -5.5f, -6.5f, -7.5f,
         .5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  6.5f,  7.5f};
    const auto table = hn::LoadU(d, levels);
#else
    const auto magnitudeMask = hn::Set(du, 7);
    const auto signMask = hn::Set(du, 8);
    const auto half = hn::Set(d, 0.5f);
#endif
    auto low = hn::Zero(d), high = hn::Zero(d);
    size_t i = 0;
    for (; i + 2 * lanes <= count; i += 2 * lanes) {
        const auto packed = hn::PromoteTo(du, hn::LoadU(db, code + i / 2));
        const auto hiCode = hn::ShiftRight<4>(packed);
#if HWY_TARGET <= HWY_AVX3
        // VPERMPS uses only the low four index bits. Use the intrinsic to avoid
        // a redundant mask and Highway IndicesFromVec debug checks on upper bits.
        const auto loValue = hn::VFromD<decltype(d)>{_mm512_permutexvar_ps(packed.raw, table.raw)};
        const auto hiValue = hn::VFromD<decltype(d)>{_mm512_permutexvar_ps(hiCode.raw, table.raw)};
#else
        auto loValue = hn::Add(hn::ConvertTo(d, hn::And(packed, magnitudeMask)), half);
        auto hiValue = hn::Add(hn::ConvertTo(d, hn::And(hiCode, magnitudeMask)), half);
        loValue = hn::IfThenElse(hn::RebindMask(d, hn::Eq(hn::And(packed, signMask), hn::Zero(du))),
                                hn::Neg(loValue), loValue);
        hiValue = hn::IfThenElse(hn::RebindMask(d, hn::Eq(hn::And(hiCode, signMask), hn::Zero(du))),
                                hn::Neg(hiValue), hiValue);
#endif
        hn::VFromD<decltype(d)> qLow, qHigh;
        hn::LoadInterleaved2(d, query + i, qLow, qHigh);
        low = hn::MulAdd(qLow, loValue, low);
        high = hn::MulAdd(qHigh, hiValue, high);
    }
    float sum = hn::ReduceSum(d, hn::Add(low, high));
    for (; i < count; i++) {
        const unsigned field = (code[i / 2] >> ((i & 1) * 4)) & 15;
        const float value = static_cast<float>(field & 7) + 0.5f;
        sum += query[i] * ((field & 8) ? value : -value);
    }
    return sum;
}
#endif // JVECTOR_ASH_INT8_NEIGHBORHOOD_ONLY

#endif // JVECTOR_ASH_NEIGHBORHOOD_INL_H
