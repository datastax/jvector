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

#ifndef JVECTOR_ASH_SYMMETRIC_INL_H
#define JVECTOR_ASH_SYMMETRIC_INL_H

// Included within JV_ISA. All sizes/pointers are checked by the Java binding.
// Products of doubled signed-magnitude components are exact integers; divide
// by four only once, after accumulation. Padding bits never contribute.
namespace {
HWY_INLINE int AshDoubledComponent(const uint8_t *code, int j, int bits) {
    const int perByte = 8 / bits;
    const int field = code[j / perByte] >> ((j % perByte) * bits);
    const int sign = 1 << (bits - 1);
    const int magnitude = 2 * (field & (sign - 1)) + 1;
    return (field & sign) ? magnitude : -magnitude;
}

HWY_INLINE float AshBinaryPair(const uint64_t *a, const uint64_t *b, int dimensions) {
    const hn::ScalableTag<uint64_t> d;
    const size_t lanes = hn::Lanes(d), words = static_cast<size_t>(dimensions) / 64;
    auto counts = hn::Zero(d);
    // Integer lanes can hold all mismatches even at the maximum Java dimension.
    // Full vectors use plain loads; only the last partial vector is masked.
    size_t word = 0;
    for (; word + lanes <= words; word += lanes) {
        counts = hn::Add(counts, hn::PopulationCount(hn::Xor(hn::LoadU(d, a + word), hn::LoadU(d, b + word))));
    }
    if (word < words) {
        const size_t n = words - word;
        counts = hn::Add(counts, hn::PopulationCount(hn::Xor(hn::LoadN(d, a + word, n), hn::LoadN(d, b + word, n))));
    }
    uint64_t mismatches = hn::ReduceSum(d, counts);
    const int tail = dimensions & 63;
    if (tail) mismatches += hwy::PopCount((a[words] ^ b[words]) & ((uint64_t{1} << tail) - 1));
    return static_cast<float>(dimensions) - 2.0f * static_cast<float>(mismatches);
}

// sign*(1+2*ma)*(1+2*mb) for each 2-bit field pair in av and bv, less one per field, evaluated
// with weighted popcounts and summed per 64-bit lane.
template <class D>
HWY_INLINE hn::VFromD<hn::RebindToSigned<D>> AshTwoBitDelta(D d, hn::VFromD<D> av, hn::VFromD<D> bv) {
    const auto slots = hn::Set(d, uint64_t{0x5555555555555555});
    const auto neg = hn::And(hn::ShiftRight<1>(hn::Xor(av, bv)), slots);
    const auto ma = hn::And(av, slots), mb = hn::And(bv, slots), both = hn::And(ma, mb);
    auto delta = hn::ShiftLeft<1>(hn::Sub(
            hn::Add(hn::PopulationCount(ma), hn::PopulationCount(mb)), hn::PopulationCount(neg)));
    delta = hn::Sub(delta, hn::ShiftLeft<2>(hn::Add(
            hn::PopulationCount(hn::And(ma, neg)), hn::PopulationCount(hn::And(mb, neg)))));
    delta = hn::Add(delta, hn::ShiftLeft<2>(hn::PopulationCount(both)));
    delta = hn::Sub(delta, hn::ShiftLeft<3>(hn::PopulationCount(hn::And(both, neg))));
    return hn::BitCast(hn::RebindToSigned<D>(), delta);
}

HWY_INLINE float AshTwoBitPair(const uint8_t *a, const uint8_t *b, int dimensions) {
    const hn::ScalableTag<uint64_t> d;
    const size_t lanes = hn::Lanes(d), words = static_cast<size_t>(dimensions) / 32;
    // Canonical codes are little-endian packed fields. LoadU/LoadN permit
    // byte alignment on all supported little-endian production targets.
    const auto *wa = reinterpret_cast<const uint64_t *>(a), *wb = reinterpret_cast<const uint64_t *>(b);
    // Keep partial sums in registers; one horizontal reduction per pair. Full vectors use
    // plain loads; only the last partial vector is masked.
    auto deltas = hn::Zero(hn::RebindToSigned<decltype(d)>());
    size_t word = 0;
    for (; word + lanes <= words; word += lanes) {
        deltas = hn::Add(deltas, AshTwoBitDelta(d, hn::LoadU(d, wa + word), hn::LoadU(d, wb + word)));
    }
    if (word < words) {
        const size_t n = words - word;
        deltas = hn::Add(deltas, AshTwoBitDelta(d, hn::LoadN(d, wa + word, n), hn::LoadN(d, wb + word, n)));
    }
    const hn::RebindToSigned<decltype(d)> di;
    int64_t total = static_cast<int64_t>(words) * 32 + hn::ReduceSum(di, deltas);
    // The last partial word: copy only its code bytes and clear the padding fields, which then
    // contribute nothing. One word of popcounts instead of one decode per leftover dimension.
    const int tail = dimensions & 31;
    if (tail) {
        // A masked load of the code bytes only; it never reads past the code.
        const hn::FixedTag<uint8_t, 8> d8;
        const hn::FixedTag<uint64_t, 1> d64;
        const size_t tailBytes = (static_cast<size_t>(tail) + 3) / 4;
        uint64_t ta = hn::GetLane(hn::BitCast(d64, hn::LoadN(d8, a + words * 8, tailBytes)));
        uint64_t tb = hn::GetLane(hn::BitCast(d64, hn::LoadN(d8, b + words * 8, tailBytes)));
        const uint64_t fields = (uint64_t{1} << (2 * tail)) - 1, slot = uint64_t{0x5555555555555555};
        ta &= fields;
        tb &= fields;
        const uint64_t neg = ((ta ^ tb) >> 1) & slot, ma = ta & slot, mb = tb & slot, both = ma & mb;
        int64_t delta = 2 * (static_cast<int64_t>(hwy::PopCount(ma)) + hwy::PopCount(mb) - hwy::PopCount(neg));
        delta -= 4 * (static_cast<int64_t>(hwy::PopCount(ma & neg)) + hwy::PopCount(mb & neg));
        delta += 4 * static_cast<int64_t>(hwy::PopCount(both));
        delta -= 8 * static_cast<int64_t>(hwy::PopCount(both & neg));
        total += tail + delta;
    }
    return static_cast<float>(total) * 0.25f;
}

// The last partial vector of 4-bit codes, n bytes. Kept out of line so the full-vector loop
// compiles as before. LoadN zero-fills the lanes past the code; a zero byte decodes to two
// (+1)*(+1) products, which are subtracted.
HWY_NOINLINE int64_t AshFourBitPartial(const uint8_t *a, const uint8_t *b, size_t n) {
    const hn::ScalableTag<int32_t> di;
    const hn::Repartition<uint8_t, decltype(di)> db;
    const hn::RebindToSigned<decltype(db)> ds;
    const auto av = hn::LoadN(db, a, n), bv = hn::LoadN(db, b, n);
    auto sum = hn::Zero(di);
    for (int shift = 0; shift < 8; shift += 4) {
        const auto ax = hn::ShiftRightSame(av, shift), bx = hn::ShiftRightSame(bv, shift);
        const auto am = hn::Add(hn::ShiftLeft<1>(hn::And(ax, hn::Set(db, 7))), hn::Set(db, 1));
        const auto bm = hn::BitCast(ds, hn::Add(hn::ShiftLeft<1>(hn::And(bx, hn::Set(db, 7))), hn::Set(db, 1)));
        const auto sameSign = hn::Eq(hn::And(hn::Xor(ax, bx), hn::Set(db, 8)), hn::Zero(db));
        const auto signedB = hn::IfThenElse(hn::RebindMask(ds, sameSign), bm, hn::Neg(bm));
        sum = hn::SumOfMulQuadAccumulate(di, am, signedB, sum);
    }
    return hn::ReduceSum(di, sum) - 2 * static_cast<int64_t>(hn::Lanes(db) - n);
}

HWY_INLINE float AshFourBitPair(const uint8_t *a, const uint8_t *b, int dimensions) {
    int64_t total = 0;
    const hn::ScalableTag<int32_t> di;
    const hn::Repartition<uint8_t, decltype(di)> db;
    const hn::RebindToSigned<decltype(db)> ds;
    const size_t bytes = static_cast<size_t>(dimensions) / 2, lanes = hn::Lanes(db);
    // One accumulator for the whole pair. Each pass adds at most 8 * 15 * 15 = 1800 per int32
    // lane, so a lane overflows only past about a million vectors, far beyond any Java code.
    auto sum = hn::Zero(di);
    size_t offset = 0;
    for (; offset + lanes <= bytes; offset += lanes) {
        const auto av = hn::LoadU(db, a + offset), bv = hn::LoadU(db, b + offset);
        for (int shift = 0; shift < 8; shift += 4) {
            const auto ax = hn::ShiftRightSame(av, shift), bx = hn::ShiftRightSame(bv, shift);
            const auto am = hn::Add(hn::ShiftLeft<1>(hn::And(ax, hn::Set(db, 7))), hn::Set(db, 1));
            const auto bm = hn::BitCast(ds, hn::Add(hn::ShiftLeft<1>(hn::And(bx, hn::Set(db, 7))), hn::Set(db, 1)));
            const auto sameSign = hn::Eq(hn::And(hn::Xor(ax, bx), hn::Set(db, 8)), hn::Zero(db));
            const auto signedB = hn::IfThenElse(hn::RebindMask(ds, sameSign), bm, hn::Neg(bm));
            // VNNI when supported; Highway supplies widening multiply otherwise.
            sum = hn::SumOfMulQuadAccumulate(di, am, signedB, sum);
        }
    }
    total += hn::ReduceSum(di, sum);
    if (offset < bytes) {
        total += AshFourBitPartial(a + offset, b + offset, bytes - offset);
        offset = bytes;
    }
    for (int j = static_cast<int>(2 * offset); j < dimensions; ++j) total += AshDoubledComponent(a, j, 4) * AshDoubledComponent(b, j, 4);
    return static_cast<float>(total) * 0.25f;
}
} // namespace

float ash_symmetric_dot(const void *a, const void *b, int dimensions, int bits) {
    if (bits == 1) return AshBinaryPair(static_cast<const uint64_t *>(a), static_cast<const uint64_t *>(b), dimensions);
    if (bits == 2) return AshTwoBitPair(static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b), dimensions);
    return AshFourBitPair(static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b), dimensions);
}

// Scores one code against count contiguous codes stride bytes apart, each exactly as ash_symmetric_dot.
void ash_symmetric_dot_rows(const void *query, const void *rows, int stride, int count, int dimensions, int bits,
                            float *HWY_RESTRICT out) {
    const auto *row = static_cast<const uint8_t *>(rows);
    for (int i = 0; i < count; ++i) out[i] = ash_symmetric_dot(query, row + static_cast<size_t>(i) * stride, dimensions, bits);
}

#ifndef JVECTOR_ASH_PAIR_ONLY
// Exact signed-short LUTs, unlike the IVF scan's approximate uint8 LUTs. Widen
// before accumulation. At most 8192 groups enter an int32 subtotal, then spill
// into int64 lane totals so arbitrary short entries and long codes cannot wrap.
void ash_symmetric_lut_i16(const uint8_t *HWY_RESTRICT codes, const int16_t *HWY_RESTRICT lut,
                           int groups, int stride, int lane, int count, float *HWY_RESTRICT out) {
    const hn::CappedTag<int32_t, 16> d;
    const hn::Rebind<uint8_t, decltype(d)> db;
    const hn::Rebind<int16_t, decltype(d)> ds;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (size_t base = 0; base < static_cast<size_t>(count); base += lanes) {
        const int active = static_cast<int>(std::min(static_cast<size_t>(lanes), static_cast<size_t>(count) - base));
        int64_t totals[16] = {};
        for (int first = 0; first < groups;) {
            const int end = first + std::min(8192, groups - first);
            auto sum = hn::Zero(d);
            for (int g = first; g < end; ++g) {
                auto packed = hn::PromoteTo(d, hn::LoadN(db, codes + static_cast<size_t>(g / 2) * stride + lane + base, active));
                const auto indices = hn::And(hn::ShiftRightSame(packed, 4 * (g & 1)), hn::Set(d, 15));
#if HWY_TARGET <= HWY_AVX3
                if constexpr (hn::MaxLanes(d) == 16) {
                    const auto table = hn::PromoteTo(d, hn::LoadU(ds, lut + static_cast<size_t>(g) * 32));
                    sum = hn::Add(sum, hn::TableLookupLanes(table, hn::IndicesFromVec(d, indices)));
                } else
#endif
                {
                    int32_t table[16];
                    for (int i = 0; i < 16; ++i) table[i] = lut[static_cast<size_t>(g) * 32 + i];
                    sum = hn::Add(sum, hn::GatherIndex(d, table, indices));
                }
            }
            int32_t partial[16];
            hn::StoreU(sum, d, partial);
            for (int i = 0; i < active; ++i) totals[i] += partial[i];
            first = end;
        }
        for (int i = 0; i < active; ++i) out[base + i] = static_cast<float>(totals[i]) * 0.25f;
    }
}
#endif
#endif
