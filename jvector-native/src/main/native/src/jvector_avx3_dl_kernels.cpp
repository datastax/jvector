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
// Highway will select HWY_AVX3_DL as the static target.
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assert_hwy_targets.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace AVX3_DL {

// Blocked PQ code scan with VBMI: for each subspace, the 64 code bytes of a block index a
// 256-entry 8-bit table. Two vpermi2b lookups (entries 0..127 and 128..255) selected by the
// index's high bit, then widened into two u16 accumulators. About 7 instructions per subspace
// per 64 codes.
HWY_FLATTEN void pq_scan_blocked_u8(const unsigned char *blocks,
                                    size_t blockCount,
                                    int subspaceCount,
                                    const unsigned char *lut,
                                    unsigned short *out)
{
    const hn::ScalableTag<uint8_t>  d8;
    const hn::ScalableTag<uint16_t> d16;
    static_assert(hn::MaxLanes(hn::ScalableTag<uint8_t>()) == 64, "AVX3_DL build must have 64 u8 lanes");
    const auto v127 = hn::Set(d8, uint8_t{127});
    for (size_t b = 0; b < blockCount; b++) {
        const uint8_t *blk = blocks + b * (size_t)subspaceCount * 64;
        auto accLo = hn::Zero(d16);
        auto accHi = hn::Zero(d16);
        for (int m = 0; m < subspaceCount; m++) {
            const auto idx = hn::LoadU(d8, blk + (size_t)m * 64);
            const uint8_t *L = lut + (size_t)m * 256;
            const auto ind = hn::IndicesFromVec(d8, hn::And(idx, v127));
            const auto t0 = hn::TwoTablesLookupLanes(hn::LoadU(d8, L), hn::LoadU(d8, L + 64), ind);
            const auto t1 = hn::TwoTablesLookupLanes(hn::LoadU(d8, L + 128), hn::LoadU(d8, L + 192), ind);
            const auto v  = hn::IfThenElse(hn::Gt(idx, v127), t1, t0);
            accLo = hn::Add(accLo, hn::PromoteLowerTo(d16, v));
            accHi = hn::Add(accHi, hn::PromoteUpperTo(d16, v));
        }
        hn::StoreU(accLo, d16, out + b * 64);
        hn::StoreU(accHi, d16, out + b * 64 + 32);
    }
}

} // namespace AVX3_DL
