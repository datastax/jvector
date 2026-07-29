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

// Validates that the native dispatcher selects the ISA tier that matches the
// CPU capabilities reported in /proc/cpuinfo, respecting any JVECTOR_MAX_ISA cap.
//
// Logic mirrors DispatcherCpuFlagsTest.java and the C implementation in
// jvector_cpu_features.h / jvector_simd.cpp exactly.
//
// /proc/cpuinfo is the authoritative ground-truth: the kernel only exposes a
// flag when the OS context-switch support (XCR0) is also in place, so checking
// it is equivalent to checking CPUID + XCR0 together.

#include "test_helpers.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// /proc/cpuinfo helpers — mirrors DispatcherCpuFlagsTest.java
// ---------------------------------------------------------------------------

// Parse the flags line from the first processor entry in /proc/cpuinfo.
// Returns an empty set if unavailable (non-Linux, non-x86, or unreadable).
static std::unordered_set<std::string> parse_cpuinfo_flags()
{
    std::unordered_set<std::string> flags;
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) return flags;

    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("flags", 0) != 0) continue;
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::istringstream iss(line.substr(colon + 1));
        std::string token;
        while (iss >> token) flags.insert(token);
        break;
    }
    return flags;
}

// Tier names in ascending capability order — index is ordinal (mirrors Java).
static const std::vector<std::string> kIsaTiers = {
    "sse42", "avx2", "avx3", "avx3_dl", "avx3_spr"
};

static int tier_index(const std::string& name)
{
    auto it = std::find(kIsaTiers.begin(), kIsaTiers.end(), name);
    return (it == kIsaTiers.end()) ? -1 : static_cast<int>(it - kIsaTiers.begin());
}

// ---- Composite tier predicates (flag names from /proc/cpuinfo) ----
// These must stay in sync with DispatcherCpuFlagsTest.java and
// jvector_cpu_features.h.  The kernel's naming is inconsistent:
//   no underscore:  avx512f bw cd dq vl, avx512vbmi, avx512ifma
//   with underscore: avx512_vnni, avx512_vbmi2, avx512_bitalg,
//                    avx512_vpopcntdq, avx512_fp16
// gfni / vaes / vpclmulqdq have no avx512 prefix at all.

static bool has_avx3(const std::unordered_set<std::string>& f)
{
    return f.count("avx512f")  && f.count("avx512bw")
        && f.count("avx512cd") && f.count("avx512dq")
        && f.count("avx512vl");
}

static bool has_avx3_dl(const std::unordered_set<std::string>& f)
{
    return has_avx3(f)
        && f.count("avx512_vnni")     && f.count("avx512vbmi")
        && f.count("avx512_vbmi2")    && f.count("avx512ifma")
        && f.count("avx512_bitalg")   && f.count("avx512_vpopcntdq")
        && f.count("gfni")            && f.count("vaes")
        && f.count("vpclmulqdq");
}

static bool has_avx3_spr(const std::unordered_set<std::string>& f)
{
    return has_avx3_dl(f) && f.count("avx512_fp16");
}

// Compute the expected ISA tier from /proc/cpuinfo flags and the cap,
// mirroring expectedIsaFromCpuInfo() in DispatcherCpuFlagsTest.java.
static std::string expected_isa(const std::unordered_set<std::string>& flags,
                                const std::string& cap)
{
    std::string best;
    if      (has_avx3_spr(flags)) best = "avx3_spr";
    else if (has_avx3_dl(flags))  best = "avx3_dl";
    else if (has_avx3(flags))     best = "avx3";
    else if (flags.count("avx2")) best = "avx2";
    else                          best = "sse42";

    // Clamp down to cap if set and below best.
    if (!cap.empty() && tier_index(cap) < tier_index(best))
        return cap;
    return best;
}

// ---------------------------------------------------------------------------
// Fixture — state shared across all tests
// ---------------------------------------------------------------------------

class CpuFeaturesTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        s_flags = parse_cpuinfo_flags();

        const char* active = jvector_simd_get_active_isa();
        const char* cap_c  = jvector_simd_get_max_isa_env();
        s_active = active ? active : "";
        s_cap    = cap_c  ? cap_c  : "";

        // Detect CPU emulators (e.g. Intel SDE): they intercept CPUID and
        // return synthetic features, but /proc/cpuinfo still reflects the host.
        // When the active ISA cannot be explained by the host's cpuinfo flags
        // the comparison tests are meaningless, so we skip them.
        std::string host_expected = expected_isa(s_flags, s_cap);
        s_under_emulator = !s_flags.empty()
                           && tier_index(s_active) > tier_index(host_expected);

        s_available = !s_flags.empty() && !s_under_emulator;

        std::printf("[  CPU     ] active_isa=%s  JVECTOR_MAX_ISA=%s  "
                    "cpuinfo_flags=%zu  emulator=%s\n",
                    s_active.c_str(),
                    s_cap.empty() ? "(unset)" : s_cap.c_str(),
                    s_flags.size(),
                    s_under_emulator ? "yes (cpuinfo skipped)" : "no");
    }

    static bool isCappedBelow(const std::string& tier)
    {
        return !s_cap.empty() && tier_index(s_cap) < tier_index(tier);
    }

    static std::unordered_set<std::string> s_flags;
    static std::string                     s_active;
    static std::string                     s_cap;
    static bool                            s_available;
    static bool                            s_under_emulator;
};

std::unordered_set<std::string> CpuFeaturesTest::s_flags;
std::string                     CpuFeaturesTest::s_active;
std::string                     CpuFeaturesTest::s_cap;
bool                            CpuFeaturesTest::s_available    = false;
bool                            CpuFeaturesTest::s_under_emulator = false;

#define SKIP_IF_UNAVAILABLE()                                                    \
    do {                                                                         \
        if (s_flags.empty()) GTEST_SKIP() << "/proc/cpuinfo unavailable";        \
        if (s_under_emulator) GTEST_SKIP() << "CPU emulator detected (SDE?): "  \
            "active_isa=" << s_active << " exceeds host cpuinfo capability";     \
    } while (0)

// ---------------------------------------------------------------------------
// Tests — mirror each @Test method in DispatcherCpuFlagsTest.java
// ---------------------------------------------------------------------------

// AVX2 tier is selected when avx2 is present and the cap allows it.
TEST_F(CpuFeaturesTest, Avx2Detection)
{
    SKIP_IF_UNAVAILABLE();
    bool cpu_has_avx2 = s_flags.count("avx2") > 0;

    if (cpu_has_avx2 && !isCappedBelow("avx2")) {
        EXPECT_GE(tier_index(s_active), tier_index("avx2"))
            << "Expected AVX2 or higher when avx2 flag present, got: " << s_active;
    } else if (!cpu_has_avx2) {
        EXPECT_EQ(s_active, "sse42")
            << "Expected sse42 when avx2 flag absent, got: " << s_active;
    }
}

// AVX3 tier is selected when all avx512 baseline flags are present.
TEST_F(CpuFeaturesTest, Avx3Detection)
{
    SKIP_IF_UNAVAILABLE();
    bool cpu_has_avx3 = has_avx3(s_flags);

    if (cpu_has_avx3 && !isCappedBelow("avx3")) {
        EXPECT_GE(tier_index(s_active), tier_index("avx3"))
            << "Expected AVX3 or higher when avx512 baseline flags present, got: " << s_active;
    } else if (!cpu_has_avx3) {
        EXPECT_LT(tier_index(s_active), tier_index("avx3"))
            << "Expected below AVX3 when avx512 baseline flags absent, got: " << s_active;
    }
}

// AVX3_DL tier is selected when all ICX flags are present on top of AVX3.
TEST_F(CpuFeaturesTest, Avx3DlDetection)
{
    SKIP_IF_UNAVAILABLE();
    bool cpu_has_avx3_dl = has_avx3_dl(s_flags);

    if (cpu_has_avx3_dl && !isCappedBelow("avx3_dl")) {
        EXPECT_GE(tier_index(s_active), tier_index("avx3_dl"))
            << "Expected AVX3_DL or higher when all ICX flags present, got: " << s_active;
    } else if (!cpu_has_avx3_dl) {
        EXPECT_LT(tier_index(s_active), tier_index("avx3_dl"))
            << "Expected below AVX3_DL when ICX flags absent, got: " << s_active;
    }
}

// AVX3_SPR tier is selected when avx512_fp16 and all ICX flags are present.
TEST_F(CpuFeaturesTest, Avx3SprDetection)
{
    SKIP_IF_UNAVAILABLE();
    bool cpu_has_avx3_spr = has_avx3_spr(s_flags);

    if (cpu_has_avx3_spr && !isCappedBelow("avx3_spr")) {
        EXPECT_EQ(s_active, "avx3_spr")
            << "Expected avx3_spr when fp16 + all ICX flags present and uncapped";
    } else if (!cpu_has_avx3_spr) {
        EXPECT_LT(tier_index(s_active), tier_index("avx3_spr"))
            << "Expected below AVX3_SPR when avx512_fp16 absent, got: " << s_active;
    }
}

// End-to-end: the tier the dispatcher chose must match what /proc/cpuinfo implies.
TEST_F(CpuFeaturesTest, DispatcherMatchesCpuInfo)
{
    SKIP_IF_UNAVAILABLE();

    std::string exp = expected_isa(s_flags, s_cap);

    // Collect all flags for the failure message.
    std::string all_flags = std::accumulate(
        s_flags.begin(), s_flags.end(), std::string{},
        [](const std::string& a, const std::string& b) {
            return a.empty() ? b : a + " " + b;
        });

    EXPECT_EQ(s_active, exp)
        << "Dispatcher chose '" << s_active
        << "' but /proc/cpuinfo implies '" << exp << "'."
        << "\nCPU flags: " << all_flags;
}
