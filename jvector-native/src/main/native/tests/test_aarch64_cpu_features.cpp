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

// Validates that the native dispatcher selects the correct AArch64 ISA tier.
//
// Ground truth on Linux:  /proc/cpuinfo "Features" line — the kernel only
//   exposes a feature token when the OS has set up the necessary context-switch
//   support, so this is the same authority as getauxval(AT_HWCAP).
//   Relevant tokens: "aes" (→ NEON), "sve" (→ SVE), "sve2" + "sveaes" (→ SVE2).
//
// Ground truth on macOS: sysctlbyname("hw.optional.arm.FEAT_AES") for NEON.
//   SVE and SVE2 are never available on any Apple Silicon through M4/A18.

#include "test_helpers.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Tier names in ascending capability order for AArch64.
static const std::vector<std::string> kIsaTiers = { "neon", "sve", "sve2" };

static int tier_index(const std::string& name)
{
    auto it = std::find(kIsaTiers.begin(), kIsaTiers.end(), name);
    return (it == kIsaTiers.end()) ? -1 : static_cast<int>(it - kIsaTiers.begin());
}

// Compute the expected ISA tier from the parsed feature set and any cap.
// Mirrors the logic in populate_cpu_features() / dispatch_kernels().
static std::string expected_isa(const std::unordered_set<std::string>& f,
                                const std::string& cap)
{
    std::string best;
    // SVE2: requires both "sve2" and "sveaes" (matches HWCAP2_SVE2 | HWCAP2_SVEAES).
    if (f.count("sve2") && f.count("sveaes")) best = "sve2";
    else if (f.count("sve"))                  best = "sve";
    else                                      best = "neon";

    if (!cap.empty() && tier_index(cap) < tier_index(best))
        return cap;
    return best;
}

// On macOS, detect NEON capability via sysctlbyname (no /proc/cpuinfo).
// SVE/SVE2 are never present on Apple Silicon, so "neon" is always the result.
#if defined(__APPLE__)
static std::string detect_host_isa_apple()
{
    int val = 0; size_t len = sizeof(val);
    if (sysctlbyname("hw.optional.arm.FEAT_AES", &val, &len, nullptr, 0) == 0 && val)
        return "neon";
    return "neon"; // baseline — every AArch64 Apple CPU has NEON
}
#endif

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class AArch64CpuFeaturesTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        const char* active = jvector_simd_get_active_isa();
        const char* cap_c  = jvector_simd_get_max_isa_env();
        s_active = active ? active : "";
        s_cap    = cap_c  ? cap_c  : "";

#if defined(__APPLE__)
        s_host_isa  = detect_host_isa_apple();
        s_available = true;
        std::printf("[  CPU     ] active_isa=%s  JVECTOR_MAX_ISA=%s  host_isa=%s  (macOS sysctl)\n",
                    s_active.c_str(),
                    s_cap.empty() ? "(unset)" : s_cap.c_str(),
                    s_host_isa.c_str());
#else
        s_cpuinfo   = parse_cpuinfo_line("Features");
        s_available = !s_cpuinfo.empty();

        if (s_available) {
            s_host_isa = expected_isa(s_cpuinfo, ""); // uncapped hardware capability
            // Collect feature string for diagnostics.
            s_feature_str = std::accumulate(
                s_cpuinfo.begin(), s_cpuinfo.end(), std::string{},
                [](const std::string& a, const std::string& b) {
                    return a.empty() ? b : a + " " + b;
                });
        }
        std::printf("[  CPU     ] active_isa=%s  JVECTOR_MAX_ISA=%s  host_isa=%s  "
                    "cpuinfo_features=%zu\n",
                    s_active.c_str(),
                    s_cap.empty() ? "(unset)" : s_cap.c_str(),
                    s_host_isa.c_str(),
                    s_cpuinfo.size());
#endif
    }

    static std::string                     s_active;
    static std::string                     s_cap;
    static std::string                     s_host_isa;
    static bool                            s_available;
    // Linux only:
    static std::unordered_set<std::string> s_cpuinfo;
    static std::string                     s_feature_str;
};

std::string                     AArch64CpuFeaturesTest::s_active;
std::string                     AArch64CpuFeaturesTest::s_cap;
std::string                     AArch64CpuFeaturesTest::s_host_isa;
bool                            AArch64CpuFeaturesTest::s_available    = false;
std::unordered_set<std::string> AArch64CpuFeaturesTest::s_cpuinfo;
std::string                     AArch64CpuFeaturesTest::s_feature_str;

#define SKIP_IF_UNAVAILABLE() \
    do { if (!s_available) GTEST_SKIP() << "/proc/cpuinfo unavailable"; } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The active ISA must be one of the three valid AArch64 tier names.
TEST_F(AArch64CpuFeaturesTest, ActiveIsaIsValidTier)
{
    EXPECT_GE(tier_index(s_active), 0)
        << "active_isa '" << s_active << "' is not a valid AArch64 tier "
        << "(expected one of: neon, sve, sve2)";
}

// NEON is always available on any AArch64 CPU.
TEST_F(AArch64CpuFeaturesTest, NeonAlwaysAvailable)
{
    SKIP_IF_UNAVAILABLE();
    EXPECT_GE(tier_index(s_host_isa), tier_index("neon"))
        << "Expected at least NEON on any AArch64 CPU";
}

// The dispatcher must not select a tier higher than the hardware supports.
TEST_F(AArch64CpuFeaturesTest, DispatcherDoesNotExceedHardware)
{
    SKIP_IF_UNAVAILABLE();
    EXPECT_LE(tier_index(s_active), tier_index(s_host_isa))
        << "Dispatcher selected '" << s_active
        << "' but host only supports up to '" << s_host_isa << "'"
        << "\nCPU features: " << s_feature_str;
}

// End-to-end: the tier the dispatcher chose must match what /proc/cpuinfo implies.
TEST_F(AArch64CpuFeaturesTest, DispatcherMatchesCpuInfo)
{
    SKIP_IF_UNAVAILABLE();
    std::string exp = expected_isa(s_cpuinfo, s_cap);
    EXPECT_EQ(s_active, exp)
        << "Dispatcher chose '" << s_active
        << "' but /proc/cpuinfo implies '" << exp << "'."
        << "\nCPU features: " << s_feature_str;
}

// When capped to "neon", the dispatcher must select the baseline tier.
TEST_F(AArch64CpuFeaturesTest, NeonCapForcesFallback)
{
    if (s_cap != "neon") GTEST_SKIP() << "JVECTOR_MAX_ISA != neon; skipping";
    EXPECT_EQ(s_active, "neon")
        << "Expected 'neon' when JVECTOR_MAX_ISA=neon, got: " << s_active;
}
