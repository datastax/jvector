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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that the native dispatcher selects the ISA tier that matches the CPU
 * capabilities reported in /proc/cpuinfo, respecting any JVECTOR_MAX_ISA cap.
 *
 * <p>The tier-promotion logic here mirrors the C implementation in:
 * <ul>
 *   <li>{@code jvector_cpu_features.h} – composite tier flag rules
 *   <li>{@code jvector_simd.cpp}       – {@code dispatch_kernels()} selection order
 * </ul>
 *
 * <p>These tests only run on Linux x86_64 with the native library available.
 */
class DispatcherCpuFlagsTest {

    // ISA tier names in ascending capability order — index doubles as ordinal.
    private static final List<String> ISA_TIERS =
            List.of("sse42", "avx2", "avx3", "avx3_dl", "avx3_spr");

    // Loaded once for the whole test class
    private static Set<String> cpuFlags;
    // The cap imposed by JVECTOR_MAX_ISA as reported by the native library, or null if unset/unrecognised.
    private static String maxIsaCap;
    // Backed by the loaded native library; used to call getActiveIsa() and getMaxIsaEnv().
    private static NativeVectorUtilSupport nativeSupport;

    @BeforeAll
    static void loadLibraryAndParseCpuInfo() throws IOException {
        // Only meaningful on x86_64 Linux; skip gracefully everywhere else
        String arch = System.getProperty("os.arch", "");
        Assumptions.assumeTrue(
                arch.equals("amd64") || arch.equals("x86_64"),
                "Dispatcher ISA tests only run on x86_64");

        Assumptions.assumeTrue(
                Files.exists(Path.of("/proc/cpuinfo")),
                "Dispatcher ISA tests require /proc/cpuinfo");

        boolean libraryLoaded = LibraryLoader.loadJvector();
        Assumptions.assumeTrue(libraryLoaded, "Native jvector library not available");

        nativeSupport = new NativeVectorUtilSupport();
        cpuFlags = parseCpuInfoFlags();

        // Ask the native library what cap it actually applied — this is the
        // value read_max_isa() recognised at static-init time, or null if the
        // variable was absent/unrecognised.
        maxIsaCap = nativeSupport.getMaxIsaEnv();

        System.out.printf("[DispatcherCpuFlagsTest] active ISA : %s%n", nativeSupport.getActiveIsa());
        System.out.printf("[DispatcherCpuFlagsTest] JVECTOR_MAX_ISA : %s%n",
                maxIsaCap != null ? maxIsaCap : "(unset)");
    }

    // -------------------------------------------------------------------------
    // Individual ISA feature tests
    // -------------------------------------------------------------------------

    /**
     * AVX2 is required before any AVX-512 tier can be selected.
     * /proc/cpuinfo flag: {@code avx2}
     */
    @Test
    void testAvx2Detection() {
        boolean cpuHasAvx2 = cpuFlags.contains("avx2");
        String activeIsa = nativeSupport.getActiveIsa();
        if (cpuHasAvx2 && !isCappedBelow("avx2")) {
            // At minimum, AVX2 tier should be active when the CPU reports avx2 and the cap allows it
            assertTrue(
                    tiersAtOrAbove("avx2").contains(activeIsa),
                    "Expected AVX2 or higher tier when avx2 flag is present, got: " + activeIsa);
        } else if (!cpuHasAvx2) {
            assertEquals("sse42", activeIsa,
                    "Expected sse42 tier when avx2 flag is absent, got: " + activeIsa);
        }
        // else: cap is below avx2 — testDispatcherMatchesCpuInfo covers the exact value
    }

    /**
     * AVX3 (AVX-512 baseline) requires avx512f + avx512bw + avx512cd + avx512dq + avx512vl.
     * The OS must also have enabled ZMM state (avx512f implies OS context-switch support on
     * all kernels that expose the flag in /proc/cpuinfo).
     */
    @Test
    void testAvx3Detection() {
        boolean cpuHasAvx3 = hasAvx3(cpuFlags);
        String activeIsa = nativeSupport.getActiveIsa();
        if (cpuHasAvx3 && !isCappedBelow("avx3")) {
            assertTrue(
                    tiersAtOrAbove("avx3").contains(activeIsa),
                    "Expected AVX3 or higher tier when all avx512 baseline flags are present, got: " + activeIsa);
        } else if (!cpuHasAvx3) {
            assertTrue(
                    tiersBelow("avx3").contains(activeIsa),
                    "Expected at most AVX2 tier when AVX3 baseline flags are absent, got: " + activeIsa);
        }
        // else: cap is below avx3 — testDispatcherMatchesCpuInfo covers the exact value
    }

    /**
     * AVX3_DL (Ice Lake) additionally requires the ICX feature set on top of AVX3.
     * Relevant /proc/cpuinfo flags: avx512_vnni, avx512_vbmi, avx512_vbmi2, avx512_ifma,
     * avx512_bitalg, avx512_vpopcntdq, gfni, vaes, vpclmulqdq.
     */
    @Test
    void testAvx3DlDetection() {
        boolean cpuHasAvx3Dl = hasAvx3Dl(cpuFlags);
        String activeIsa = nativeSupport.getActiveIsa();
        if (cpuHasAvx3Dl && !isCappedBelow("avx3_dl")) {
            assertTrue(
                    tiersAtOrAbove("avx3_dl").contains(activeIsa),
                    "Expected AVX3_DL or higher tier when all ICX flags are present, got: " + activeIsa);
        } else if (!cpuHasAvx3Dl) {
            assertTrue(
                    tiersBelow("avx3_dl").contains(activeIsa),
                    "Expected below AVX3_DL when ICX flags are absent, got: " + activeIsa);
        }
        // else: cap is below avx3_dl — testDispatcherMatchesCpuInfo covers the exact value
    }

    /**
     * AVX3_SPR (Sapphire Rapids) additionally requires avx512_fp16 on top of AVX3_DL.
     */
    @Test
    void testAvx3SprDetection() {
        boolean cpuHasAvx3Spr = hasAvx3Spr(cpuFlags);
        String activeIsa = nativeSupport.getActiveIsa();
        if (cpuHasAvx3Spr && !isCappedBelow("avx3_spr")) {
            assertEquals("avx3_spr", activeIsa,
                    "Expected avx3_spr when avx512_fp16 and all ICX flags are present and no cap");
        } else if (!cpuHasAvx3Spr) {
            assertTrue(
                    tiersBelow("avx3_spr").contains(activeIsa),
                    "Expected below AVX3_SPR when avx512_fp16 is absent, got: " + activeIsa);
        }
        // else: cap is below avx3_spr — testDispatcherMatchesCpuInfo covers the exact value
    }

    // -------------------------------------------------------------------------
    // End-to-end dispatcher correctness test
    // -------------------------------------------------------------------------

    /**
     * Computes the expected ISA tier from /proc/cpuinfo using the same promotion
     * rules as {@code dispatch_kernels()} in {@code jvector_simd.cpp}, then
     * asserts it matches the ISA tier the native dispatcher actually chose.
     */
    @Test
    void testDispatcherMatchesCpuInfo() {
        String expected = expectedIsaFromCpuInfo(cpuFlags);
        String actual   = nativeSupport.getActiveIsa();

        assertNotNull(actual, "jvector_simd_get_active_isa() must not return null");
        assertEquals(expected, actual,
                "Dispatcher chose '" + actual + "' but /proc/cpuinfo implies '" + expected + "'."
                + "\nCPU flags: " + cpuFlags.stream().sorted().collect(Collectors.joining(", ")));
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Returns the ISA tier name that {@code dispatch_kernels()} should select
     * given the set of flags from /proc/cpuinfo and the JVECTOR_MAX_ISA cap.
     * Logic mirrors the C source exactly: pick the best hardware tier, then
     * clamp it down to the cap.
     */
    static String expectedIsaFromCpuInfo(Set<String> flags) {
        // Best tier supported by hardware (no cap applied yet)
        String best;
        if (hasAvx3Spr(flags))      best = "avx3_spr";
        else if (hasAvx3Dl(flags))  best = "avx3_dl";
        else if (hasAvx3(flags))    best = "avx3";
        else if (flags.contains("avx2")) best = "avx2";
        else                        best = "sse42";

        // Apply cap: if maxIsaCap is set and is below best, clamp down to it.
        if (maxIsaCap != null && ISA_TIERS.indexOf(maxIsaCap) < ISA_TIERS.indexOf(best)) {
            return maxIsaCap;
        }
        return best;
    }

    /**
     * Returns true if the JVECTOR_MAX_ISA cap prevents selecting {@code tier} or anything above it.
     * That is, the cap is set to something strictly below {@code tier}.
     */
    private static boolean isCappedBelow(String tier) {
        return maxIsaCap != null && ISA_TIERS.indexOf(maxIsaCap) < ISA_TIERS.indexOf(tier);
    }

    /** Returns the set of tier names at or above {@code tier} in ISA_TIERS. */
    private static Set<String> tiersAtOrAbove(String tier) {
        int idx = ISA_TIERS.indexOf(tier);
        return Set.copyOf(ISA_TIERS.subList(idx, ISA_TIERS.size()));
    }

    /** Returns the set of tier names strictly below {@code tier} in ISA_TIERS. */
    private static Set<String> tiersBelow(String tier) {
        int idx = ISA_TIERS.indexOf(tier);
        return Set.copyOf(ISA_TIERS.subList(0, idx));
    }

    /**
     * AVX3 = avx512f + avx512bw + avx512cd + avx512dq + avx512vl (all present in /proc/cpuinfo
     * only when the OS also saves ZMM state, so the XCR0 check is implicit).
     */
    private static boolean hasAvx3(Set<String> flags) {
        return flags.contains("avx512f")
                && flags.contains("avx512bw")
                && flags.contains("avx512cd")
                && flags.contains("avx512dq")
                && flags.contains("avx512vl");
    }

    /**
     * AVX3_DL = AVX3 + the full ICX feature set.
     * Flag names in /proc/cpuinfo use underscores and no "avx512_" prefix for gfni/vaes/vpclmulqdq.
     */
    private static boolean hasAvx3Dl(Set<String> flags) {
        return hasAvx3(flags)
                && flags.contains("avx512_vnni")
                && flags.contains("avx512vbmi")
                && flags.contains("avx512_vbmi2")
                && flags.contains("avx512ifma")
                && flags.contains("avx512_bitalg")
                && flags.contains("avx512_vpopcntdq")
                && flags.contains("gfni")
                && flags.contains("vaes")
                && flags.contains("vpclmulqdq");
    }

    /** AVX3_SPR = AVX3_DL + avx512_fp16. */
    private static boolean hasAvx3Spr(Set<String> flags) {
        return hasAvx3Dl(flags) && flags.contains("avx512_fp16");
    }

    /**
     * Reads the {@code flags} line of the first processor entry in /proc/cpuinfo
     * and returns the individual flag tokens as a lower-case Set.
     */
    private static Set<String> parseCpuInfoFlags() throws IOException {
        try (Stream<String> lines = Files.lines(Path.of("/proc/cpuinfo"))) {
            return lines
                    .filter(l -> l.startsWith("flags"))
                    .findFirst()
                    .map(l -> l.substring(l.indexOf(':') + 1).trim())
                    .map(flagStr -> Stream.of(flagStr.split("\\s+"))
                            .map(String::toLowerCase)
                            .collect(Collectors.toSet()))
                    .orElse(Set.of());
        }
    }
}
