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

import io.github.jbellis.jvector.quantization.FusedASHLayout;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Exact binary-fraction oracles cover lane tails, both FFM modes and concurrent callers. */
class NativeASHCanonicalLutTest {
    @BeforeAll static void loadLibrary() {
        assumeTrue(LibraryLoader.loadJvector(), "Native library unavailable");
        assertTrue(NativeASHKernels.isCanonicalLutAvailable(), "New native kernel must be packaged");
        assertTrue(NativeASHKernels.isFourBitDotAvailable(), "Native 4-bit dot must be packaged");
    }

    @Test void canonicalCodesMatchIndependentProjectionOracle() {
        Random random = new Random(103);
        for (int bits : new int[] {2, 4}) {
            for (int dims : new int[] {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 127, 384, 512, 768, 1536, 3073}) {
                float[] q = new float[dims];
                for (int j = 0; j < dims; j++) q[j] = (random.nextInt(33) - 16) / 32f;
                byte[] code = new byte[(dims * bits + 7) / 8];
                random.nextBytes(code);
                int groups = FusedASHLayout.codeGroups(dims, bits);
                float[] lut = new float[groups * 16];
                FusedASHLayout.buildQueryLut(q, dims, bits, lut);
                double expected = 0;
                for (int j = 0; j < dims; j++) {
                    int field = (code[j * bits / 8] >>> ((j * bits) % 8)) & ((1 << bits) - 1);
                    int sign = 1 << (bits - 1);
                    double component = ((field & (sign - 1)) + .5) * ((field & sign) == 0 ? -1 : 1);
                    expected += q[j] * component;
                }
                assertEquals((float) expected, NativeASHKernels.dotCanonical(code, groups, lut), 0f,
                        "bits=" + bits + " dims=" + dims);
            }
        }
    }

    @Test void generalFp32InputsMatchDoublePrecisionDecoder() {
        Random random = new Random(83417);
        for (int bits : new int[] {2, 4}) {
            for (int dims : new int[] {33, 127, 384, 512, 768, 1536, 3073}) {
                float[] q = new float[dims];
                for (int j = 0; j < dims; j++) q[j] = random.nextFloat() - .5f;
                byte[] code = new byte[(dims * bits + 7) / 8];
                random.nextBytes(code);
                int groups = FusedASHLayout.codeGroups(dims, bits);
                float[] lut = new float[groups * 16];
                FusedASHLayout.buildQueryLut(q, dims, bits, lut);
                double expected = 0, magnitude = 0;
                for (int j = 0; j < dims; j++) {
                    int field = (code[j * bits / 8] >>> ((j * bits) % 8)) & ((1 << bits) - 1);
                    int sign = 1 << (bits - 1);
                    double component = ((field & (sign - 1)) + .5) * ((field & sign) == 0 ? -1 : 1);
                    expected += q[j] * component;
                    magnitude += Math.abs(q[j] * component);
                }
                // Allow FP32 LUT construction and reduction order; scale by the
                // absolute sum so cancellation does not make the tolerance misleading.
                assertEquals(expected, NativeASHKernels.dotCanonical(code, groups, lut), 1e-6 + magnitude * 2e-6);
            }
        }
    }

    @Test void arbitraryLutsCoverOddGroupsAndOrdinaryDowncalls() {
        Random random = new Random(123);
        for (int groups : new int[] {1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                NativeASHKernels.MAX_CRITICAL_LANE_GROUPS, NativeASHKernels.MAX_CRITICAL_LANE_GROUPS + 1}) {
            byte[] code = new byte[(groups + 1) / 2];
            float[] lut = new float[groups * 16];
            random.nextBytes(code);
            for (int i = 0; i < lut.length; i++) lut[i] = (random.nextInt(33) - 16) / 32f;
            float expected = ASHLutScoring.dotCanonical(code, groups, lut);
            assertEquals(expected, NativeASHKernels.dotCanonical(code, groups, lut), 0f);
            if ((groups & 1) != 0) {
                code[code.length - 1] ^= (byte) 0xf0;
                assertEquals(expected, NativeASHKernels.dotCanonical(code, groups, lut), 0f);
            }
        }
    }

    @Test void contiguousFourBitDotMatchesIndependentDecoder() {
        Random random = new Random(89217);
        for (int dims : new int[] {0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 383, 384, 1536, 3073,
                NativeASHKernels.MAX_CRITICAL_LANE_GROUPS, NativeASHKernels.MAX_CRITICAL_LANE_GROUPS + 1}) {
            float[] query = new float[dims];
            byte[] code = new byte[(dims + 1) / 2];
            random.nextBytes(code);
            double expected = 0, magnitude = 0;
            for (int j = 0; j < dims; j++) {
                query[j] = random.nextFloat() - .5f;
                int field = (code[j / 2] >>> ((j & 1) * 4)) & 15;
                double component = ((field & 7) + .5) * ((field & 8) == 0 ? -1 : 1);
                expected += query[j] * component;
                magnitude += Math.abs(query[j] * component);
            }
            assertEquals(expected, NativeASHKernels.dotFourBit(query, code, dims), 1e-6 + magnitude * 2e-6);
        }
    }

    @Test void contiguousFourBitDotChecksBounds() {
        assertEquals(0f, NativeASHKernels.dotFourBit(new float[0], new byte[0], 0));
        assertThrows(NullPointerException.class, () -> NativeASHKernels.dotFourBit(null, new byte[1], 1));
        assertThrows(NullPointerException.class, () -> NativeASHKernels.dotFourBit(new float[1], null, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotFourBit(new float[1], new byte[1], -1));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotFourBit(new float[1], new byte[1], 2));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotFourBit(new float[3], new byte[1], 3));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotFourBit(new float[1], new byte[1], Integer.MAX_VALUE));
    }

    @Test void rejectsInvalidInputsBeforeNativeAccess() {
        assertThrows(NullPointerException.class, () -> NativeASHKernels.dotCanonical(null, 1, new float[16]));
        assertThrows(NullPointerException.class, () -> NativeASHKernels.dotCanonical(new byte[1], 1, null));
        for (int groups : new int[] {0, -1})
            assertThrows(IllegalArgumentException.class, () -> NativeASHKernels.dotCanonical(new byte[1], groups, new float[16]));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotCanonical(new byte[0], 1, new float[16]));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotCanonical(new byte[1], 1, new float[15]));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHKernels.dotCanonical(new byte[1], Integer.MAX_VALUE, new float[16]));
    }

    @Test void propagatesSelectedNonFiniteEntries() {
        float[] lut = new float[32];
        lut[0] = Float.NaN;
        assertTrue(Float.isNaN(NativeASHKernels.dotCanonical(new byte[1], 2, lut)));
        lut[0] = Float.POSITIVE_INFINITY;
        assertEquals(Float.POSITIVE_INFINITY, NativeASHKernels.dotCanonical(new byte[1], 2, lut));
        assertTrue(Float.isNaN(NativeASHKernels.dotFourBit(new float[]{Float.NaN}, new byte[]{8}, 1)));
        assertEquals(Float.POSITIVE_INFINITY,
                NativeASHKernels.dotFourBit(new float[]{Float.POSITIVE_INFINITY}, new byte[]{8}, 1));
    }

    @Test void concurrentCallsRetainNoQueryState() {
        CompletableFuture<?>[] jobs = IntStream.range(0, 8).mapToObj(seed -> CompletableFuture.runAsync(() -> {
            Random random = new Random(seed);
            byte[] code = new byte[257];
            float[] lut = new float[513 * 16];
            random.nextBytes(code);
            for (int i = 0; i < lut.length; i++) lut[i] = (random.nextInt(33) - 16) / 32f;
            float expected = ASHLutScoring.dotCanonical(code, 513, lut);
            float[] query = new float[513];
            float expectedFourBit = 0;
            for (int j = 0; j < query.length; j++) {
                query[j] = (random.nextInt(33) - 16) / 32f;
                int field = (code[j / 2] >>> ((j & 1) * 4)) & 15;
                float value = (field & 7) + .5f;
                expectedFourBit += query[j] * ((field & 8) == 0 ? -value : value);
            }
            for (int repeat = 0; repeat < 100; repeat++) {
                assertEquals(expected, NativeASHKernels.dotCanonical(code, 513, lut), 0f);
                assertEquals(expectedFourBit, NativeASHKernels.dotFourBit(query, code, query.length), 0f);
            }
        })).toArray(CompletableFuture<?>[]::new);
        CompletableFuture.allOf(jobs).join();
    }
}
