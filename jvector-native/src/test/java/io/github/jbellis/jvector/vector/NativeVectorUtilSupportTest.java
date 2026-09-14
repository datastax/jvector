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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeVectorUtilSupportTest {
    private final NativeVectorUtilSupport support = new NativeVectorUtilSupport();
    private final MemorySegmentVectorProvider types = new MemorySegmentVectorProvider();

    @Test
    void subtractionRejectsShorterRightHandSide() {
        var a = types.createFloatVector(new float[]{1, 2});
        var b = types.createFloatVector(new float[]{1});
        assertThrows(IndexOutOfBoundsException.class, () -> support.sub(a, b));
        assertThrows(IndexOutOfBoundsException.class, () -> support.sub(a, 0, b, 0, 2));
    }

    @Test
    void subtractionChecksBothSourceRanges() {
        var a = types.createFloatVector(new float[]{1, 2, 3, 4});
        var b = types.createFloatVector(new float[]{4, 3, 2, 1});
        // Negative offsets, overruns, negative lengths and overflowing offset + length.
        int[][] invalidRanges = {{-1, 1}, {3, 2}, {5, 0}, {0, -1},
                                 {Integer.MAX_VALUE, 1}, {1, Integer.MAX_VALUE}};
        for (int[] range : invalidRanges) {
            int offset = range[0];
            int length = range[1];
            assertThrows(IndexOutOfBoundsException.class, () -> support.sub(a, offset, b, 0, length));
            assertThrows(IndexOutOfBoundsException.class, () -> support.sub(a, 0, b, offset, length));
            assertThrows(IndexOutOfBoundsException.class, () -> support.sub(a, offset, 1f, length));
        }
    }

    @Test
    void hammingRejectsShorterRightHandSide() {
        assertThrows(IndexOutOfBoundsException.class,
                     () -> support.hammingDistance(new long[2], new long[1]));
        assertThrows(IndexOutOfBoundsException.class,
                     () -> support.hammingDistance(new long[1], new long[0]));
    }

    @Test
    void subtractionAcceptsValidRangesAndEmptyEndpoints() {
        requireNativeProvider();
        var a = types.createFloatVector(new float[]{1, 2, 3, 4});
        var b = types.createFloatVector(new float[]{4, 3, 2, 1, 0});
        assertVectorEquals(new float[]{-1, 1, 3}, support.sub(a, 1, b, 1, 3));
        assertVectorEquals(new float[]{1, 2, 3}, support.sub(a, 1, 1f, 3));
        assertVectorEquals(new float[]{-3, -1, 1, 3}, support.sub(a, b));
        assertVectorEquals(new float[]{0, 1, 2, 3}, support.sub(a, 1f));
        assertEquals(0, support.sub(a, a.length(), b, b.length(), 0).length());
        assertEquals(0, support.sub(a, a.length(), 1f, 0).length());
    }

    @Test
    void hammingAcceptsEmptyEqualAndLongerRightHandSide() {
        requireNativeProvider();
        assertEquals(0, support.hammingDistance(new long[0], new long[0]));
        assertEquals(0, support.hammingDistance(new long[0], new long[]{-1L}));
        assertEquals(65, support.hammingDistance(new long[]{0L, 1L}, new long[]{-1L, 0L}));
        // Like the Java implementation, only the first a.length elements are compared.
        assertEquals(1, support.hammingDistance(new long[]{1L}, new long[]{0L, -1L}));
    }

    @Test
    void hammingMatchesScalarAcrossWordBoundaries() {
        requireNativeProvider();
        var random = new java.util.Random(753);
        for (int length : new int[]{0, 1, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 64, 71, 128, 255}) {
            long[] a = new long[length];
            long[] b = new long[length + 1];
            int expected = 0;
            for (int i = 0; i < length; i++) {
                a[i] = random.nextLong();
                b[i] = random.nextLong();
                expected += Long.bitCount(a[i] ^ b[i]);
            }
            b[length] = -1L;
            assertEquals(expected, support.hammingDistance(a, b), "length=" + length);
        }
    }

    @Test
    void minPropagatesNaNInVectorAndTailPositions() {
        requireNativeProvider();
        for (int length : new int[]{1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 37, 64, 71, 128, 255}) {
            for (int pos = 0; pos < length; pos++) {
                float[] values = new float[length];
                java.util.Arrays.fill(values, 2f);
                values[pos] = Float.NaN;
                assertTrue(Float.isNaN(support.min(types.createFloatVector(values))),
                           "length=" + length + ", NaN at " + pos);
            }
        }
    }

    @Test
    void minPreservesSignedZeroInVectorAndTailPositions() {
        requireNativeProvider();
        assertEquals(Float.MAX_VALUE, support.min(types.createFloatVector(0)));
        for (int length : new int[]{1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 37, 64, 71, 128, 255}) {
            float[] values = new float[length];
            assertEquals(0, Float.floatToRawIntBits(support.min(types.createFloatVector(values))));
            for (int pos = 0; pos < length; pos++) {
                java.util.Arrays.fill(values, 0f);
                values[pos] = -0f;
                assertEquals(Float.floatToRawIntBits(-0f),
                             Float.floatToRawIntBits(support.min(types.createFloatVector(values))),
                             "length=" + length + ", negative zero at " + pos);
                values[(pos + 1) % length] = -1f;
                assertEquals(-1f, support.min(types.createFloatVector(values)));
            }
        }
    }

    private static void requireNativeProvider() {
        assumeTrue(VectorizationProvider.getInstance() instanceof NativeVectorizationProvider,
                   "Native jvector provider not available");
    }

    private static void assertVectorEquals(float[] expected, VectorFloat<?> actual) {
        assertEquals(expected.length, actual.length());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.get(i));
        }
    }
}
