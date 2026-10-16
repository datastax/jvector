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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NativeInheritedArithmeticTest {
    @BeforeAll
    static void loadNativeLibrary() {
        assertTrue(LibraryLoader.loadJvector(), "Native library is required for native kernel checks");
    }

    private static final int[] LENGTHS = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 256, 768, 1536, 3072};

    @Test
    void inheritedPqHelpersRetainNativeStorageAndSlices() {
        var support = new NativeVectorUtilSupport();
        int count = 19, clusters = 4, block = clusters * (clusters + 1) / 2;
        byte[] raw = new byte[count + 6];
        for (int i = 0; i < count; i++) raw[i + 3] = (byte) (i % clusters);
        var codes = new MemorySegmentByteSequence(raw).slice(3, count);
        float[] adc = new float[count * clusters], packed = new float[count * block];
        for (int i = 0; i < adc.length; i++) adc[i] = i % 7;
        for (int i = 0; i < packed.length; i++) packed[i] = i % 11;
        var table = new MemorySegmentVectorFloat(adc);
        var symmetric = new MemorySegmentVectorFloat(packed);
        for (int length : new int[] {0, 1, 7, 8, 15, 16, 19}) {
            float expected = 0, expectedSymmetric = 0;
            for (int i = 0; i < length; i++) {
                int code = codes.get(i);
                expected += adc[i * clusters + code];
                int row = code * clusters - code * (code - 1) / 2;
                expectedSymmetric += packed[i * block + row];
            }
            assertEquals(expected, support.assembleAndSum256(table, clusters, codes, 0, length));
            assertEquals(expected, support.assembleAndSum512(table, clusters, codes, 0, length));
            assertEquals(expectedSymmetric, support.assembleAndSumPQ_256(symmetric, length, codes, 0, codes, 0, clusters));
            assertEquals(expectedSymmetric, support.assembleAndSumPQ_512(symmetric, length, codes, 0, codes, 0, clusters));
        }
    }

    @Test
    void inheritedCosinePqRetainsNativeStorageAndSlices() {
        var support = new NativeVectorUtilSupport();
        int count = 19, clusters = 256;
        byte[] raw = new byte[count + 6];
        for (int i = 0; i < count; i++) raw[i + 3] = (byte) (255 - i);
        var codes = new MemorySegmentByteSequence(raw).slice(3, count);
        float[] partials = new float[count * clusters], magnitudes = new float[partials.length];
        for (int i = 0; i < partials.length; i++) {
            partials[i] = i % 9 - 4;
            magnitudes[i] = 1 + i % 7;
        }
        var table = new MemorySegmentVectorFloat(partials);
        var norms = new MemorySegmentVectorFloat(magnitudes);
        for (int length : new int[] {0, 1, 7, 8, 15, 16, 19}) {
            float sum = 0, magnitude = 0;
            for (int i = 0; i < length; i++) {
                int index = i * clusters + Byte.toUnsignedInt(codes.get(i));
                sum += partials[index];
                magnitude += magnitudes[index];
            }
            float expected = (float) (sum / Math.sqrt(magnitude * 3.25f));
            assertEquals(expected, support.pqDecodedCosineSimilarity256(codes, 0, length, clusters, table, norms, 3.25f));
            assertEquals(expected, support.pqDecodedCosineSimilarity512(codes, 0, length, clusters, table, norms, 3.25f));
        }
    }

    @Test
    void inheritedDistanceHelpersRetainNativeStorageSupport() {
        var support = new NativeVectorUtilSupport();
        float[] values = new float[137];
        for (int i = 0; i < values.length; i++) values[i] = i % 7 - 3;
        var vector = new MemorySegmentVectorFloat(values);
        for (int length : new int[] {0, 1, 2, 4, 8, 16, 31, 64, 128}) {
            float expected = 0;
            for (int i = 0; i < length; i++) expected += values[i + 3] * values[i + 5];
            assertEquals(expected, support.dotProduct64(vector, 3, vector, 5, length));
            assertEquals(expected, support.dotProduct128(vector, 3, vector, 5, length));
            assertEquals(expected, support.dotProduct256(vector, 3, vector, 5, length));
            assertEquals(expected, support.dotProductPreferred(vector, 3, vector, 5, length));
            expected = 0;
            for (int i = 0; i < length; i++) {
                float difference = values[i + 3] - values[i + 5];
                expected += difference * difference;
            }
            assertEquals(expected, support.squareDistance64(vector, 3, vector, 5, length));
            assertEquals(expected, support.squareDistance128(vector, 3, vector, 5, length));
            assertEquals(expected, support.squareDistance256(vector, 3, vector, 5, length));
            assertEquals(expected, support.squareDistancePreferred(vector, 3, vector, 5, length));
        }
    }

    @Test
    void inheritedSumAcceptsNativeStorage() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] values = new float[length];
            float expected = 0;
            for (int i = 0; i < length; i++) {
                // Exactly representable values and sums make the oracle independent of reduction order.
                values[i] = ((i * 17) % 127 - 63) / 256f;
                expected += values[i];
            }
            assertEquals(expected, support.sum(new MemorySegmentVectorFloat(values)), "length=" + length);
        }
    }

    @Test
    void inheritedScalePreservesSpecialValuesAndTails() {
        var support = new NativeVectorUtilSupport();
        float[] special = {0f, -0f, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -1f, 0.12345f};
        for (int length : LENGTHS) {
            for (float factor : new float[]{0f, -0f, 0.25f, -2f, Float.NaN, Float.POSITIVE_INFINITY}) {
                float[] values = new float[length];
                for (int i = 0; i < length; i++) values[i] = special[i % special.length];
                var actual = new MemorySegmentVectorFloat(values.clone());
                support.scale(actual, factor);
                for (int i = 0; i < length; i++) {
                    assertEquals(Float.floatToIntBits(values[i] * factor), Float.floatToIntBits(actual.get(i)),
                            "length=" + length + " lane=" + i);
                }
            }
        }
    }

    @Test
    void inheritedMinAndScalarSubtractionAcceptNativeStorage() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] values = new float[length];
            float expectedMin = Float.MAX_VALUE;
            for (int i = 0; i < length; i++) {
                values[i] = (i % 19 - 9) / 16f;
                expectedMin = Math.min(expectedMin, values[i]);
            }
            var vector = new MemorySegmentVectorFloat(values.clone());
            assertEquals(expectedMin, support.min(vector));
            var result = support.sub(vector, 0.25f);
            assertEquals(length, result.length());
            for (int i = 0; i < length; i++) {
                assertEquals(values[i] - 0.25f, result.get(i));
                assertEquals(values[i], vector.get(i));
            }
        }
    }

    @Test
    void inheritedSubtractionAcceptsNativeInputsAndPreservesOffsets() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] a = new float[length + 9], b = new float[length + 11];
            for (int i = 0; i < a.length; i++) a[i] = i * 0.125f - 4;
            for (int i = 0; i < b.length; i++) b[i] = i * -0.0625f + 2;
            var av = new MemorySegmentVectorFloat(a.clone());
            var bv = new MemorySegmentVectorFloat(b.clone());
            var result = support.sub(av, 3, bv, 7, length);
            assertEquals(length, result.length());
            for (int i = 0; i < length; i++) assertEquals(a[i + 3] - b[i + 7], result.get(i));
            // Inputs remain unchanged, including when both arguments reference the same storage.
            var aliased = support.sub(av, 1, av, 3, length);
            for (int i = 0; i < length; i++) assertEquals(a[i + 1] - a[i + 3], aliased.get(i));
            for (int i = 0; i < a.length; i++) assertEquals(a[i], av.get(i));
            for (int i = 0; i < b.length; i++) assertEquals(b[i], bv.get(i));
        }
    }
}
