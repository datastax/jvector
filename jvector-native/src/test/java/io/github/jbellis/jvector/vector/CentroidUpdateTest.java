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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class CentroidUpdateTest {
    @BeforeAll
    static void requireNativeProvider() {
        NativeTestSupport.requireNativeProvider();
    }

    private final VectorUtilSupport support = new NativeVectorizationProvider().getVectorUtilSupport();
    private final MemorySegmentVectorProvider vectors = new MemorySegmentVectorProvider();

    @Test
    void normalizationMatchesCopyScaleCopyAndPreservesOtherRegions() {
        Random random = new Random(3001);
        for (int length : new int[]{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 24, 32, 65, 1536}) {
            float[] values = new float[length];
            for (int i = 0; i < length; i++) values[i] = random.nextFloat() * 200 - 100;
            for (float multiplier : new float[]{0, -0.0f, 1, -1, 1.0f / 3, 1.0f / 127}) {
                var source = vectors.createFloatVector(values.clone());
                var expected = source.copy();
                support.scale(expected, multiplier);
                float[] initial = new float[length + 11];
                java.util.Arrays.fill(initial, 17);
                var destination = vectors.createFloatVector(initial);
                assertTrue(support.tryScaleTo(source, destination, 5, multiplier));
                for (int i = 0; i < length; i++) {
                    assertEquals(Float.floatToRawIntBits(expected.get(i)), Float.floatToRawIntBits(destination.get(5 + i)));
                    assertEquals(Float.floatToRawIntBits(values[i]), Float.floatToRawIntBits(source.get(i)));
                }
                for (int i = 0; i < 5; i++) assertEquals(17, destination.get(i));
                for (int i = length + 5; i < destination.length(); i++) assertEquals(17, destination.get(i));
            }
        }
    }

    @Test
    void assignmentMatchesSeparateUpdatesBitwiseAndKeepsPointUnchanged() {
        Random random = new Random(3002);
        for (int length : new int[]{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 24, 32, 65, 1536}) {
            float[] a = new float[length], b = new float[length], point = new float[length];
            for (int i = 0; i < length; i++) { a[i] = random.nextFloat() * 100; b[i] = -random.nextFloat() * 100; point[i] = random.nextFloat(); }
            var oldSum = vectors.createFloatVector(a.clone());
            var newSum = vectors.createFloatVector(b.clone());
            var p = vectors.createFloatVector(point.clone());
            var oldReference = oldSum.copy();
            var newReference = newSum.copy();
            support.subInPlace(oldReference, p);
            support.addInPlace(newReference, p);
            assertTrue(support.trySubtractAddInPlace(oldSum, newSum, p));
            for (int i = 0; i < length; i++) {
                assertEquals(Float.floatToRawIntBits(oldReference.get(i)), Float.floatToRawIntBits(oldSum.get(i)));
                assertEquals(Float.floatToRawIntBits(newReference.get(i)), Float.floatToRawIntBits(newSum.get(i)));
                assertEquals(Float.floatToRawIntBits(point[i]), Float.floatToRawIntBits(p.get(i)));
            }
        }
    }

    @Test
    void invalidRegionsAndAliasingAreRejectedBeforeMutation() {
        var a = vectors.createFloatVector(8);
        var b = vectors.createFloatVector(17);
        var p = vectors.createFloatVector(8);
        assertThrows(IndexOutOfBoundsException.class, () -> support.tryScaleTo(a, b, -1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> support.tryScaleTo(a, b, 10, 1));
        assertThrows(IllegalArgumentException.class, () -> support.tryScaleTo(a, a, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> support.trySubtractAddInPlace(a, b, p));
        assertThrows(IllegalArgumentException.class, () -> support.trySubtractAddInPlace(a, a, p));
        assertThrows(IllegalArgumentException.class, () -> support.trySubtractAddInPlace(a, p, a));
        for (int i = 0; i < a.length(); i++) assertEquals(0, a.get(i));
    }
}
