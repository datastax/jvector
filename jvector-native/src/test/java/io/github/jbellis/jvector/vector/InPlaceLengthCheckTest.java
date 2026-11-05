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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.*;

class InPlaceLengthCheckTest {
    @BeforeAll
    static void requireNativeProvider() {
        NativeTestSupport.requireNativeProvider();
    }

    private final VectorUtilSupport support = new NativeVectorizationProvider().getVectorUtilSupport();
    private final MemorySegmentVectorProvider vectors = new MemorySegmentVectorProvider();

    private VectorFloat<?> vec(int n, float v) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, v);
        return vectors.createFloatVector(a);
    }

    private void check(BiConsumer<VectorFloat<?>, VectorFloat<?>> op) {
        for (int[] lens : new int[][]{{8, 4}, {4, 8}, {1, 0}, {0, 1}, {33, 32}}) {
            var a = vec(lens[0], 1);
            var b = vec(lens[1], 2);
            assertThrows(IllegalArgumentException.class, () -> op.accept(a, b));
        }
        // valid empty inputs and equal lengths succeed
        op.accept(vec(0, 1), vec(0, 2));
        op.accept(vec(17, 1), vec(17, 2));
    }

    @Test
    void addInPlaceRejectsMismatchedLengths() { check(support::addInPlace); }

    @Test
    void subInPlaceRejectsMismatchedLengths() { check(support::subInPlace); }

    @Test
    void minInPlaceRejectsMismatchedLengths() { check(support::minInPlace); }

    @Test
    void validResultsUnchanged() {
        var a = vec(17, 5);
        support.addInPlace(a, vec(17, 2));
        assertEquals(7f, a.get(16));
        support.subInPlace(a, vec(17, 3));
        assertEquals(4f, a.get(0));
        support.minInPlace(a, vec(17, 1));
        assertEquals(1f, a.get(10));
    }
}
