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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class TestRangeRandomAccessVectorValues extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DIMENSION = 4;

    private static List<VectorFloat<?>> randomVectors(int count) {
        List<VectorFloat<?>> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            float[] values = new float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                values[d] = randomFloat();
            }
            vectors.add(vts.createFloatVector(values));
        }
        return vectors;
    }

    private static void assertVectorEquals(VectorFloat<?> expected, VectorFloat<?> actual) {
        assertEquals(expected.length(), actual.length());
        for (int d = 0; d < expected.length(); d++) {
            assertEquals(expected.get(d), actual.get(d), 0f);
        }
    }

    @Test
    public void testRangeReadsAreRebased() {
        List<VectorFloat<?>> vectors = randomVectors(20);
        var full = new ListRandomAccessVectorValues(vectors, DIMENSION);

        var view = full.range(5, 12);
        assertEquals(7, view.size());
        assertEquals(DIMENSION, view.dimension());
        assertFalse(view.isValueShared());
        for (int i = 0; i < view.size(); i++) {
            assertSame(vectors.get(5 + i), view.getVector(i));

            var dest = vts.createFloatVector(2 * DIMENSION);
            view.getVectorInto(i, dest, DIMENSION);
            for (int d = 0; d < DIMENSION; d++) {
                assertEquals(vectors.get(5 + i).get(d), dest.get(DIMENSION + d), 0f);
            }
        }
    }

    @Test
    public void testFullAndEmptyRanges() {
        List<VectorFloat<?>> vectors = randomVectors(10);
        var full = new ListRandomAccessVectorValues(vectors, DIMENSION);

        var whole = full.range(0, full.size());
        assertEquals(full.size(), whole.size());
        for (int i = 0; i < full.size(); i++) {
            assertSame(full.getVector(i), whole.getVector(i));
        }

        var empty = full.range(4, 4);
        assertEquals(0, empty.size());
        assertThrows(IndexOutOfBoundsException.class, () -> empty.getVector(0));

        var tail = full.range(full.size(), full.size());
        assertEquals(0, tail.size());
    }

    @Test
    public void testRangeBoundsAreValidated() {
        var full = new ListRandomAccessVectorValues(randomVectors(10), DIMENSION);

        assertThrows(IndexOutOfBoundsException.class, () -> full.range(-1, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> full.range(6, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> full.range(0, 11));
        assertThrows(IndexOutOfBoundsException.class, () -> full.range(11, 11));

        var view = full.range(2, 6);
        assertThrows(IndexOutOfBoundsException.class, () -> view.getVector(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> view.getVector(4));
        assertThrows(IndexOutOfBoundsException.class, () -> view.getVectorInto(4, vts.createFloatVector(DIMENSION), 0));
    }

    @Test
    public void testNestedRangeCollapsesToBacking() {
        List<VectorFloat<?>> vectors = randomVectors(20);
        var full = new ListRandomAccessVectorValues(vectors, DIMENSION);

        var outer = full.range(2, 12);
        var inner = outer.range(3, 7);

        assertTrue(inner instanceof RangeRandomAccessVectorValues);
        var range = (RangeRandomAccessVectorValues) inner;
        assertEquals(5, range.fromOrdinal());
        assertEquals(9, range.toOrdinal());
        assertEquals(4, inner.size());
        for (int i = 0; i < inner.size(); i++) {
            assertSame(vectors.get(5 + i), inner.getVector(i));
        }

        assertThrows(IndexOutOfBoundsException.class, () -> outer.range(0, 11));
    }

    @Test
    public void testSizeIsFixedAtCreation() {
        List<VectorFloat<?>> vectors = new ArrayList<>(randomVectors(6));
        var full = new ListRandomAccessVectorValues(vectors, DIMENSION);

        var view = full.range(2, 6);
        vectors.addAll(randomVectors(4));

        assertEquals(10, full.size());
        assertEquals(4, view.size());
        assertThrows(IndexOutOfBoundsException.class, () -> view.getVector(4));
        assertThrows(IndexOutOfBoundsException.class, () -> full.range(0, 6).range(0, 7));
    }

    @Test
    public void testCopyFollowsBackingSemantics() {
        List<VectorFloat<?>> vectors = randomVectors(8);
        var unshared = new ListRandomAccessVectorValues(vectors, DIMENSION);
        var unsharedView = unshared.range(1, 5);
        assertSame(unsharedView, unsharedView.copy());

        var shared = MockVectorValues.fromValues(vectors.toArray(new VectorFloat<?>[0]));
        var sharedView = shared.range(1, 5);
        assertTrue(sharedView.isValueShared());

        var copied = sharedView.copy();
        assertNotSame(sharedView, copied);
        assertEquals(sharedView.size(), copied.size());
        for (int i = 0; i < sharedView.size(); i++) {
            assertVectorEquals(vectors.get(1 + i), copied.getVector(i));
        }

        // a shared backing RAVV hands back one scratch reference; the view must not hide that
        var first = sharedView.getVector(0);
        var second = sharedView.getVector(1);
        assertSame(first, second);
        assertVectorEquals(vectors.get(2), second);
    }
}
