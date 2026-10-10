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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import static org.junit.Assert.*;

public class TestVectorAccess {
    @Test public void fallbackPreservesSharedValuesAndDuplicates() {
        RandomAccessVectorValues source = new RandomAccessVectorValues() {
            final VectorFloat<?> value = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(1);
            public int size() { return 8; }
            public int dimension() { return 1; }
            public boolean isValueShared() { return true; }
            public VectorFloat<?> getVector(int ordinal) { value.set(0, ordinal); return value; }
            public RandomAccessVectorValues copy() { return new RandomAccessVectorValues() {
                final VectorFloat<?> v = value.copy();
                public int size() { return 8; }
                public int dimension() { return 1; }
                public boolean isValueShared() { return true; }
                public VectorFloat<?> getVector(int ordinal) { v.set(0, ordinal); return v; }
                public RandomAccessVectorValues copy() { return this; }
            }; }
        };
        var pool = new ForkJoinPool(4);
        try {
            var result = VectorAccess.copySelected(source, new int[] {7, 1, 7, 0}, pool);
            source.getVector(2);
            for (int i = 0; i < 4; i++) assertEquals(new float[] {7, 1, 7, 0}[i], result.get(i).get(0), 0);
            var cursor = VectorAccess.openSelection(source, new int[] {5, 2, 5}, 0, 3);
            try (cursor) {
                assertThrows(IllegalStateException.class, cursor::vector);
                for (int expected : new int[] {5, 2, 5}) { assertTrue(cursor.next()); assertEquals(expected, cursor.ordinal()); assertEquals(expected, cursor.vector().get(0), 0); }
                assertFalse(cursor.next()); assertFalse(cursor.next());
                assertThrows(IllegalStateException.class, cursor::vector);
            }
            assertThrows(IllegalStateException.class, cursor::next);
        } finally { pool.shutdown(); }
    }

    @Test public void fallbackRangeUsesExclusiveEndAndRetainedCopiesStayStable() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var source = new ListRandomAccessVectorValues(Arrays.asList(
                vts.createFloatVector(new float[] {0}), vts.createFloatVector(new float[] {1}),
                vts.createFloatVector(new float[] {2}), vts.createFloatVector(new float[] {3})), 1);
        VectorFloat<?> retained;
        try (var cursor = VectorAccess.openRange(source, 1, 3)) {
            assertTrue(cursor.next()); assertEquals(1, cursor.ordinal());
            retained = cursor.vector().copy();
            assertTrue(cursor.next()); assertEquals(2, cursor.ordinal());
            assertFalse(cursor.next());
        }
        assertEquals(1f, retained.get(0), 0);
        try (var cursor = VectorAccess.openRange(source, 4, 4)) { assertFalse(cursor.next()); }
        assertThrows(IndexOutOfBoundsException.class, () -> VectorAccess.openRange(source, 3, 1));
    }

    @Test public void residentFallbackDoesNotCopyVectors() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vector = vts.createFloatVector(new float[] {3});
        var source = new ListRandomAccessVectorValues(Arrays.asList(vector), 1);
        var pool = new ForkJoinPool(2);
        try { assertSame(vector, VectorAccess.copySelected(source, new int[] {0}, pool).get(0)); }
        finally { pool.shutdown(); }
        assertThrows(IndexOutOfBoundsException.class, () -> VectorAccess.openRange(source, 1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> VectorAccess.openSelection(source, new int[] {1}, 0, 1));
        try (var cursor = VectorAccess.openRange(source, 0, 0)) { assertFalse(cursor.next()); }
    }
}
