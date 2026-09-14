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

    @Test public void fullScanFallbackVisitsEachPositionOnceIncludingEmptySources() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vectors = java.util.stream.IntStream.range(0, 19)
                .<VectorFloat<?>>mapToObj(i -> vts.createFloatVector(new float[] {i})).collect(java.util.stream.Collectors.toList());
        var source = new ListRandomAccessVectorValues(vectors, 1);
        var visits = new java.util.concurrent.atomic.AtomicIntegerArray(source.size());
        var pool = new ForkJoinPool(3);
        try {
            VectorAccess.forEach(source, pool, (ordinal, vector) -> {
                assertEquals(ordinal, vector.get(0), 0f);
                visits.incrementAndGet(ordinal);
            });
            for (int i = 0; i < source.size(); i++) assertEquals(1, visits.get(i));
            var empty = new ListRandomAccessVectorValues(java.util.List.of(), 1);
            VectorAccess.forEach(empty, pool, (ordinal, vector) -> fail("Empty source invoked consumer"));
        } finally { pool.shutdown(); }
    }

    @Test public void fullScanClosesItsCursorWhenConsumerFails() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var resident = new ListRandomAccessVectorValues(java.util.List.of(
                vts.createFloatVector(new float[] {1}), vts.createFloatVector(new float[] {2})), 1);
        var closed = new java.util.concurrent.atomic.AtomicInteger();
        BatchedVectorValues source = new BatchedVectorValues() {
            public int size() { return resident.size(); }
            public int dimension() { return resident.dimension(); }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { return this; }
            public VectorFloat<?> getVector(int ordinal) { return resident.getVector(ordinal); }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) {
                throw new AssertionError("Full scan should use range access");
            }
            public VectorCursor openRange(int start, int end) {
                var cursor = VectorAccess.openRange(resident, start, end);
                return new VectorCursor() {
                    public boolean next() { return cursor.next(); }
                    public int ordinal() { return cursor.ordinal(); }
                    public VectorFloat<?> vector() { return cursor.vector(); }
                    public boolean isValueShared() { return cursor.isValueShared(); }
                    public void close() { cursor.close(); closed.incrementAndGet(); }
                };
            }
        };
        var pool = new ForkJoinPool(1);
        try {
            assertThrows(IllegalStateException.class, () -> VectorAccess.forEach(source, pool,
                    (ordinal, vector) -> { throw new IllegalStateException("sink failed"); }));
            assertEquals(1, closed.get());
            VectorAccess.forEach(source, pool, (ordinal, vector) -> assertEquals(ordinal + 1, vector.get(0), 0f));
            assertEquals(2, closed.get());
        } finally { pool.shutdown(); }
    }

    @Test public void unsharedPointSourceDoesNotNeedCopy() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var values = java.util.List.of(vts.createFloatVector(new float[] {1}), vts.createFloatVector(new float[] {2}));
        RandomAccessVectorValues source = new RandomAccessVectorValues() {
            public int size() { return values.size(); }
            public int dimension() { return 1; }
            public boolean isValueShared() { return false; }
            public VectorFloat<?> getVector(int ordinal) { return values.get(ordinal); }
            public RandomAccessVectorValues copy() { throw new UnsupportedOperationException("No copy needed"); }
        };
        try (var cursor = VectorAccess.openRange(source, 0, 2)) {
            assertTrue(cursor.next()); assertSame(values.get(0), cursor.vector());
            assertTrue(cursor.next()); assertSame(values.get(1), cursor.vector());
            assertFalse(cursor.next());
        }
        try (var cursor = VectorAccess.openSelection(source, new int[] {1, 0, 1}, 0, 3)) {
            for (int ordinal : new int[] {1, 0, 1}) {
                assertTrue(cursor.next()); assertEquals(ordinal, cursor.ordinal()); assertSame(values.get(ordinal), cursor.vector());
            }
            assertFalse(cursor.next());
        }
        var pool = new ForkJoinPool(2);
        try {
            assertSame(values.get(1), VectorAccess.copySelected(source, new int[] {1}, pool).get(0));
            VectorAccess.forEach(source, pool, (ordinal, vector) -> assertSame(values.get(ordinal), vector));
        } finally { pool.shutdown(); }
    }

    @Test public void failedPointAdvanceInvalidatesCursor() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var value = vts.createFloatVector(new float[] {1});
        var failure = new IllegalStateException("Source read failed");
        RandomAccessVectorValues source = new RandomAccessVectorValues() {
            public int size() { return 3; }
            public int dimension() { return 1; }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { throw new UnsupportedOperationException(); }
            public VectorFloat<?> getVector(int ordinal) {
                if (ordinal == 1) throw failure;
                return value;
            }
        };
        for (boolean selection : new boolean[] {false, true}) {
            for (int start : new int[] {0, 1}) {
                try (var cursor = selection
                        ? VectorAccess.openSelection(source, new int[] {0, 1, 2}, start, 3 - start)
                        : VectorAccess.openRange(source, start, 3)) {
                    if (start == 0) { assertTrue(cursor.next()); assertSame(value, cursor.vector()); }
                    assertSame(failure, assertThrows(IllegalStateException.class, cursor::next));
                    assertThrows(IllegalStateException.class, cursor::vector);
                    assertThrows(IllegalStateException.class, cursor::ordinal);
                    assertThrows(IllegalStateException.class, cursor::next);
                }
            }
        }
        try (var cursor = VectorAccess.openRange(source, 2, 3)) {
            assertTrue(cursor.next()); assertSame(value, cursor.vector()); assertFalse(cursor.next());
        }
    }
    @Test public void mappedScanBoundsSelectionMetadataAndPreservesOutputPositions() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        int count = 1_000_003;
        var largest = new java.util.concurrent.atomic.AtomicInteger();
        var reads = new java.util.concurrent.atomic.AtomicInteger();
        BatchedVectorValues source = new BatchedVectorValues() {
            public int size() { return count; }
            public int dimension() { return 1; }
            public boolean isValueShared() { return true; }
            public RandomAccessVectorValues copy() { throw new AssertionError("Point copy used"); }
            public VectorFloat<?> getVector(int ordinal) { throw new AssertionError("Point read used"); }
            public VectorCursor openRange(int start, int end) { throw new AssertionError("Mapped scan used range"); }
            public VectorCursor openSelection(int[] ordinals, int offset, int length) {
                largest.accumulateAndGet(length, Math::max);
                assertTrue("Dataset-sized request", length <= 4096);
                final int[] snapshot = Arrays.copyOfRange(ordinals, offset, offset + length);
                final VectorFloat<?> value = vts.createFloatVector(1);
                return new VectorCursor() {
                    int position = -1;
                    public boolean next() {
                        if (++position == length) return false;
                        assertEquals(snapshot[position], ordinals[offset + position]);
                        value.set(0, snapshot[position]); reads.incrementAndGet(); return true;
                    }
                    public int ordinal() { return snapshot[position]; }
                    public VectorFloat<?> vector() { return value; }
                    public boolean isValueShared() { return true; }
                    public void close() {
                        // The caller cannot overwrite this request before the cursor closes.
                        assertArrayEquals(snapshot, Arrays.copyOfRange(ordinals, offset, offset + length));
                    }
                };
            }
        };
        var visits = new java.util.concurrent.atomic.AtomicIntegerArray(count);
        var pool = new ForkJoinPool(2);
        try {
            // A permutation followed by duplicates exercises both mapping and output positions.
            VectorAccess.forEach(source, count, i -> (count - 1 - i) / 2, pool, (position, vector) -> {
                assertEquals((count - 1 - position) / 2, vector.get(0), 0f);
                visits.incrementAndGet(position);
            });
            assertEquals(count, reads.get());
            assertEquals(4096, largest.get());
            for (int i = 0; i < count; i++) assertEquals(1, visits.get(i));
        } finally { pool.shutdown(); }
    }

}
