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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.Assert.*;

public class TestFinalWriteVectors {
    private static ListRandomAccessVectorValues values() {
        var vectors = new ArrayList<VectorFloat<?>>();
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int i = 0; i < 131; i++) vectors.add(vts.createFloatVector(new float[]{i, -i, i * 2}));
        return new ListRandomAccessVectorValues(vectors, 3);
    }

    @Test
    public void rangesAreBoundedAndReleased() {
        var resident = values();
        var opens = new AtomicInteger();
        var closes = new AtomicInteger();
        BatchedVectorValues source = new BatchedVectorValues() {
            public int size() { return resident.size(); }
            public int dimension() { return resident.dimension(); }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { return this; }
            public VectorFloat<?> getVector(int ordinal) { throw new AssertionError("Point access"); }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) {
                throw new AssertionError("Selection access");
            }
            public VectorCursor openRange(int start, int end) {
                assertTrue(end - start <= 64);
                opens.incrementAndGet();
                var cursor = VectorAccess.openRange(resident, start, end);
                return new VectorCursor() {
                    boolean closed;
                    public boolean next() { return cursor.next(); }
                    public int ordinal() { return cursor.ordinal(); }
                    public VectorFloat<?> vector() { return cursor.vector(); }
                    public boolean isValueShared() { return cursor.isValueShared(); }
                    public void close() {
                        if (!closed) { closed = true; closes.incrementAndGet(); cursor.close(); }
                    }
                };
            }
        };
        try (var input = new FinalWriteVectors(source)) {
            for (int i = 0; i < source.size(); i++) {
                var v = input.get(i);
                assertEquals(i, v.get(0), 0);
                assertSame("Features for one record reuse the same input", v, input.get(i));
            }
            assertEquals(3, opens.get());
            assertEquals(17, input.get(17).get(0), 0);
            assertEquals(0, input.get(0).get(0), 0);
        }
        assertEquals(opens.get(), closes.get());
    }

    @Test(timeout = 60000)
    public void ordinaryBorrowedSourcesHaveIndependentWorkerViews() throws Exception {
        var resident = values();
        class Borrowed implements RandomAccessVectorValues {
            final VectorFloat<?> scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(3);
            public int size() { return resident.size(); }
            public int dimension() { return 3; }
            public boolean isValueShared() { return true; }
            public RandomAccessVectorValues copy() { return new Borrowed(); }
            public VectorFloat<?> getVector(int ordinal) {
                resident.getVectorInto(ordinal, scratch, 0);
                return scratch;
            }
        }
        var pool = Executors.newFixedThreadPool(4);
        try (var input = new FinalWriteVectors(new Borrowed())) {
            var tasks = new ArrayList<Callable<Void>>();
            for (int worker = 0; worker < 8; worker++) tasks.add(() -> {
                for (int ordinal = 0; ordinal < resident.size(); ordinal++) {
                    var vector = input.get(ordinal);
                    Thread.yield();
                    assertEquals(ordinal, vector.get(0), 0);
                    assertEquals(-ordinal, vector.get(1), 0);
                    assertSame(vector, input.get(ordinal));
                }
                return null;
            });
            for (var future : pool.invokeAll(tasks)) future.get();
        } finally { pool.shutdownNow(); }
    }

    @Test(timeout = 60000)
    public void closeReleasesOtherWorkersWhenOneCursorFails() throws Exception {
        var resident = values();
        var closes = new AtomicInteger();
        BatchedVectorValues source = new BatchedVectorValues() {
            public int size() { return resident.size(); }
            public int dimension() { return resident.dimension(); }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { return this; }
            public VectorFloat<?> getVector(int ordinal) { throw new AssertionError("Point access"); }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) { throw new AssertionError("Selection access"); }
            public VectorCursor openRange(int start, int end) {
                var cursor = VectorAccess.openRange(resident, start, end);
                return new VectorCursor() {
                    public boolean next() { return cursor.next(); }
                    public int ordinal() { return cursor.ordinal(); }
                    public VectorFloat<?> vector() { return cursor.vector(); }
                    public boolean isValueShared() { return false; }
                    public void close() {
                        closes.incrementAndGet(); cursor.close();
                        if (start == 0) throw new IllegalStateException("cursor close failure");
                    }
                };
            }
        };
        var pool = Executors.newFixedThreadPool(2);
        var input = new FinalWriteVectors(source);
        try {
            var barrier = new CyclicBarrier(2);
            for (var future : pool.invokeAll(List.<Callable<Void>>of(
                    () -> { input.get(0); barrier.await(10, TimeUnit.SECONDS); return null; },
                    () -> { input.get(64); barrier.await(10, TimeUnit.SECONDS); return null; }))) future.get();
            var failure = assertThrows(IllegalStateException.class, input::close);
            assertEquals("cursor close failure", failure.getMessage());
            assertEquals(2, closes.get());
            input.close();
            assertEquals(2, closes.get());
        } finally {
            pool.shutdownNow(); pool.awaitTermination(10, TimeUnit.SECONDS);
            input.close();
        }
    }
}
