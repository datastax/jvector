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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

public class TestThreadLocalCopies {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Shared RAVV whose copies count their close() calls; the source itself never counts. */
    static class CloseTrackingRavv implements RandomAccessVectorValues, AutoCloseable {
        private final AtomicInteger closedCount;
        private final boolean source;

        CloseTrackingRavv(AtomicInteger closedCount) {
            this(closedCount, true);
        }

        CloseTrackingRavv(AtomicInteger closedCount, boolean source) {
            this.closedCount = closedCount;
            this.source = source;
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public int dimension() {
            return 1;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            return vts.createFloatVector(1);
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return new CloseTrackingRavv(closedCount, false);
        }

        @Override
        public void close() {
            if (!source) {
                closedCount.incrementAndGet();
            }
        }
    }

    @Test
    public void closingSupplierClosesAllThreadLocalCopies() throws Exception {
        AtomicInteger closed = new AtomicInteger();
        CloseTrackingRavv source = new CloseTrackingRavv(closed);
        var supplier = source.threadLocalSupplier();

        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            for (int i = 0; i < 2; i++) {
                pool.submit(() -> supplier.get()).get();
            }
        }
        finally {
            pool.shutdown();
            pool.awaitTermination(10, TimeUnit.SECONDS);
        }

        ((AutoCloseable) supplier).close();
        assertEquals(2, closed.get());
    }

    @Test
    public void supplierRemainsUsableAfterClose() throws Exception {
        AtomicInteger closed = new AtomicInteger();
        CloseTrackingRavv source = new CloseTrackingRavv(closed);
        var supplier = source.threadLocalSupplier();

        supplier.get();
        ((AutoCloseable) supplier).close();
        assertEquals(1, closed.get());

        // The per-thread cache is dropped on close: a fresh copy is created and closed again.
        supplier.get();
        ((AutoCloseable) supplier).close();
        assertEquals(2, closed.get());
    }

    @Test
    public void unsharedRavvSupplierReturnsTheSource() {
        AtomicInteger closed = new AtomicInteger();
        RandomAccessVectorValues unshared = new CloseTrackingRavv(closed) {
            @Override
            public boolean isValueShared() {
                return false;
            }
        };
        var supplier = unshared.threadLocalSupplier();
        assertSame(unshared, supplier.get());
    }

    /** Shared RAVV whose copies fail node 0 and record any read that overlaps close(). */
    static class SlowFailingRavv implements RandomAccessVectorValues, AutoCloseable {
        static final int SIZE = 64;
        private final AtomicInteger activeReads;
        private final AtomicInteger readsAfterClose;
        private volatile boolean closed;

        SlowFailingRavv(AtomicInteger activeReads, AtomicInteger readsAfterClose) {
            this.activeReads = activeReads;
            this.readsAfterClose = readsAfterClose;
        }

        @Override
        public int size() {
            return SIZE;
        }

        @Override
        public int dimension() {
            return 2;
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            if (node == 0) {
                throw new IllegalStateException("simulated read failure");
            }
            activeReads.incrementAndGet();
            try {
                Thread.sleep(20);
                if (closed) {
                    readsAfterClose.incrementAndGet();
                }
                var v = vts.createFloatVector(2);
                v.set(0, node);
                v.set(1, -node);
                return v;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } finally {
                activeReads.decrementAndGet();
            }
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return new SlowFailingRavv(activeReads, readsAfterClose);
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    @Test
    public void failedBuildClosesCopiesOnlyAfterAllWorkersStop() throws Exception {
        AtomicInteger activeReads = new AtomicInteger();
        AtomicInteger readsAfterClose = new AtomicInteger();
        var ravv = new SlowFailingRavv(activeReads, readsAfterClose);
        // Score against a plain in-memory copy so only the build loop's reads are slow.
        var inMemory = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < SlowFailingRavv.SIZE; i++) {
            var v = vts.createFloatVector(2);
            v.set(0, i);
            v.set(1, -i);
            inMemory.add(v);
        }
        var bsp = BuildScoreProvider.randomAccessScoreProvider(new ListRandomAccessVectorValues(inMemory, 2),
                                                               VectorSimilarityFunction.EUCLIDEAN);
        var pool = new ForkJoinPool(4);
        try (var builder = GraphIndexBuilder.builder(bsp, 2, 8)
                .withBeamWidth(16)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withSimdExecutor(pool)
                .withParallelExecutor(pool)
                .build()) {
            assertThrows(RuntimeException.class, () -> builder.build(ravv));
            assertEquals("workers still reading after build() returned", 0, activeReads.get());
            assertEquals("copy closed during an active read", 0, readsAfterClose.get());
        } finally {
            pool.shutdownNow();
        }
    }
}
