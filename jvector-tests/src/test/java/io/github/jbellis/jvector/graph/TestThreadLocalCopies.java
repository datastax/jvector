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
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

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

    private static CloseTrackingRavv copiesThatFailOnClose(AtomicInteger closed, Supplier<RuntimeException> failure) {
        return new CloseTrackingRavv(closed) {
            @Override
            public RandomAccessVectorValues copy() {
                return new CloseTrackingRavv(closed, false) {
                    @Override
                    public void close() {
                        super.close();
                        throw failure.get();
                    }
                };
            }
        };
    }

    @Test
    public void throwingCopiesAreAllClosedAndDropped() throws Exception {
        var closed = new AtomicInteger();
        var source = copiesThatFailOnClose(closed, () -> new IllegalStateException("close failed"));
        var supplier = source.threadLocalSupplier();
        var first = supplier.get();
        var worker = new Thread(supplier::get);
        worker.start();
        worker.join();

        var failure = assertThrows(IllegalStateException.class, ((AutoCloseable) supplier)::close);
        assertEquals(2, closed.get());
        assertEquals(1, failure.getSuppressed().length);
        assertTrue(failure.getSuppressed()[0] instanceof IllegalStateException);
        assertNotSame(first, supplier.get());
    }

    @Test
    public void laterErrorTakesPrecedenceDuringClose() throws Exception {
        var closed = new AtomicInteger();
        var exception = new IOException("first close");
        var error = new AssertionError("second close");
        var copies = ExplicitThreadLocal.<AutoCloseable>withInitial(() -> () -> {
            if (closed.getAndIncrement() == 0) {
                throw exception;
            }
            throw error;
        });
        copies.get();
        var worker = new Thread(copies::get);
        worker.start();
        worker.join();

        assertSame(error, assertThrows(AssertionError.class, copies::close));
        assertArrayEquals(new Throwable[] {exception}, error.getSuppressed());
        assertEquals(2, closed.get());
    }

    @Test
    public void successfulBuildPropagatesUncheckedCloseFailure() throws Exception {
        var closed = new AtomicInteger();
        var failure = new IllegalStateException("close failed");
        var source = copiesThatFailOnClose(closed, () -> failure);
        var scores = BuildScoreProvider.randomAccessScoreProvider(
                new ListRandomAccessVectorValues(List.of(source.getVector(0)), 1),
                VectorSimilarityFunction.EUCLIDEAN);
        try (var builder = GraphIndexBuilder.builder(scores, 1, 8)
                .withBeamWidth(16)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .build()) {
            assertSame(failure, assertThrows(IllegalStateException.class, () -> builder.build(source)));
            assertEquals(1, builder.getGraph().size(0));
            assertEquals(1, closed.get());
        }
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

    @Test
    public void closingSupplierClosesCopiesInsideRemappedRavv() throws Exception {
        AtomicInteger closed = new AtomicInteger();
        var remapped = new RemappedRandomAccessVectorValues(new CloseTrackingRavv(closed), new int[] {0});
        var supplier = remapped.threadLocalSupplier();

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
    public void remappedCopyOfUnsharedRavvDoesNotCloseTheSource() throws Exception {
        AtomicInteger closed = new AtomicInteger();
        RandomAccessVectorValues unshared = new CloseTrackingRavv(closed, false) {
            @Override
            public boolean isValueShared() {
                return false;
            }

            @Override
            public RandomAccessVectorValues copy() {
                return this;
            }
        };
        var copy = new RemappedRandomAccessVectorValues(unshared, new int[] {0}).copy();
        if (copy instanceof AutoCloseable) {
            ((AutoCloseable) copy).close();
        }
        assertEquals(0, closed.get());
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
            throw new AssertionError("simulated close failure");
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
            var failure = assertThrows(IllegalStateException.class, () -> builder.build(ravv));
            assertTrue(failure.toString(), failure.toString().contains("simulated read failure"));
            assertEquals(1, failure.getSuppressed().length);
            assertTrue(failure.getSuppressed()[0] instanceof AssertionError);
            assertEquals("workers still reading after build() returned", 0, activeReads.get());
            assertEquals("copy closed during an active read", 0, readsAfterClose.get());
        } finally {
            pool.shutdownNow();
        }
    }

    /** Shared int8 RAVV that counts the copies it makes and their close() calls; the source itself never counts. */
    static class CloseTrackingByteRavv implements RandomAccessByteVectorValues, AutoCloseable {
        static final int SIZE = 16;
        private final AtomicInteger createdCount;
        private final AtomicInteger closedCount;
        private final boolean source;

        CloseTrackingByteRavv(AtomicInteger createdCount, AtomicInteger closedCount, boolean source) {
            this.createdCount = createdCount;
            this.closedCount = closedCount;
            this.source = source;
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
        public ByteSequence<?> getVector(int nodeId) {
            return byteVector(nodeId);
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessByteVectorValues copy() {
            createdCount.incrementAndGet();
            return new CloseTrackingByteRavv(createdCount, closedCount, false);
        }

        @Override
        public void close() {
            if (!source) {
                closedCount.incrementAndGet();
            }
        }
    }

    private static ByteSequence<?> byteVector(int node) {
        var v = vts.createByteSequence(2);
        v.set(0, (byte) node);
        v.set(1, (byte) -node);
        return v;
    }

    @Test
    public void closingByteSupplierClosesAllThreadLocalCopies() throws Exception {
        AtomicInteger closed = new AtomicInteger();
        var supplier = new CloseTrackingByteRavv(new AtomicInteger(), closed, true).threadLocalSupplier();

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
    public void byteBuildClosesItsThreadLocalCopies() throws Exception {
        AtomicInteger created = new AtomicInteger();
        AtomicInteger closed = new AtomicInteger();
        var inMemory = new ArrayList<ByteSequence<?>>();
        for (int i = 0; i < CloseTrackingByteRavv.SIZE; i++) {
            inMemory.add(byteVector(i));
        }
        var bsp = BuildScoreProvider.byteVectorScoreProvider(new ListRandomAccessByteVectorValues(inMemory, 2),
                                                             ByteVectorSimilarityFunction.EUCLIDEAN);
        var pool = new ForkJoinPool(2);
        try (var builder = GraphIndexBuilder.builder(bsp, 2, 8)
                .withBeamWidth(16)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withSimdExecutor(pool)
                .withParallelExecutor(pool)
                .build()) {
            builder.build(new CloseTrackingByteRavv(created, closed, true));
            assertEquals(CloseTrackingByteRavv.SIZE, builder.getGraph().size(0));
            assertTrue("build() made no per-thread copies", created.get() > 0);
            assertEquals("per-thread copies not closed after build()", created.get(), closed.get());
        } finally {
            pool.shutdownNow();
        }
    }
}
