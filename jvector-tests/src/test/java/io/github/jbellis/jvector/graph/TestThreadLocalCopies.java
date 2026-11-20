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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

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
}
