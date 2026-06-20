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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

public class TestRandomAccessVectorValues {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void threadLocalSupplierClosesSharedCopies() throws Exception {
        var ravv = new CloseTrackingRandomAccessVectorValues();
        var supplier = ravv.threadLocalSupplier();

        assertTrue("thread-local suppliers should expose a close hook", supplier instanceof AutoCloseable);
        assertNotSame(ravv, supplier.get());

        ((AutoCloseable) supplier).close();

        assertEquals(1, ravv.copyCount());
        assertEquals(1, ravv.copyCloseCount());
        assertEquals(0, ravv.originalCloseCount());
    }

    @Test
    public void graphIndexBuilderClosesThreadLocalVectorCopies() throws IOException {
        var ravv = new CloseTrackingRandomAccessVectorValues();

        try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.EUCLIDEAN, 2, 10, 1.0f, 1.0f, false)) {
            builder.build(ravv);
        }

        assertTrue("graph build should create thread-local RAVV copies", ravv.copyCount() > 0);
        assertEquals("all thread-local copies should be closed", ravv.copyCount(), ravv.copyCloseCount());
        assertEquals(0, ravv.originalCloseCount());
    }

    private static class CloseTrackingRandomAccessVectorValues implements RandomAccessVectorValues, AutoCloseable {
        private final List<VectorFloat<?>> vectors;
        private final AtomicInteger copyCount;
        private final AtomicInteger copyCloseCount;
        private final AtomicInteger originalCloseCount;
        private final boolean original;

        CloseTrackingRandomAccessVectorValues() {
            this(List.of(
                    vts.createFloatVector(new float[] {0, 0}),
                    vts.createFloatVector(new float[] {1, 0}),
                    vts.createFloatVector(new float[] {2, 0}),
                    vts.createFloatVector(new float[] {3, 0})),
                 new AtomicInteger(),
                 new AtomicInteger(),
                 new AtomicInteger(),
                 true);
        }

        private CloseTrackingRandomAccessVectorValues(List<VectorFloat<?>> vectors,
                                                      AtomicInteger copyCount,
                                                      AtomicInteger copyCloseCount,
                                                      AtomicInteger originalCloseCount,
                                                      boolean original) {
            this.vectors = vectors;
            this.copyCount = copyCount;
            this.copyCloseCount = copyCloseCount;
            this.originalCloseCount = originalCloseCount;
            this.original = original;
        }

        @Override
        public int size() {
            return vectors.size();
        }

        @Override
        public int dimension() {
            return 2;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            return vectors.get(nodeId);
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessVectorValues copy() {
            copyCount.incrementAndGet();
            return new CloseTrackingRandomAccessVectorValues(vectors, copyCount, copyCloseCount, originalCloseCount, false);
        }

        @Override
        public void close() {
            if (original) {
                originalCloseCount.incrementAndGet();
            } else {
                copyCloseCount.incrementAndGet();
            }
        }

        int copyCount() {
            return copyCount.get();
        }

        int copyCloseCount() {
            return copyCloseCount.get();
        }

        int originalCloseCount() {
            return originalCloseCount.get();
        }
    }
}
