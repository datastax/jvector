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

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class TestConcurrentBufferedGraphConstruction {
    private static final class Reads {
        final CountDownLatch rendezvous = new CountDownLatch(2);
        final AtomicInteger tickets = new AtomicInteger();
        final AtomicInteger active = new AtomicInteger();
        final AtomicInteger maximum = new AtomicInteger();
        final AtomicInteger copies = new AtomicInteger();
    }

    private static final class CheckedSource implements RandomAccessVectorValues {
        final RandomAccessVectorValues source, resident;
        final Reads reads;

        CheckedSource(RandomAccessVectorValues source, RandomAccessVectorValues resident, Reads reads) {
            this.source = source;
            this.resident = resident;
            this.reads = reads;
        }
        public int size() { return source.size(); }
        public int dimension() { return source.dimension(); }
        public boolean isValueShared() { return source.isValueShared(); }
        public RandomAccessVectorValues copy() {
            reads.copies.incrementAndGet();
            return new CheckedSource(source.copy(), resident, reads);
        }
        public VectorFloat<?> getVector(int ordinal) {
            reads.maximum.accumulateAndGet(reads.active.incrementAndGet(), Math::max);
            try {
                var vector = source.getVector(ordinal);
                // Force overlapping borrowed reads; an accidentally shared scratch buffer is observable.
                if (ForkJoinTask.inForkJoinPool() && reads.tickets.getAndIncrement() < 2) {
                    reads.rendezvous.countDown();
                    try { assertTrue("construction reads did not overlap", reads.rendezvous.await(10, TimeUnit.SECONDS)); }
                    catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new AssertionError(e); }
                }
                var expected = resident.getVector(ordinal);
                for (int d = 0; d < dimension(); d++)
                    assertEquals("borrowed vector changed at ordinal=" + ordinal + ", dimension=" + d,
                            Float.floatToIntBits(expected.get(d)), Float.floatToIntBits(vector.get(d)));
                return vector;
            } finally {
                reads.active.decrementAndGet();
            }
        }
    }

    @Test(timeout = 180000)
    public void exactPqAndBinaryBuildsReadIndependentBorrowedVectorsConcurrently() throws Exception {
        var support = VectorizationProvider.getInstance().getVectorTypeSupport();
        var pool = new ForkJoinPool(3);
        try {
            for (int dimension : new int[] {17, 768}) {
                int count = 257;
                var vectors = new ArrayList<VectorFloat<?>>();
                var bytes = ByteBuffer.allocate(count * (dimension + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
                var random = new Random(42);
                for (int n = 0; n < count; n++) {
                    var vector = support.createFloatVector(dimension);
                    for (int d = 0; d < dimension; d++) vector.set(d, random.nextFloat() * 2 - 1);
                    VectorUtil.l2normalize(vector);
                    vectors.add(vector);
                    bytes.putInt(dimension);
                    for (int d = 0; d < dimension; d++) bytes.putFloat(vector.get(d));
                }
                var path = Files.createTempFile("concurrent-graph-input", ".fvecs");
                Files.write(path, bytes.array());
                var resident = new ListRandomAccessVectorValues(vectors, dimension);
                try (var file = FvecFileVectorValues.open(path)) {
                    var pq = ProductQuantization.compute(resident, Math.max(1, dimension / 8), 16, false,
                            KMeansPlusPlusClusterer.UNWEIGHTED, pool, pool);
                    var pqv = pq.encodeAll(file, pool);
                    var bqv = (BQVectors) new BinaryQuantization(dimension).encodeAll(file, pool);
                    for (var metric : VectorSimilarityFunction.values()) {
                        for (int mode = 0; mode < 3; mode++) {
                            var reads = new Reads();
                            var source = new CheckedSource(file, resident, reads);
                            var scores = mode == 0 ? BuildScoreProvider.randomAccessScoreProvider(source, metric)
                                    : mode == 1 ? BuildScoreProvider.pqBuildScoreProvider(metric, pqv)
                                    : BuildScoreProvider.bqBuildScoreProvider(bqv);
                            try (var builder = new GraphIndexBuilder(scores, dimension, 8, 64, 1.2f, 1.2f,
                                    true, true, pool, pool)) {
                                var graph = builder.build(source);
                                assertEquals(count, graph.size(0));
                                assertTrue("source copies were not used", reads.copies.get() >= 2);
                                assertTrue("construction was not concurrent", reads.maximum.get() >= 2);
                                try (var view = graph.getView()) {
                                    for (int level = 0; level <= graph.getMaxLevel(); level++) {
                                        for (var nodes = graph.getNodes(level); nodes.hasNext(); ) {
                                            int node = nodes.nextInt();
                                            var neighbors = view.getNeighborsIterator(level, node);
                                            assertTrue(neighbors.size() <= graph.getDegree(level));
                                            while (neighbors.hasNext()) {
                                                int neighbor = neighbors.nextInt();
                                                assertTrue(neighbor >= 0 && neighbor < count);
                                                assertNotEquals(node, neighbor);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } finally { Files.deleteIfExists(path); }
            }
        } finally { pool.shutdown(); }
    }
}
