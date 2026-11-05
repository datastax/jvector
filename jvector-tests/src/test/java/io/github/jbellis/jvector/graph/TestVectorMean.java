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

import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class TestVectorMean {
    private static VectorFloat<?> vector(float... values) {
        return VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(values);
    }

    private static class BorrowedValues implements RandomAccessVectorValues {
        final List<VectorFloat<?>> vectors;
        final AtomicInteger reads;
        final VectorFloat<?> scratch;
        BorrowedValues(List<VectorFloat<?>> vectors, AtomicInteger reads) {
            this.vectors = vectors;
            this.reads = reads;
            scratch = vector(new float[vectors.get(0).length()]);
        }
        public int size() { return vectors.size(); }
        public int dimension() { return scratch.length(); }
        public boolean isValueShared() { return true; }
        public RandomAccessVectorValues copy() { return new BorrowedValues(vectors, reads); }
        public VectorFloat<?> getVector(int ordinal) {
            reads.incrementAndGet();
            var v = vectors.get(ordinal);
            if (v == null) return null;
            scratch.copyFrom(v, 0, 0, dimension());
            return scratch;
        }
    }

    @Test public void fullMeanMatchesExistingNVQWithoutAnotherRead() {
        var input = List.of(vector(1, 2), vector(3, 6), vector(5, 4));
        var reads = new AtomicInteger();
        var source = new BorrowedValues(input, reads);
        var collector = new VectorMean(3, 2, 3, 7);
        var pool = new ForkJoinPool(3);
        try { VectorAccess.forEach(collector.duringScan(source), pool, (i, v) -> {}); }
        finally { pool.shutdown(); }
        assertEquals(3, reads.get());
        assertEquals(3, collector.count());
        var mean = collector.mean();
        var reused = NVQuantization.create(mean, 1);
        assertEquals(3, reads.get());
        var original = NVQuantization.compute(source, 1);
        for (int d = 0; d < 2; d++) assertEquals(original.globalMean.get(d), reused.globalMean.get(d), 1e-6);
        mean.set(0, -100);
        assertEquals(3, collector.mean().get(0), 0);
    }

    @Test public void deterministicSampleIsAccurateAndIgnoresRepeatedAccess() {
        int n = 100000, dimensions = 4;
        var one = new VectorMean(n, dimensions);
        var two = new VectorMean(n, dimensions);
        var v = vector(new float[dimensions]);
        var random = new Random(19);
        double[] exact = new double[dimensions];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dimensions; d++) {
                float value = (float) (random.nextGaussian() + d);
                v.set(d, value);
                exact[d] += value;
            }
            one.accept(i, v);
            one.accept(i, v); // repetitions never weight a sample twice
            two.accept(i, v);
        }
        assertEquals(VectorMean.DEFAULT_SAMPLE_SIZE, one.count());
        for (int d = 0; d < dimensions; d++) {
            assertEquals(one.mean().get(d), two.mean().get(d), 0);
            assertEquals(exact[d] / n, one.mean().get(d), .025);
        }
    }

    @Test public void sampledMeanRetainsNVQReconstructionQuality() {
        int n = 20000, dimensions = 16;
        var random = new Random(28);
        var inputs = new ArrayList<VectorFloat<?>>();
        var collector = new VectorMean(n, dimensions);
        for (int i = 0; i < n; i++) {
            var v = vector(new float[dimensions]);
            for (int d = 0; d < dimensions; d++) v.set(d, (float) (random.nextGaussian() + d));
            inputs.add(v);
            collector.accept(i, v);
        }
        var full = NVQuantization.compute(new ListRandomAccessVectorValues(inputs, dimensions), 2);
        var sampled = NVQuantization.create(collector.mean(), 2);
        double fullError = 0, sampledError = 0;
        for (int i = 0; i < 100; i++) {
            fullError += full.reconstructionError(inputs.get(i));
            sampledError += sampled.reconstructionError(inputs.get(i));
        }
        assertTrue("Sampled mean changed NVQ reconstruction error materially: " + sampledError + " vs " + fullError,
                sampledError <= fullError * 1.05);
    }

    @Test public void pointReadsDoNotCountAndIncompleteScanFails() {
        var source = new BorrowedValues(List.of(vector(1), vector(2), vector(3)), new AtomicInteger());
        var mean = new VectorMean(3, 1, 3, 0);
        var scan = mean.duringScan(source);
        for (int i = 0; i < 3; i++) scan.getVector(i);
        assertEquals(0, mean.count());
        assertThrows(IllegalStateException.class, mean::mean);
        try (var cursor = VectorAccess.openRange(scan.copy(), 0, 2)) { while (cursor.next()) {} }
        assertFalse(mean.isComplete());
        try (var cursor = VectorAccess.openSelection(scan, new int[]{2, 0, 2}, 0, 3)) { while (cursor.next()) {} }
        assertEquals(3, mean.count());
        assertEquals(2, mean.mean().get(0), 0);
    }

    @Test public void databasePushInputNeedsNoLoaderOrWrapper() {
        var mean = new VectorMean(4, 2, 4, 0);
        mean.accept(3, vector(7, 8));
        mean.accept(1, vector(3, 4));
        mean.accept(0, vector(1, 2));
        mean.accept(2, vector(5, 6));
        assertEquals(4, mean.mean().get(0), 0);
        assertEquals(5, mean.mean().get(1), 0);
    }

    @Test public void invalidValuesAndMissingInputsAreHandledExplicitly() {
        assertThrows(IllegalArgumentException.class, () -> new VectorMean(0, 2));
        assertThrows(IllegalArgumentException.class, () -> new VectorMean(1, 0));
        var mean = new VectorMean(2, 2, 2, 0);
        assertThrows(IllegalArgumentException.class, () -> mean.accept(0, vector(1)));
        assertThrows(IllegalArgumentException.class, () -> mean.accept(0, vector(1, Float.NaN)));
        assertThrows(IllegalArgumentException.class, () -> mean.accept(0, vector(1, Float.POSITIVE_INFINITY)));
        mean.accept(0, vector(1, 2));
        mean.accept(1, null);
        assertTrue(mean.isComplete());
        assertEquals(1, mean.count());
        assertEquals(1, mean.mean().get(0), 0);
        var empty = new VectorMean(1, 1);
        empty.accept(0, null);
        assertThrows(IllegalStateException.class, empty::mean);
    }

    @Test public void existingEncodersCollectOriginalCoordinatesWithOneReadEach() throws Exception {
        var input = new ArrayList<VectorFloat<?>>();
        var random = new Random(81);
        for (int i = 0; i < 128; i++) {
            var v = vector(new float[16]);
            for (int d = 0; d < 16; d++) v.set(d, random.nextFloat() + 2);
            input.add(v);
        }
        var resident = new ListRandomAccessVectorValues(input, 16);
        var pq = ProductQuantization.compute(resident, 2, 4, true);
        var ash = AsymmetricHashing.initialize(resident, AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 8 * 2, 1, 2);
        var pool = new ForkJoinPool(4);
        try {
            for (var compressor : new VectorCompressor<?>[]{pq, ash, new BinaryQuantization(16)}) {
                var reads = new AtomicInteger();
                var source = new BorrowedValues(input, reads);
                var collector = new VectorMean(128, 16, 128, 0);
                compressor.encodeAll(collector.duringScan(source), pool);
                assertEquals(compressor.getClass().getName(), 128, reads.get());
                assertEquals(128, collector.count());
                var nvq = NVQuantization.create(collector.mean(), 2);
                assertEquals(128, reads.get());
                var full = NVQuantization.compute(source, 2);
                for (int d = 0; d < 16; d++) assertEquals(full.globalMean.get(d), nvq.globalMean.get(d), 2e-6);
            }
        } finally { pool.shutdown(); }
    }

    @Test public void rangeSourceKeepsBatchAccessAndClosesCursors() {
        var reads = new AtomicInteger();
        var closes = new AtomicInteger();
        var values = new BorrowedValues(List.of(vector(2), vector(4), vector(6)), reads);
        BatchedVectorValues batched = new BatchedVectorValues() {
            public int size() { return values.size(); }
            public int dimension() { return values.dimension(); }
            public boolean isValueShared() { return true; }
            public RandomAccessVectorValues copy() { return this; }
            public VectorFloat<?> getVector(int ordinal) { throw new AssertionError("Point read used"); }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) { throw new AssertionError(); }
            public VectorCursor openRange(int start, int end) {
                var cursor = VectorAccess.openRange(values, start, end);
                return new VectorCursor() {
                    public boolean next() { return cursor.next(); }
                    public int ordinal() { return cursor.ordinal(); }
                    public VectorFloat<?> vector() { return cursor.vector(); }
                    public boolean isValueShared() { return true; }
                    public void close() { closes.incrementAndGet(); cursor.close(); }
                };
            }
        };
        var mean = new VectorMean(3, 1);
        try (var cursor = mean.duringScan(batched).openRange(0, 3)) { while (cursor.next()) {} }
        assertEquals(3, reads.get());
        assertEquals(1, closes.get());
        assertEquals(4, mean.mean().get(0), 0);
    }
}
