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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Random;

/**
 * Collects an original-coordinate mean during an existing, explicit input scan.
 * This utility does not read vectors. Database integrations may feed already-read
 * vectors to {@link #accept}; bulk encoders can use {@link #duringScan} instead.
 * Neither loaders nor vector compressors need to implement a statistics interface.
 */
public final class VectorMean {
    /** A bounded uniform sample, independent of the input size and quantizer. */
    public static final int DEFAULT_SAMPLE_SIZE = 16384;

    private final int size;
    private final int dimension;
    private final int[] ordinals;
    private final boolean[] seen;
    private final double[] sum;
    private int visited;
    private int count;

    public VectorMean(int size, int dimension) {
        this(size, dimension, DEFAULT_SAMPLE_SIZE, 0);
    }

    /** maxSamples >= size collects the full mean; otherwise samples without replacement. */
    public VectorMean(int size, int dimension, int maxSamples, long seed) {
        if (size <= 0 || dimension <= 0 || maxSamples <= 0)
            throw new IllegalArgumentException("Positive size, dimension and sample limit required");
        this.size = size;
        this.dimension = dimension;
        int n = Math.min(size, maxSamples);
        ordinals = new int[n];
        // Floyd's algorithm selects in O(sample size), without scanning all ordinals.
        var selected = new HashSet<Integer>();
        var random = new Random(seed);
        int i = 0;
        for (int j = size - n; j < size; j++) {
            int ordinal = random.nextInt(j + 1);
            if (!selected.add(ordinal)) {
                selected.add(j);
                ordinal = j;
            }
            ordinals[i++] = ordinal;
        }
        Arrays.sort(ordinals);
        seen = new boolean[n];
        sum = new double[dimension];
    }

    /**
     * Observe an already-read, untransformed vector at its source ordinal. A borrowed
     * buffer is consumed immediately, never retained. Repeated ordinals are ignored.
     * Null/missing vectors are excluded from the mean, but count toward scan completion.
     */
    public void accept(int ordinal, VectorFloat<?> vector) {
        Objects.checkIndex(ordinal, size);
        int sample = Arrays.binarySearch(ordinals, ordinal);
        if (sample >= 0) acceptSample(sample, vector);
    }

    private synchronized void acceptSample(int sample, VectorFloat<?> vector) {
        if (seen[sample]) return;
        if (vector != null) {
            if (vector.length() != dimension) throw new IllegalArgumentException("Vector dimension mismatch");
            // Validate before mutating state, including on a failed/retried scan.
            for (int d = 0; d < dimension; d++) {
                if (!Float.isFinite(vector.get(d))) throw new IllegalArgumentException("Non-finite vector component");
            }
            for (int d = 0; d < dimension; d++) sum[d] += vector.get(d);
            count++;
        }
        seen[sample] = true;
        visited++;
    }

    public synchronized boolean isComplete() { return visited == ordinals.length; }
    public synchronized int count() { return count; }
    public int sampleSize() { return ordinals.length; }

    /** Return an owned snapshot. Fails if the selected input scan is incomplete or empty. */
    public synchronized VectorFloat<?> mean() {
        if (!isComplete()) throw new IllegalStateException("Incomplete mean collection");
        if (count == 0) throw new IllegalStateException("No present vectors in mean sample");
        var mean = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimension);
        for (int d = 0; d < dimension; d++) mean.set(d, (float) (sum[d] / count));
        return mean;
    }

    /**
     * Scope collection to ONE explicit bulk scan. Range/selection cursors observe
     * vectors before handing them to the consumer. Point reads are delegated without
     * collection, so arbitrary training/search accesses cannot become statistics.
     * The wrapper does not own the source; closing a cursor still releases its reads.
     * Do not pass this wrapper to training or reuse it for another dataset.
     */
    public BatchedVectorValues duringScan(RandomAccessVectorValues source) {
        if (source.size() != size || source.dimension() != dimension)
            throw new IllegalArgumentException("Mean/source shape mismatch");
        return new BatchedVectorValues() {
            public int size() { return source.size(); }
            public int dimension() { return source.dimension(); }
            public boolean isValueShared() { return source.isValueShared(); }
            public VectorFloat<?> getVector(int ordinal) { return source.getVector(ordinal); }
            public void getVectorInto(int ordinal, VectorFloat<?> destination, int offset) {
                source.getVectorInto(ordinal, destination, offset);
            }
            public RandomAccessVectorValues copy() { return duringScan(source.copy()); }
            public VectorCursor openRange(int start, int end) {
                var cursor = VectorAccess.openRange(source, start, end);
                int first = Arrays.binarySearch(ordinals, start);
                return observe(cursor, first >= 0 ? first : -first - 1, true);
            }
            public VectorCursor openSelection(int[] requested, int offset, int count) {
                return observe(VectorAccess.openSelection(source, requested, offset, count), 0, false);
            }
        };
    }

    private VectorCursor observe(VectorCursor cursor, int firstSample, boolean range) {
        return new VectorCursor() {
            int nextSample = firstSample;
            public boolean next() {
                try {
                    if (!cursor.next()) return false;
                    int ordinal = cursor.ordinal();
                    if (range) {
                        if (nextSample < ordinals.length && ordinals[nextSample] == ordinal)
                            acceptSample(nextSample++, cursor.vector());
                    } else {
                        accept(ordinal, cursor.vector());
                    }
                    return true;
                } catch (RuntimeException | Error failure) {
                    try { close(); } catch (RuntimeException | Error closing) { failure.addSuppressed(closing); }
                    throw failure;
                }
            }
            public int ordinal() { return cursor.ordinal(); }
            public VectorFloat<?> vector() { return cursor.vector(); }
            public boolean isValueShared() { return cursor.isValueShared(); }
            public void close() { cursor.close(); }
        };
    }
}
