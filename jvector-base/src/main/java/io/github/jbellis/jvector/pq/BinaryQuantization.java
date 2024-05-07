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

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Binary Quantization of float vectors: each float is compressed to a single bit,
 * and similarity is computed with a simple Hamming distance.
 */
public class BinaryQuantization implements VectorCompressor<long[]> {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimension;

    public BinaryQuantization(int dimension) {
        this.dimension = dimension;
    }

    /**
     * Use BQ constructor instead
     */
    @Deprecated
    public static BinaryQuantization compute(RandomAccessVectorValues ravv) {
        return compute(ravv, ForkJoinPool.commonPool());
    }

    /**
     * Use BQ constructor instead
     */
    @Deprecated
    public static BinaryQuantization compute(RandomAccessVectorValues ravv, ForkJoinPool parallelExecutor) {
        return new BinaryQuantization(ravv.dimension());
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new BQVectors(this, (long[][]) compressedVectors);
    }

    @Override
    public long[][] encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> IntStream.range(0, ravv.size())
                .parallel()
                .mapToObj(i -> encode(ravv.getVector(i)))
                .toArray(long[][]::new))
                .join();
    }

    /**
     * Encodes the input vector
     *
     * @return one bit per original f32
     */
    @Override
    public long[] encode(VectorFloat<?> v) {
        int M = (int) Math.ceil(v.length() / 64.0);
        long[] encoded = new long[M];
        for (int i = 0; i < M; i++) {
            long bits = 0;
            for (int j = 0; j < 64; j++) {
                int idx = i * 64 + j;
                if (idx >= v.length()) {
                    break;
                }
                if (v.get(idx) > 0) {
                    bits |= 1L << j;
                }
            }
            encoded[i] = bits;
        }
        return encoded;
    }

    @Override
    public int compressorSize() {
        return Integer.BYTES + dimension * Float.BYTES;
    }

    @Override
    public int compressedVectorSize() {
        int M = (int) Math.ceil(dimension / 64.0);
        return Long.BYTES * M;
    }

    @Override
    public void write(DataOutput out, int version) throws IOException {
        out.writeInt(dimension);
        // We used to record the center of the dataset but this actually degrades performance.
        // Write a zero vector to maintain compatibility.
        vts.writeFloatVector(out, vts.createFloatVector(dimension));
    }

    public int getOriginalDimension() {
        return dimension;
    }

    public static BinaryQuantization load(RandomAccessReader in) throws IOException {
        int dimension = in.readInt();
        // We used to record the center of the dataset but this actually degrades performance.
        // Read it and discard it.
        vts.readFloatVector(in, dimension);
        return new BinaryQuantization(dimension);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BinaryQuantization that = (BinaryQuantization) o;
        return Objects.equals(dimension, that.dimension);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(dimension);
    }

    @Override
    public String toString() {
        return "BinaryQuantization";
    }
}
