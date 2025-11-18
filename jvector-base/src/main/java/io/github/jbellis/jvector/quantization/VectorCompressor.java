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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Interface for vector compression.  T is the encoded (compressed) vector type;
 * it will be an array type.
 */
public interface VectorCompressor<T> {

    default CompressedVectors encodeAll(RandomAccessVectorValues ravv) {
        return encodeAll(ravv, PhysicalCoreExecutor.pool());
    }

    /**
     * Encode all vectors in the RandomAccessVectorValues. If the RandomAccessVectorValues
     * has a missing vector for a given ordinal, the value will be encoded as a zero vector.
     * @param ravv RandomAccessVectorValues to encode
     * @param simdExecutor ForkJoinPool to use for SIMD operations
     * @return CompressedVectors containing the encoded vectors
     */
    CompressedVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor);

    T encode(VectorFloat<?> v);

    void encodeTo(VectorFloat<?> v, T dest);

    /**
     * @param out DataOutput to write to
     * @param version serialization version.  Versions 2 and 3 are supported
     */
    void write(DataOutput out, int version) throws IOException;

    /** Write with the current serialization version */
    default void write(DataOutput out) throws IOException {
        write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    /**
     * @param compressedVectors must match the type T for this VectorCompressor, but
     *                          it is declared as Object because we want callers to be able to use this
     *                          without committing to a specific type T.
     */
    @Deprecated
    CompressedVectors createCompressedVectors(Object[] compressedVectors);

    /** the size of the serialized compressor itself (NOT the size of compressed vectors) */
    int compressorSize();

    /** the size of a compressed vector */
    int compressedVectorSize();

    /**
     * Compute the mean squared error (MSE) for the given vector.
     * </p>
     * MSE = (sum of squared errors over all dimensions) / (number of dimensions)
     * @param vector the vector to compute the reconstruction error for
     * @return the reconstruction error for the given vector
     */
    double reconstructionError(VectorFloat<?> vector);

    /**
     * Compute the mean squared error (MSE) for each vector in the stream.
     * </p>
     * MSE = (sum of squared errors over all dimensions) / (number of dimensions)
     * @param ravv the vectors to compute the reconstruction error for
     * @return the reconstruction error for each vector
     */
    default double[] reconstructionErrors(RandomAccessVectorValues ravv)  {
        return IntStream.range(0, ravv.size()).mapToDouble(i -> reconstructionError(ravv.getVector(i))).toArray();
    }

    /**
     * Compute the mean squared error (MSE) for each vector in the stream in parallel.
     * </p>
     * MSE = (sum of squared errors over all dimensions) / (number of dimensions)
     * @param ravv the vectors to compute the reconstruction error for
     * @param simdExecutor the ForkJoinPool to use for SIMD operations
     * @return the reconstruction error for each vector
     */
    default double[] reconstructionErrors(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() ->
                IntStream.range(0, ravv.size()).mapToDouble(i -> reconstructionError(ravv.getVector(i))).toArray()
        ).join();
    }
}
