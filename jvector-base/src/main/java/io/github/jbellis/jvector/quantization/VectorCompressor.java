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

/**
 * Interface for vector compression.  T is the encoded (compressed) vector type;
 * it will be an array type.
 * @param <T> the T type parameter
 */
public interface VectorCompressor<T> {

    /**
     * Encodes all vectors using the default physical core executor pool.
     * @param ravv the vectors to encode
     * @return compressed vectors containing all encoded vectors
     */
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

    /**
     * Encodes a vector.
     * @param v the vector to encode
     * @return the encoded vector
     */
    T encode(VectorFloat<?> v);

    /**
     * Encodes a vector into a destination.
     * @param v the vector to encode
     * @param dest the destination
     */
    void encodeTo(VectorFloat<?> v, T dest);

    /**
     * Writes the compressor to output.
     * @param out DataOutput to write to
     * @param version serialization version.  Versions 2 and 3 are supported
     * @throws IOException if an I/O error occurs
     */
    void write(DataOutput out, int version) throws IOException;

    /**
     * Write with the current serialization version.
     * @param out the output
     * @throws IOException if an I/O error occurs
     */
    default void write(DataOutput out) throws IOException {
        write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    /**
     * Creates compressed vectors.
     * @param compressedVectors must match the type T for this VectorCompressor, but
     *                          it is declared as Object because we want callers to be able to use this
     *                          without committing to a specific type T.
     * @return the compressed vectors
     */
    @Deprecated
    CompressedVectors createCompressedVectors(Object[] compressedVectors);

    /**
     * Returns the size of the serialized compressor itself (NOT the size of compressed vectors).
     * @return the compressor size
     */
    int compressorSize();

    /**
     * Returns the size of a compressed vector.
     * @return the compressed vector size
     */
    int compressedVectorSize();
}
