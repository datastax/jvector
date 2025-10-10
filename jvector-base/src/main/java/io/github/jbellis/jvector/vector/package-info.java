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

/**
 * Provides vector data structures and operations for high-performance vector similarity search.
 * <p>
 * This package contains the core abstractions and implementations for representing and manipulating
 * vectors in JVector. The design supports both standard array-based implementations and pluggable
 * SIMD-accelerated operations through the {@link io.github.jbellis.jvector.vector.VectorizationProvider}
 * interface.
 * <p>
 * <b>Key Components:</b>
 * <ul>
 *   <li><b>Vector Representations</b> - The {@link io.github.jbellis.jvector.vector.types.VectorFloat}
 *       interface (in the types subpackage) defines the contract for floating-point vectors, with
 *       {@link io.github.jbellis.jvector.vector.ArrayVectorFloat} providing the standard array-based
 *       implementation.</li>
 *   <li><b>Byte Sequences</b> - The {@link io.github.jbellis.jvector.vector.types.ByteSequence}
 *       interface (in the types subpackage) represents sequences of bytes, used for compressed vectors
 *       and other byte-level operations. Implementations include
 *       {@link io.github.jbellis.jvector.vector.ArrayByteSequence} and
 *       {@link io.github.jbellis.jvector.vector.ArraySliceByteSequence}.</li>
 *   <li><b>Vectorization</b> - {@link io.github.jbellis.jvector.vector.VectorizationProvider} defines
 *       the interface for SIMD-accelerated vector operations. The default implementation is
 *       {@link io.github.jbellis.jvector.vector.DefaultVectorizationProvider}, which uses standard
 *       Java array operations. SIMD-accelerated implementations are provided in separate modules
 *       using the Panama Vector API.</li>
 *   <li><b>Similarity Functions</b> - {@link io.github.jbellis.jvector.vector.VectorSimilarityFunction}
 *       enumerates the supported similarity metrics (DOT_PRODUCT, COSINE, EUCLIDEAN) and provides
 *       methods for computing vector similarity scores.</li>
 *   <li><b>Vector Utilities</b> - {@link io.github.jbellis.jvector.vector.VectorUtil} provides static
 *       utility methods for common vector operations, delegating to the appropriate
 *       {@link io.github.jbellis.jvector.vector.VectorUtilSupport} implementation for performance.</li>
 *   <li><b>Matrix Operations</b> - {@link io.github.jbellis.jvector.vector.Matrix} provides matrix
 *       operations for vectors, used in quantization and other linear algebra operations.</li>
 * </ul>
 * <p>
 * <b>Usage Example:</b>
 * <pre>{@code
 * // Create a vector
 * VectorFloat<?> vector = ArrayVectorFloat.create(new float[]{1.0f, 2.0f, 3.0f});
 *
 * // Compute similarity
 * float similarity = VectorSimilarityFunction.COSINE.compare(vector1, vector2);
 *
 * // Use vector utilities
 * float norm = VectorUtil.norm(vector);
 * }</pre>
 *
 * @see io.github.jbellis.jvector.vector.types
 * @see io.github.jbellis.jvector.vector.VectorizationProvider
 * @see io.github.jbellis.jvector.vector.VectorSimilarityFunction
 * @see io.github.jbellis.jvector.vector.VectorUtil
 */
package io.github.jbellis.jvector.vector;
