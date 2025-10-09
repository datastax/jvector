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
 * Provides vector quantization implementations for reducing memory footprint and improving search performance.
 * <p>
 * This package contains multiple quantization techniques:
 * <ul>
 *   <li><b>Binary Quantization (BQ)</b> - Compresses vectors to binary representations using
 *       {@link io.github.jbellis.jvector.quantization.BinaryQuantization}. This provides the highest
 *       compression ratio at the cost of some accuracy. Binary quantized vectors are stored in
 *       {@link io.github.jbellis.jvector.quantization.BQVectors}.</li>
 *   <li><b>Product Quantization (PQ)</b> - Divides vectors into subvectors and quantizes each independently
 *       using {@link io.github.jbellis.jvector.quantization.ProductQuantization}. This balances compression
 *       ratio and accuracy. Product quantized vectors are stored in
 *       {@link io.github.jbellis.jvector.quantization.PQVectors}.</li>
 *   <li><b>Neighborhood Vector Quantization (NVQ)</b> - A variant of PQ that uses neighborhood information
 *       to improve quantization quality, implemented in {@link io.github.jbellis.jvector.quantization.NVQuantization}.
 *       NVQ vectors are stored in {@link io.github.jbellis.jvector.quantization.NVQVectors}.</li>
 * </ul>
 * <p>
 * All quantization methods implement the {@link io.github.jbellis.jvector.quantization.VectorCompressor}
 * interface, which provides methods for encoding vectors and persisting the compressed representation.
 * <p>
 * The {@link io.github.jbellis.jvector.quantization.CompressedVectors} interface represents the
 * compressed form of vectors and provides methods for similarity scoring between compressed vectors
 * and both compressed and uncompressed query vectors.
 * <p>
 * <b>Usage Example:</b>
 * <pre>{@code
 * // Create a Product Quantization compressor with 16 subvectors and 256 clusters per subvector
 * ProductQuantization pq = ProductQuantization.compute(vectors, 16, 256);
 *
 * // Encode all vectors
 * CompressedVectors compressed = pq.encodeAll(vectors);
 *
 * // Perform similarity scoring
 * float score = compressed.score(queryVector, vectorOrdinal);
 * }</pre>
 *
 * @see io.github.jbellis.jvector.quantization.VectorCompressor
 * @see io.github.jbellis.jvector.quantization.CompressedVectors
 * @see io.github.jbellis.jvector.quantization.BinaryQuantization
 * @see io.github.jbellis.jvector.quantization.ProductQuantization
 * @see io.github.jbellis.jvector.quantization.NVQuantization
 */
package io.github.jbellis.jvector.quantization;
