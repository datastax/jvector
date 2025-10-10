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
 * Provides abstractions for vector similarity scoring during graph construction and search.
 * <p>
 * This package defines a layered abstraction for computing similarity scores between vectors,
 * supporting both exact and approximate scoring strategies, optional reranking, and various
 * optimization techniques like quantization and caching.
 *
 * <h2>Scoring Architecture</h2>
 * <p>
 * The package provides a three-level scoring hierarchy:
 * <ol>
 *   <li><b>BuildScoreProvider</b> - Top-level provider configured during graph construction.
 *       Creates search-specific score providers for each node or query.</li>
 *   <li><b>SearchScoreProvider</b> - Per-query or per-node provider that creates actual
 *       score functions and manages approximate/exact scoring strategies.</li>
 *   <li><b>ScoreFunction</b> - Performs the actual similarity computations between vectors.</li>
 * </ol>
 *
 * <h2>Core Interfaces</h2>
 *
 * <h3>Score Providers</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.BuildScoreProvider} - Top-level
 *       interface for creating score providers. Maintains shared state like vector data and
 *       quantization codebooks. Factory methods support various use cases:
 *       <ul>
 *         <li>{@code randomAccessScoreProvider()} - For in-memory vectors</li>
 *         <li>{@code pqBuildScoreProvider()} - For Product Quantization</li>
 *         <li>Other variants for NVQ, binary quantization, and fused approaches</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.SearchScoreProvider} - Per-query
 *       interface that creates score functions. Supports:
 *       <ul>
 *         <li>Approximate scoring (using quantized vectors)</li>
 *         <li>Exact scoring (using full-precision vectors)</li>
 *         <li>Optional reranking to improve precision</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider} - Default
 *       implementation wrapping a single score function.</li>
 * </ul>
 *
 * <h3>Score Functions</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.ScoreFunction} - Core interface for
 *       computing similarity scores. Methods:
 *       <ul>
 *         <li>{@code similarityTo(int)} - Compute similarity to a single node</li>
 *         <li>{@code edgeLoadingSimilarityTo(int)} - Bulk similarity for all neighbors (optional)</li>
 *         <li>{@code isExact()} - Indicates if scores are exact or approximate</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.ScoreFunction.ExactScoreFunction} -
 *       Marker interface for exact scoring implementations.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.ScoreFunction.ApproximateScoreFunction} -
 *       Marker interface for approximate scoring implementations (typically using quantization).</li>
 * </ul>
 *
 * <h3>Utility Classes</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity.CachingVectorValues} - Wrapper that
 *       caches vector access to improve performance when vectors are accessed multiple times.</li>
 * </ul>
 *
 * <h2>Approximate vs. Exact Scoring</h2>
 * <p>
 * JVector supports a two-phase scoring strategy:
 * <ol>
 *   <li><b>Approximate scoring</b>: Used during graph traversal to quickly identify candidates.
 *       Typically uses quantized vectors (PQ, NVQ, or binary quantization) for speed.</li>
 *   <li><b>Exact scoring</b>: Optional reranking of top candidates using full-precision vectors
 *       for better accuracy. Controlled by {@code rerankFloor} parameter in search.</li>
 * </ol>
 *
 * <h2>Usage Examples</h2>
 *
 * <h3>Simple Exact Scoring</h3>
 * <pre>{@code
 * // Create a score provider for exact scoring
 * RandomAccessVectorValues vectors = ...;
 * BuildScoreProvider buildProvider = BuildScoreProvider.randomAccessScoreProvider(
 *     vectors,
 *     VectorSimilarityFunction.COSINE
 * );
 *
 * // Create a search provider for a query
 * VectorFloat<?> query = ...;
 * SearchScoreProvider searchProvider = buildProvider.searchProviderFor(query);
 *
 * // Get a score function and compute similarities
 * ScoreFunction scoreFunction = searchProvider.scoreFunction();
 * float score = scoreFunction.similarityTo(nodeId);
 * }</pre>
 *
 * <h3>Approximate Scoring with Reranking</h3>
 * <pre>{@code
 * // Create a PQ-based score provider
 * ProductQuantization pq = ProductQuantization.compute(vectors, 16, 256);
 * BuildScoreProvider buildProvider = BuildScoreProvider.pqBuildScoreProvider(
 *     vectors,
 *     VectorSimilarityFunction.COSINE,
 *     pq
 * );
 *
 * // During search, use approximate scores for traversal
 * SearchScoreProvider searchProvider = buildProvider.searchProviderFor(query);
 * ScoreFunction.ApproximateScoreFunction approx = searchProvider.scoreFunction();
 *
 * // Rerank top candidates with exact scores
 * ScoreFunction.ExactScoreFunction exact = searchProvider.exactScoreFunction();
 * for (int candidate : topCandidates) {
 *     float exactScore = exact.similarityTo(candidate);
 * }
 * }</pre>
 *
 * <h2>Performance Considerations</h2>
 * <ul>
 *   <li><b>Approximate scoring</b>: 5-10x faster than exact scoring with quantization, at the
 *       cost of some precision loss.</li>
 *   <li><b>Reranking</b>: Adding exact reranking typically improves recall by 1-5% with minimal
 *       performance impact when reranking only top-k candidates.</li>
 *   <li><b>Edge loading</b>: Bulk similarity computation can be 2-3x faster than individual
 *       queries when supported by the quantization method.</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <ul>
 *   <li>{@code BuildScoreProvider} implementations are typically thread-safe and can be shared.</li>
 *   <li>{@code SearchScoreProvider} instances are lightweight and can be created per query.</li>
 *   <li>{@code ScoreFunction} instances are typically <b>not thread-safe</b> and should be
 *       created per thread (or per search operation).</li>
 * </ul>
 *
 * @see io.github.jbellis.jvector.graph.similarity.BuildScoreProvider
 * @see io.github.jbellis.jvector.graph.similarity.SearchScoreProvider
 * @see io.github.jbellis.jvector.graph.similarity.ScoreFunction
 * @see io.github.jbellis.jvector.quantization
 */
package io.github.jbellis.jvector.graph.similarity;
