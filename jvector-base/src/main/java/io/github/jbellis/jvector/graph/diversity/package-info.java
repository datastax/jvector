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
 * Provides neighbor diversity selection strategies for graph construction.
 * <p>
 * This package contains implementations of diversity providers that determine which neighbors
 * to retain during graph construction. Diversity selection is critical for building high-quality
 * proximity graphs that balance local connectivity with long-range edges.
 *
 * <h2>Diversity and Graph Quality</h2>
 * <p>
 * In graph-based vector search, simply connecting each node to its k nearest neighbors can lead
 * to poor search performance. Diversity selection addresses two key issues:
 * <ul>
 *   <li><b>Clustering</b>: Without diversity, nodes in dense regions may connect only to their
 *       immediate cluster, making it difficult to reach distant regions of the vector space.</li>
 *   <li><b>Graph traversability</b>: A diverse neighbor set includes both short edges (for local
 *       precision) and longer edges (for efficient navigation across the space).</li>
 * </ul>
 *
 * <h2>Core Abstractions</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.diversity.DiversityProvider} - Interface for
 *       diversity selection algorithms. Implementations select which neighbors to retain from
 *       a candidate set while maintaining graph quality constraints.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider} - Implementation
 *       based on the DiskANN/Vamana Robust Prune algorithm. Uses an alpha parameter to control
 *       the trade-off between short edges (high recall) and longer edges (better graph
 *       navigability).</li>
 * </ul>
 *
 * <h2>Vamana Diversity Algorithm</h2>
 * <p>
 * The Vamana diversity provider implements the Robust Prune algorithm from the DiskANN paper:
 * <ol>
 *   <li>Start with a candidate set of potential neighbors</li>
 *   <li>For each candidate, check if adding it would create a "shortcut" - that is, if there's
 *       already a neighbor that's closer to both the target node and the candidate</li>
 *   <li>The alpha parameter controls how strict this test is:
 *     <ul>
 *       <li>alpha = 1.0: Only keep edges where no existing neighbor is closer (strictest)</li>
 *       <li>alpha &gt; 1.0: Allow longer edges even when shortcuts exist (recommended)</li>
 *       <li>Higher alpha values create more diverse graphs with better long-range connectivity</li>
 *     </ul>
 *   </li>
 *   <li>Select up to maxDegree diverse neighbors</li>
 * </ol>
 *
 * <h2>Usage in Graph Construction</h2>
 * <p>
 * Diversity providers are used by {@link io.github.jbellis.jvector.graph.GraphIndexBuilder}
 * during graph construction:
 * <pre>{@code
 * // Create a diversity provider with alpha=1.2 for balanced diversity
 * BuildScoreProvider scoreProvider = ...;
 * DiversityProvider diversityProvider = new VamanaDiversityProvider(scoreProvider, 1.2f);
 *
 * // The GraphIndexBuilder uses the diversity provider internally
 * GraphIndexBuilder builder = new GraphIndexBuilder(
 *     scoreProvider,
 *     dimension,
 *     maxDegree,
 *     beamWidth,
 *     neighborOverflow,
 *     1.2f,  // alpha passed to diversity provider
 *     addHierarchy
 * );
 * }</pre>
 *
 * <h2>Alpha Parameter Guidelines</h2>
 * <ul>
 *   <li><b>alpha = 1.0</b>: Creates an HNSW-like graph at the base layer (not recommended for
 *       JVector's Vamana-based approach)</li>
 *   <li><b>alpha = 1.2</b> (default): Good balance between recall and build efficiency</li>
 *   <li><b>alpha &gt; 1.5</b>: More diverse graphs with better long-range connectivity but
 *       potentially lower recall for small beam widths</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>
 * {@code DiversityProvider} implementations are typically stateless (beyond immutable
 * configuration) and thread-safe. The same instance can be shared across multiple threads
 * during concurrent graph construction.
 *
 * @see io.github.jbellis.jvector.graph.diversity.DiversityProvider
 * @see io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider
 * @see io.github.jbellis.jvector.graph.GraphIndexBuilder
 */
package io.github.jbellis.jvector.graph.diversity;
