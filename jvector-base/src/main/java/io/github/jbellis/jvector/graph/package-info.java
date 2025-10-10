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
 * Provides core graph-based approximate nearest neighbor (ANN) search implementations.
 * <p>
 * This package contains the primary graph data structures and algorithms for building and
 * searching vector similarity indexes. JVector implements a hybrid approach combining
 * DiskANN-inspired graph construction with optional HNSW-style hierarchical layers.
 *
 * <h2>Core Concepts</h2>
 *
 * <h3>Graph Index</h3>
 * <p>
 * The graph index is a proximity graph where nodes represent vectors and edges connect
 * similar vectors. JVector uses a Vamana-based construction algorithm that builds a
 * high-quality base layer, with optional hierarchical layers for faster entry point selection.
 *
 * <h3>Key Interfaces and Classes</h3>
 *
 * <h4>Graph Representations</h4>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.ImmutableGraphIndex} - Immutable view of a graph
 *       index. All graph implementations provide this interface for thread-safe read access.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.MutableGraphIndex} - Mutable graph index interface
 *       that supports adding nodes and edges.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.OnHeapGraphIndex} - In-memory graph index
 *       implementation supporting concurrent construction and search.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex} - Memory-mapped graph
 *       index loaded from persistent storage (see {@link io.github.jbellis.jvector.graph.disk}
 *       package).</li>
 * </ul>
 *
 * <h4>Graph Construction</h4>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.GraphIndexBuilder} - Builder for constructing
 *       graph indexes. Supports concurrent graph construction with configurable parameters:
 *       <ul>
 *         <li>M - maximum edges per node (degree)</li>
 *         <li>beamWidth - search beam width during construction</li>
 *         <li>neighborOverflow - temporary overflow ratio during insertion</li>
 *         <li>alpha - diversity pruning parameter (controls edge length distribution)</li>
 *         <li>addHierarchy - whether to build HNSW-style hierarchical layers</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * <h4>Graph Search</h4>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.GraphSearcher} - Performs beam search on graph
 *       indexes to find approximate nearest neighbors. Supports:
 *       <ul>
 *         <li>Multi-layer hierarchical search</li>
 *         <li>Result reranking with exact distances</li>
 *         <li>Filtered search using {@link io.github.jbellis.jvector.util.Bits}</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.SearchResult} - Encapsulates search results with
 *       node IDs, scores, and search statistics.</li>
 * </ul>
 *
 * <h4>Vector Access</h4>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.RandomAccessVectorValues} - Interface for random
 *       access to vectors by ordinal. Supports both shared and unshared implementations.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.ListRandomAccessVectorValues} - In-memory vector
 *       storage backed by a List.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.MapRandomAccessVectorValues} - Vector storage
 *       backed by a Map, useful for sparse vector sets.</li>
 * </ul>
 *
 * <h4>Data Structures</h4>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.NodeArray} - Specialized array for storing node
 *       IDs and scores, supporting efficient sorted insertion.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.NodeQueue} - Priority queue for graph search.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.NodesIterator} - Iterator over node ordinals.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.ConcurrentNeighborMap} - Thread-safe neighbor
 *       storage for concurrent graph construction.</li>
 * </ul>
 *
 * <h2>Graph Construction Algorithm</h2>
 * <p>
 * JVector's graph construction is based on the DiskANN/Vamana algorithm with extensions:
 * <ol>
 *   <li>For each new node:
 *     <ul>
 *       <li>Assign a hierarchical level (if hierarchy enabled)</li>
 *       <li>Search for approximate nearest neighbors using beam search</li>
 *       <li>Connect to diverse neighbors using robust pruning</li>
 *       <li>Update existing nodes' neighbor lists (backlinks)</li>
 *     </ul>
 *   </li>
 *   <li>Concurrent insertions track in-progress nodes to maintain consistency</li>
 *   <li>After all insertions, cleanup phase:
 *     <ul>
 *       <li>Remove deleted nodes and update connections</li>
 *       <li>Optionally refine connections for improved recall</li>
 *       <li>Enforce degree constraints</li>
 *     </ul>
 *   </li>
 * </ol>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Build a graph index
 * RandomAccessVectorValues vectors = new ListRandomAccessVectorValues(vectorList, dimension);
 * GraphIndexBuilder builder = new GraphIndexBuilder(
 *     vectors,
 *     VectorSimilarityFunction.COSINE,
 *     16,    // M (max degree)
 *     100,   // beamWidth
 *     1.2f,  // neighborOverflow
 *     1.2f,  // alpha
 *     true   // addHierarchy
 * );
 * ImmutableGraphIndex graph = builder.build(vectors);
 *
 * // Search the graph
 * try (var view = graph.getView();
 *      var searcher = new GraphSearcher(graph)) {
 *     VectorFloat<?> query = ...;
 *     SearchScoreProvider ssp = BuildScoreProvider
 *         .randomAccessScoreProvider(vectors, VectorSimilarityFunction.COSINE)
 *         .searchProviderFor(query);
 *     SearchResult result = searcher.search(ssp, 10, Bits.ALL);
 *     for (SearchResult.NodeScore ns : result.getNodes()) {
 *         System.out.printf("Node %d: score %.4f%n", ns.node, ns.score);
 *     }
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <ul>
 *   <li>{@code ImmutableGraphIndex} and its views are thread-safe for concurrent reads</li>
 *   <li>{@code GraphIndexBuilder} supports concurrent insertions via {@code addGraphNode}</li>
 *   <li>{@code GraphSearcher} instances are stateful and not thread-safe; create one per thread</li>
 *   <li>{@code RandomAccessVectorValues} implementations may be shared or unshared; check
 *       {@code isValueShared()} and use {@code threadLocalSupplier()} for thread-safe access</li>
 * </ul>
 *
 * <h2>Related Packages</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk} - On-disk graph persistence</li>
 *   <li>{@link io.github.jbellis.jvector.graph.similarity} - Similarity scoring abstractions</li>
 *   <li>{@link io.github.jbellis.jvector.graph.diversity} - Diversity providers for neighbor selection</li>
 * </ul>
 *
 * @see io.github.jbellis.jvector.graph.GraphIndexBuilder
 * @see io.github.jbellis.jvector.graph.GraphSearcher
 * @see io.github.jbellis.jvector.graph.ImmutableGraphIndex
 * @see io.github.jbellis.jvector.graph.MutableGraphIndex
 */
package io.github.jbellis.jvector.graph;
