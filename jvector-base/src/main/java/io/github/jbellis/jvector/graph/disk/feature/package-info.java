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
 * Provides feature types for storing additional data with on-disk graph indexes.
 * <p>
 * This package contains implementations of features that can be stored alongside graph nodes
 * in persistent indexes. Features represent additional per-node data such as vectors, compressed
 * vectors, or other metadata. The feature system supports both inline storage (data stored with
 * each node) and separated storage (data stored in a dedicated section for better cache locality).
 *
 * <h2>Feature Architecture</h2>
 * <p>
 * Features are identified by {@link io.github.jbellis.jvector.graph.disk.feature.FeatureId}
 * and implement the {@link io.github.jbellis.jvector.graph.disk.feature.Feature} interface.
 * During graph writing, features are serialized to the index file. During reading, features
 * are loaded from the header and provide access to per-node data.
 *
 * <h2>Core Abstractions</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.Feature} - Base interface for all
 *       features. Defines methods for:
 *       <ul>
 *         <li>Writing header metadata (dimensions, compression parameters, etc.)</li>
 *         <li>Writing per-node inline data</li>
 *         <li>Querying feature size and storage layout</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.FeatureId} - Enum identifying
 *       available feature types. New features should be added to the end to maintain
 *       serialization compatibility.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.FeatureSource} - Marker interface
 *       for features that provide data during graph writing.</li>
 * </ul>
 *
 * <h2>Available Features</h2>
 *
 * <h3>Vector Storage</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.InlineVectors} - Stores full-precision
 *       vectors inline with each graph node. Best for:
 *       <ul>
 *         <li>Small to medium dimensional vectors (&lt; 512 dimensions)</li>
 *         <li>When exact similarity computation is always required</li>
 *       </ul>
 *   </li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors} - Stores vectors
 *       in a dedicated section separate from the graph structure. Benefits:
 *       <ul>
 *         <li>Better cache locality during graph traversal (when vectors aren't needed)</li>
 *         <li>More efficient when using approximate scoring during search</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * <h3>Compressed Vector Storage</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.NVQ} - Stores vectors compressed
 *       using Neighborhood Vector Quantization (NVQ). Inline storage variant.</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ} - Separated storage
 *       variant of NVQ compression. Recommended for most use cases combining compression with
 *       cache-friendly layout.</li>
 * </ul>
 *
 * <h3>Specialized Features</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.feature.FusedADC} - Combines compressed
 *       vectors with precomputed query-dependent data for faster similarity computation. Used
 *       with Product Quantization (PQ) for asymmetric distance computation.</li>
 * </ul>
 *
 * <h2>Storage Strategies</h2>
 *
 * <h3>Inline Storage</h3>
 * <p>
 * Inline features store data directly with each graph node. This provides:
 * <ul>
 *   <li><b>Advantages</b>: Single random access to get both graph structure and feature data</li>
 *   <li><b>Disadvantages</b>: Larger per-node size reduces cache efficiency during graph traversal</li>
 * </ul>
 *
 * <h3>Separated Storage</h3>
 * <p>
 * Separated features ({@link io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature})
 * store data in a dedicated section. This provides:
 * <ul>
 *   <li><b>Advantages</b>:
 *     <ul>
 *       <li>Smaller per-node size improves cache utilization during traversal</li>
 *       <li>Feature data only accessed when needed (e.g., for reranking)</li>
 *       <li>Better suited for approximate + exact scoring workflows</li>
 *     </ul>
 *   </li>
 *   <li><b>Disadvantages</b>: Requires additional seek for feature access</li>
 * </ul>
 *
 * <h2>Feature Selection Guidelines</h2>
 * <table border="1">
 *   <caption>Feature Selection by Use Case</caption>
 *   <tr>
 *     <th>Use Case</th>
 *     <th>Recommended Feature</th>
 *     <th>Rationale</th>
 *   </tr>
 *   <tr>
 *     <td>High-dimensional vectors (&gt; 512d)</td>
 *     <td>SeparatedVectors or SeparatedNVQ</td>
 *     <td>Reduces per-node size for better cache efficiency</td>
 *   </tr>
 *   <tr>
 *     <td>Memory-constrained environments</td>
 *     <td>SeparatedNVQ or FusedADC</td>
 *     <td>Compression reduces memory footprint</td>
 *   </tr>
 *   <tr>
 *     <td>Low-dimensional vectors (&lt; 128d)</td>
 *     <td>InlineVectors</td>
 *     <td>Minimal overhead, single access pattern</td>
 *   </tr>
 *   <tr>
 *     <td>Approximate + exact reranking</td>
 *     <td>SeparatedNVQ + SeparatedVectors</td>
 *     <td>Use compressed for search, exact for reranking</td>
 *   </tr>
 * </table>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Writing a graph with separated vectors
 * try (var writer = new OnDiskGraphIndexWriter.Builder(graph, output)
 *         .with(new SeparatedVectors(dimension))
 *         .build()) {
 *     // Create feature state for each node
 *     var features = Feature.singleStateFactory(
 *         FeatureId.SEPARATED_VECTORS,
 *         nodeId -> new SeparatedVectors.State(vectors.getVector(nodeId))
 *     );
 *     writer.write(features);
 * }
 *
 * // Reading and accessing feature data
 * var reader = OnDiskGraphIndex.load(...);
 * try (var view = reader.getView()) {
 *     VectorFloat<?> vector = view.getVector(nodeId);
 * }
 * }</pre>
 *
 * <h2>Adding New Features</h2>
 * <p>
 * To add a new feature type:
 * <ol>
 *   <li>Add a new entry to {@code FeatureId} enum (at the end to maintain compatibility)</li>
 *   <li>Implement the {@code Feature} interface with:
 *     <ul>
 *       <li>Header serialization (metadata like dimensions, compression parameters)</li>
 *       <li>Per-node data serialization (inline or separated)</li>
 *       <li>Loading logic to reconstruct from disk</li>
 *     </ul>
 *   </li>
 *   <li>Update graph writers to support the new feature</li>
 *   <li>Update graph readers to provide access to the feature data</li>
 * </ol>
 *
 * <h2>Thread Safety</h2>
 * <ul>
 *   <li>Feature instances are typically immutable after construction and thread-safe</li>
 *   <li>Feature.State instances are per-write operation and not thread-safe</li>
 *   <li>Feature data access through graph views is thread-safe</li>
 * </ul>
 *
 * @see io.github.jbellis.jvector.graph.disk.feature.Feature
 * @see io.github.jbellis.jvector.graph.disk.feature.FeatureId
 * @see io.github.jbellis.jvector.graph.disk
 */
package io.github.jbellis.jvector.graph.disk.feature;
