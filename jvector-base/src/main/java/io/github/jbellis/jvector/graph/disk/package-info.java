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
 * Provides classes for reading and writing graph indexes to persistent storage.
 * <p>
 * This package contains the core infrastructure for serializing and deserializing vector search
 * graph indexes. It supports both sequential and random-access writing strategies, multiple
 * format versions, and flexible feature storage (inline or separated).
 *
 * <h2>Key Components</h2>
 *
 * <h3>Writers</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter} - Abstract base class
 *       for all graph index writers, providing common functionality for header/footer writing,
 *       feature handling, and ordinal mapping</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter} - Random-access writer
 *       that can write nodes in any order</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.OnDiskSequentialGraphIndexWriter} - Sequential
 *       writer optimized for writing nodes in ordinal order</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.GraphIndexWriter} - Interface defining the
 *       contract for writing graph indexes</li>
 * </ul>
 *
 * <h3>Reader</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex} - Memory-mapped reader for
 *       accessing on-disk graph indexes efficiently</li>
 * </ul>
 *
 * <h3>Utilities</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.OrdinalMapper} - Maps between original graph
 *       ordinals and on-disk ordinals, useful for compacting deleted nodes or mapping to external IDs</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.Header} - Encapsulates the index header format</li>
 *   <li>{@link io.github.jbellis.jvector.graph.disk.CommonHeader} - Common header information
 *       shared across format versions</li>
 * </ul>
 *
 * <h2>On-Disk Format</h2>
 * <p>
 * The on-disk format consists of the following sections:
 * <ol>
 *   <li><b>Header</b> - Contains metadata about the graph (version, dimension, entry node, layer info)
 *       and feature configuration</li>
 *   <li><b>Dense Level (Level 0)</b> - All graph nodes with their inline features and neighbor lists</li>
 *   <li><b>Sparse Levels</b> - For hierarchical graphs, additional levels containing only nodes
 *       participating in those levels</li>
 *   <li><b>Separated Features</b> - Optional section containing feature data that is stored
 *       separately from nodes for better cache locality</li>
 *   <li><b>Footer</b> - Contains the header offset (allowing the header to be located) and a
 *       magic number for file validation</li>
 * </ol>
 *
 * <h2>Format Versions</h2>
 * <ul>
 *   <li><b>Version 1-2</b>: Basic format with inline vectors only</li>
 *   <li><b>Version 3</b>: Support for multiple feature types</li>
 *   <li><b>Version 4+</b>: Support for multilayer (hierarchical) graphs</li>
 * </ul>
 *
 * <h2>Features</h2>
 * <p>
 * Features represent additional data stored with graph nodes (e.g., vectors, compressed vectors).
 * Features can be stored inline (with each node) or separated (in a dedicated section).
 * See the {@link io.github.jbellis.jvector.graph.disk.feature} package for available feature types.
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Writing a graph index
 * try (var output = new BufferedRandomAccessWriter(...)) {
 *     var writer = new OnDiskGraphIndexWriter.Builder(graph, output)
 *         .withVersion(4)
 *         .with(new InlineVectors(dimension))
 *         .build();
 *
 *     writer.write(featureSuppliers);
 * }
 *
 * // Reading a graph index
 * var reader = OnDiskGraphIndex.load(...);
 * try (var view = reader.getView()) {
 *     // Access nodes and neighbors
 *     var neighbors = view.getNeighborsIterator(level, ordinal);
 *     // Read features
 *     var vector = view.getVector(ordinal);
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>
 * Writers are not thread-safe for concurrent writes to the same instance.
 * Readers ({@code OnDiskGraphIndex}) are thread-safe and support concurrent read access
 * through separate views.
 *
 * @see io.github.jbellis.jvector.graph.disk.feature
 * @see io.github.jbellis.jvector.graph.ImmutableGraphIndex
 */
package io.github.jbellis.jvector.graph.disk;
