/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * Root interface for all graph-based vector indexes, both mutable and immutable.
 * <p>
 * Nodes are represented as ints; edges are represented as adjacency lists.
 * All read methods are threadsafe. Operations that require persistent state are
 * wrapped in a {@link View} that should be created per accessing thread.
 * <p>
 * Subtypes:
 * <ul>
 *   <li>{@link ImmutableGraphIndex} – a sealed, fully-built graph that can be persisted.</li>
 *   <li>{@link MutableGraphIndex} – a live graph that accepts concurrent modifications.</li>
 * </ul>
 */
public interface GraphIndex extends AutoCloseable, Accountable {

    /** @deprecated use {@link #size(int)} with level 0 */
    @Deprecated
    default int size() {
        return size(0);
    }

    /**
     * Returns an iterator over all node ordinals in the given level.
     * The nodes are NOT guaranteed to be in any particular order.
     */
    NodesIterator getNodes(int level);

    /**
     * Returns a View for navigating the graph.  Views are not threadsafe –
     * only one search at a time should run per View.
     * <p>
     * A View represents a point-in-time snapshot of the graph; in-use Views prevent
     * removal of marked-deleted nodes from graphs being concurrently modified. It is
     * good practice to re-use Views for read-only (on-disk) graphs, but for in-memory
     * graphs a new View per search is preferable.
     */
    View getView();

    /** @return the maximum number of edges per node across any layer */
    int maxDegree();

    List<Integer> maxDegrees();

    /** @return the dimension of the vectors stored in the graph */
    int getDimension();

    /**
     * @return the first ordinal greater than all node ids in the graph. Equal to
     * {@link #size(int)} in simple cases; may differ when nodes are added concurrently
     * or after deletion cleanup.
     */
    default int getIdUpperBound() {
        return size();
    }

    /** @return true iff the graph contains the node with the given ordinal id */
    default boolean containsNode(int nodeId) {
        return nodeId >= 0 && nodeId < size();
    }

    @Override
    void close() throws IOException;

    /**
     * Returns true if this graph is hierarchical (has multiple layers), false otherwise.
     * Note that a graph can be hierarchical even with a single layer, e.g. while a new
     * hierarchical graph is being built.
     */
    boolean isHierarchical();

    /** @return the maximum (coarsest) level that contains a node in the graph */
    int getMaxLevel();

    /** @return the maximum out-degree allowed at the given level */
    int getDegree(int level);

    /**
     * @return the average degree computed over nodes at the given level,
     * or NaN if no nodes are present
     */
    double getAverageDegree(int level);

    /** @return the number of nodes/vectors in the given level */
    int size(int level);

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /**
     * Returns a {@link WriteBuilder} configured to write this graph to {@code path}
     * using the default writer strategy for this index type.
     * <p>
     * Call configuration methods on the returned builder before invoking
     * {@link WriteBuilder#write}.
     */
    default WriteBuilder writer(Path path) throws FileNotFoundException {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support path-based writing");
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /**
     * Returns a new {@link GraphSearcher} backed by this graph.
     * <p>
     * The returned searcher is <em>not</em> thread-safe; use one per thread.
     * The caller is responsible for closing the searcher when done.
     */
    default GraphSearcher searcher() {
        return new GraphSearcher(this);
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    static String prettyPrint(GraphIndex graph) {
        StringBuilder sb = new StringBuilder();
        sb.append(graph);
        sb.append("\n");
        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                sb.append(String.format("# Level %d\n", level));
                NodesIterator it = graph.getNodes(level);
                while (it.hasNext()) {
                    int node = it.nextInt();
                    sb.append("  ").append(node).append(" ->");
                    for (var neighbors = view.getNeighborsIterator(level, node); neighbors.hasNext(); ) {
                        sb.append(" ").append(neighbors.nextInt());
                    }
                    sb.append("\n");
                }
                sb.append("\n");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return sb.toString();
    }

    // -----------------------------------------------------------------------
    // Nested types
    // -----------------------------------------------------------------------

    /** Callback for processing a neighbor during a search. */
    interface NeighborProcessor {
        void process(int friendOrd, float similarity);
    }

    /** Marks node ordinals as visited during a search. */
    @FunctionalInterface
    interface IntMarker {
        /** Marks the node and returns true if it was not previously marked. */
        boolean mark(int value);
    }

    /**
     * A thread-local snapshot of the graph for search traversal.
     * Re-usable across search calls on the same thread; not threadsafe.
     */
    interface View extends Closeable {
        /** Returns an iterator over the neighbors of {@code node} at {@code level}. */
        NodesIterator getNeighborsIterator(int level, int node);

        /**
         * Iterates over the unvisited neighbors of {@code node}, computes their similarity,
         * and invokes {@code neighborProcessor} for each.
         */
        void processNeighbors(int level, int node, ScoreFunction scoreFunction,
                              IntMarker visited, NeighborProcessor neighborProcessor);

        /** @deprecated most View usages do not need size; access via the graph instead */
        @Deprecated
        int size();

        /** @return the entry node for graph traversal */
        NodeAtLevel entryNode();

        /**
         * Returns a Bits instance indicating which nodes are live. The result is
         * undefined for ordinals that do not correspond to nodes in the graph.
         */
        Bits liveNodes();

        default int getIdUpperBound() {
            return size();
        }

        /** @return true iff {@code node} is present in the given layer */
        boolean contains(int level, int node);
    }

    /**
     * A View that can compute scores against a query vector. This applies to all
     * Views except {@code OnHeapGraphIndex.ConcurrentGraphIndexView}.
     */
    interface ScoringView extends View {
        ScoreFunction.ExactScoreFunction refinerFor(VectorFloat<?> queryVector,
                                                    VectorSimilarityFunction vsf);
        ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector,
                                                                           VectorSimilarityFunction vsf);
    }

    /** Identifies a node at a specific level in the graph hierarchy. */
    final class NodeAtLevel implements Comparable<NodeAtLevel> {
        public final int level;
        public final int node;

        public NodeAtLevel(int level, int node) {
            assert level >= 0 : level;
            assert node >= 0 : node;
            this.level = level;
            this.node = node;
        }

        @Override
        public int compareTo(NodeAtLevel o) {
            int cmp = Integer.compare(level, o.level);
            if (cmp == 0) {
                cmp = Integer.compare(node, o.node);
            }
            return cmp;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof NodeAtLevel)) return false;
            NodeAtLevel that = (NodeAtLevel) o;
            return level == that.level && node == that.node;
        }

        @Override
        public int hashCode() {
            return Objects.hash(level, node);
        }

        @Override
        public String toString() {
            return "NodeAtLevel(level=" + level + ", node=" + node + ")";
        }
    }

    /**
     * Fluent builder for persisting a {@link GraphIndex} to disk.
     * <p>
     * Obtain an instance via {@link GraphIndex#writer(Path)} or
     * {@link ImmutableGraphIndex#writer(IndexWriter)} (for sequential writes).
     * Configuration methods must be called before the first call to
     * {@link #writeFeaturesInline} or {@link #write}.
     */
    interface WriteBuilder extends Closeable {
        /** Adds a feature to be written with this graph. */
        WriteBuilder with(Feature feature);

        /** Sets the ordinal mapper used to renumber node ids on write. */
        WriteBuilder withMapper(OrdinalMapper mapper);

        /** Convenience for {@link #withMapper} using a pre-computed old-to-new mapping. */
        WriteBuilder withMap(Map<Integer, Integer> oldToNew);

        /** Sets the on-disk format version (defaults to the current version). */
        WriteBuilder withVersion(int version);

        /**
         * Sets the byte offset at which writing begins in the output file.
         * Useful when appending a graph to an existing file.
         */
        WriteBuilder withStartOffset(long offset);

        /**
         * Sets the number of worker threads for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         *
         * @param n number of threads; negative means use all available processors; 0 (default) disables parallel writes
         */
        WriteBuilder withParallelWorkerThreads(int n);

        /**
         * Whether to use direct ByteBuffers for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         */
        WriteBuilder withParallelDirectBuffers(boolean useDirectBuffers);

        /**
         * Writes the inline features for a single node ordinal without writing graph
         * structure. Used for incremental (node-at-a-time) construction patterns.
         * Must be called after all configuration methods and before {@link #write}.
         */
        WriteBuilder writeFeaturesInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException;

        /**
         * Writes the complete graph header, edge lists, and any remaining features
         * to the configured output. Closes the underlying output stream when done.
         *
         * @param featureStateSuppliers per-node suppliers for each configured feature;
         *                              features already written via {@link #writeFeaturesInline}
         *                              can be omitted from this map
         */
        void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException;

        /** Writes the graph header/metadata at the current position. */
        void writeHeader(GraphIndex.View view) throws IOException;

        /** @return the CRC32 checksum of all bytes written since the start offset */
        long checksum() throws IOException;

        @Override
        void close() throws IOException;
    }
}
