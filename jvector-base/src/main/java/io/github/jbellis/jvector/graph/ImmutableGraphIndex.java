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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;
import java.util.Objects;

import java.io.Closeable;
import java.io.IOException;

/**
 * Represents a graph-based vector index.  Nodes are represented as ints, and edges are
 * represented as adjacency lists.
 * <p>
 * Mostly this applies to any graph index, but a few methods (e.g. getVector()) are
 * specifically included to support the DiskANN-based design of OnDiskGraphIndex.
 * <p>
 * All methods are threadsafe.  Operations that require persistent state are wrapped
 * in a View that should be created per accessing thread.
 */
public interface ImmutableGraphIndex extends AutoCloseable, Accountable {
    /**
     * Returns the number of nodes in the graph.
     *
     * @return the number of nodes in the graph
     * @deprecated Use {@link #size(int)} with level 0 instead
     */
    @Deprecated
    default int size() {
        return size(0);
    }

    /**
     * Get all node ordinals included in the graph. The nodes are NOT guaranteed to be
     * presented in any particular order.
     *
     * @param level the level of the graph to get nodes from
     * @return an iterator over nodes where {@code nextInt} returns the next node.
     */
    NodesIterator getNodes(int level);

    /**
     * Return a View with which to navigate the graph.  Views are not threadsafe -- that is,
     * only one search at a time should be run per View.
     * <p>
     * Additionally, the View represents a point of consistency in the graph, and in-use
     * Views prevent the removal of marked-deleted nodes from graphs that are being
     * concurrently modified.  Thus, it is good (and encouraged) to re-use Views for
     * on-disk, read-only graphs, but for in-memory graphs, it is better to create a new
     * View per search.
     *
     * @return a View for navigating the graph
     */
    View getView();

    /**
     * Returns the maximum number of edges per node across any layer.
     *
     * @return the maximum degree
     */
    int maxDegree();

    /**
     * Returns a list of maximum degrees for each layer of the graph.
     * If fewer entries are specified than the number of layers, the last entry applies to all remaining layers.
     *
     * @return a list of maximum degrees per layer
     */
    List<Integer> maxDegrees();

    /**
     * Returns the first ordinal greater than all node ids in the graph.  Equal to size() in simple cases.
     * May be different from size() if nodes are being added concurrently, or if nodes have been
     * deleted (and cleaned up).
     *
     * @return the first ordinal greater than all node ids in the graph.  Equal to size() in simple cases;
     * May be different from size() if nodes are being added concurrently, or if nodes have been
     * deleted (and cleaned up).
     */
    default int getIdUpperBound() {
        return size();
    }

    /**
     * Returns true if and only if the graph contains the node with the given ordinal id.
     *
     * @param nodeId the node identifier to check
     * @return true iff the graph contains the node with the given ordinal id
     */
    default boolean containsNode(int nodeId) {
        return nodeId >= 0 && nodeId < size();
    }

    /**
     * Closes this graph index and releases any resources.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    void close() throws IOException;

    /**
     * Returns the maximum (coarser) level that contains a vector in the graph.
     *
     * @return The maximum (coarser) level that contains a vector in the graph.
     */
    int getMaxLevel();

    /**
     * Return the maximum out-degree allowed of the given level.
     * @param level The level of interest
     * @return the maximum out-degree of the given level
     */
    int getDegree(int level);

    /**
     * Returns the average degree computed over nodes in the specified layer.
     *
     * @param level the level of interest.
     * @return the average degree or NaN if no nodes are present.
     */
    double getAverageDegree(int level);

    /**
     * Return the number of vectors/nodes in the given level.
     * @param level The level of interest
     * @return the number of vectors in the given level
     */
    int size(int level);

    /**
     * Encapsulates the state of a graph for searching.  Re-usable across search calls,
     * but each thread needs its own.
     */
    interface View extends Closeable {
        /**
         * Iterator over the neighbors of a given node.  Only the most recently instantiated iterator
         * is guaranteed to be valid.
         *
         * @param level the level of the graph
         * @param node the node whose neighbors to iterate
         * @return an iterator over the neighbors of the node
         */
        NodesIterator getNeighborsIterator(int level, int node);

        /**
         * This method is deprecated as most View usages should not need size.
         * Where they do, they could access the graph.
         *
         * @return the number of nodes in the graph
         * @deprecated Use the graph's size() method instead
         */
        @Deprecated
        int size();

        /**
         * Returns the node of the graph to start searches at.
         *
         * @return the node of the graph to start searches at
         */
        NodeAtLevel entryNode();

        /**
         * Return a Bits instance indicating which nodes are live.  The result is undefined for
         * ordinals that do not correspond to nodes in the graph.
         *
         * @return a Bits instance indicating which nodes are live
         */
        Bits liveNodes();

        /**
         * Returns the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
         *
         * @return the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
         */
        default int getIdUpperBound() {
            return size();
        }

        /**
         * Whether the given node is present in the given layer of the graph.
         *
         * @param level the level to check
         * @param node the node to check
         * @return true if the node is present in the layer, false otherwise
         */
        boolean contains(int level, int node);
    }

    /**
     * A View that knows how to compute scores against a query vector.  (This is all Views
     * except for OnHeapGraphIndex.ConcurrentGraphIndexView.)
     */
    interface ScoringView extends View {
        /**
         * Returns an exact score function for reranking results.
         *
         * @param queryVector the query vector to compute scores against
         * @param vsf the vector similarity function to use
         * @return an exact score function
         */
        ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf);

        /**
         * Returns an approximate score function for initial candidate scoring.
         *
         * @param queryVector the query vector to compute scores against
         * @param vsf the vector similarity function to use
         * @return an approximate score function
         */
        ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf);
    }

    /**
     * Returns a human-readable string representation of the graph structure showing all nodes and their neighbors.
     *
     * @param graph the graph index to format
     * @return a formatted string representation of the graph
     */
    static String prettyPrint(ImmutableGraphIndex graph) {
        StringBuilder sb = new StringBuilder();
        sb.append(graph);
        sb.append("\n");

        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                sb.append(String.format("# Level %d\n", level));
                NodesIterator it = graph.getNodes(level);
                while (it.hasNext()) {
                    int node = it.nextInt();
                    sb.append("  ").append(node).append(" -> ");
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

    /**
     * Represents a node at a specific level in the hierarchical graph structure.
     * Comparable to support use in ConcurrentSkipListMap.
     */
    final class NodeAtLevel implements Comparable<NodeAtLevel> {
        /** The level in the hierarchy where this node exists */
        public final int level;
        /** The node identifier */
        public final int node;

        /**
         * Creates a new NodeAtLevel instance.
         *
         * @param level the level in the hierarchy (must be non-negative)
         * @param node the node identifier (must be non-negative)
         */
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
}
