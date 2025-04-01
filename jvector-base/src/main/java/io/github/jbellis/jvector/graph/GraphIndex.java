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
public interface GraphIndex extends AutoCloseable, Accountable {
    /** Returns the number of nodes in the graph */
    @Deprecated
    default int size() {
        return size(0);
    }

    /**
     * Get all node ordinals included in the graph. The nodes are NOT guaranteed to be
     * presented in any particular order.
     *
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
     */
    View getView();

    /**
     * @return the maximum number of edges per node across any layer
     */
    int maxDegree();

    /**
     * @return the first ordinal greater than all node ids in the graph.  Equal to size() in simple cases;
     * May be different from size() if nodes are being added concurrently, or if nodes have been
     * deleted (and cleaned up).
     */
    default int getIdUpperBound() {
        return size();
    }

    /**
     * @return true iff the graph contains the node with the given ordinal id
     */
    default boolean containsNode(int nodeId) {
        return nodeId >= 0 && nodeId < size();
    }

    @Override
    void close() throws IOException;

    /**
     * @return The maximum (coarser) level with that contains a vector in the graph.
     */
    int getMaxLevel();

    /**
     * Return the maximum out-degree allowed of the given level.
     * @param level The level of interest
     * @return the maximum out-degree of the given level
     */
    int getDegree(int level);

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
         */
        NodesIterator getNeighborsIterator(int level, int node);

        /**
         * This method is deprecated as most View usages should not need size.
         * Where they do, they could access the graph.
         * @return the number of nodes in the graph
         */
        @Deprecated
        int size();

        /**
         * @return the node of the graph to start searches at
         */
        NodeAtLevel entryNode();

        /**
         * Return a Bits instance indicating which nodes are live.  The result is undefined for
         * ordinals that do not correspond to nodes in the graph.
         */
        Bits liveNodes();

        /**
         * @return the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
         */
        default int getIdUpperBound() {
            return size();
        }
    }

    /**
     * A View that knows how to compute scores against a query vector.  (This is all Views
     * except for OnHeapGraphIndex.ConcurrentGraphIndexView.)
     */
    interface ScoringView extends View {
        ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf);
        ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf);
    }

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

    // Comparable b/c it gets used in ConcurrentSkipListMap
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
}
