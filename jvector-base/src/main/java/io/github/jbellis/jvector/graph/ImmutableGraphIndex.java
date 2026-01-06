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

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorRepresentation;

import java.util.List;
import java.util.Objects;

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
     * Return a GraphSearcher with which to navigate the graph. A GraphSearcher is not threadsafe -- that is,
     * only one search at a time should be run per GraphSearcher.
     */
    GraphSearcher getSearcher();

    /**
     * @return the maximum number of edges per node across any layer
     */
    int maxDegree();

    List<Integer> maxDegrees();

    /**
     * @return the dimension of the vectors in the graph
     */
    int getDimension();

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

    /**
     * @return the vector representation corresponding to the specified ordinal
     */
    VectorRepresentation getVector(int node);

    @Override
    void close() throws IOException;

    /**
     * Returns true if this graph is hierarchical, false otherwise.
     * Note that a graph can be hierarchical even if it has a single layer, i.e., getMaxLevel() == 0.
     * For example, while building a new hierarchical graph, we may temporarily only have nodes at level 0
     * because of the random assignment of nodes to levels.
     */
    boolean isHierarchical();

    /**
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

    String prettyPrint();

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
