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

import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;

import java.util.List;
import java.util.stream.IntStream;

/**
 * A {@link GraphIndex} that accepts concurrent modifications during construction.
 * <p>
 * Typical graphs will see significant throughput improvements as additional threads
 * are used for construction and searching.
 * <p>
 * The base layer (layer 0) contains all nodes; higher layers are stored in sparse maps.
 * For searching, obtain a View via {@link #getView()} which supports level-aware operations.
 */
public interface MutableGraphIndex extends GraphIndex {

    /**
     * Add the given node ordinal with an empty set of neighbors.
     * <p>
     * Nodes may be inserted out of order, but all ordinals below the inserted node
     * must eventually be added. Populating neighbors and establishing bidirectional
     * links is the caller's responsibility, as is ensuring each node is added only once.
     */
    void addNode(GraphIndex.NodeAtLevel nodeLevel);

    /** @see #addNode(GraphIndex.NodeAtLevel) */
    void addNode(int level, int node);

    /** @return true iff the given node is present in the graph */
    boolean contains(GraphIndex.NodeAtLevel nodeLevel);

    /** @return true iff the given node is present at the given layer */
    boolean contains(int level, int node);

    /**
     * Connect {@code node} at {@code level} to the given neighbor set.
     * Use with extreme caution.
     */
    void connectNode(int level, int node, NodeArray nodes);

    /** @see #connectNode(int, int, NodeArray) */
    void connectNode(GraphIndex.NodeAtLevel nodeLevel, NodeArray nodes);

    /** Marks the given node as deleted. Does NOT remove it from the graph immediately. */
    void markDeleted(int node);

    /** Must be called after {@link #addNode} once all neighbors are linked at all levels. */
    void markComplete(GraphIndex.NodeAtLevel nodeLevel);

    void updateEntryNode(GraphIndex.NodeAtLevel newEntry);

    /** @return an upper bound on the RAM used by a single node, in bytes */
    long ramBytesUsedOneNode(int layer);

    ThreadSafeGrowableBitSet getDeletedNodes();

    void setDegrees(List<Integer> layerDegrees);

    /** Enforce the degree constraint on the given node across all layers. */
    void enforceDegree(int node);

    /** Returns an iterator over the neighbors of the given node at the given level. */
    NodesIterator getNeighborsIterator(GraphIndex.NodeAtLevel nodeLevel);

    /** @see #getNeighborsIterator(GraphIndex.NodeAtLevel) */
    NodesIterator getNeighborsIterator(int level, int node);

    /**
     * Removes the given node from all layers.
     *
     * @return the number of layers from which it was removed
     */
    int removeNode(int node);

    /** @return an IntStream of node ids contained in the given level */
    IntStream nodeStream(int level);

    /**
     * @return the maximum level containing this node, or -1 if the node is not in the graph
     */
    int getMaxLevelForNode(int node);

    /** @return the current entry node for graph traversal */
    GraphIndex.NodeAtLevel entryNode();

    /**
     * Adds the given neighbors to {@code node} at {@code level}, maintaining diversity.
     * Also adds backlinks from the neighbors to {@code node}.
     * Edges are only added while the out-degree is below {@code overflowRatio} times max degree.
     */
    void addEdges(int level, int node, NodeArray candidates, float overflowRatio);

    /**
     * Removes edges to deleted nodes from {@code node} at {@code level} and
     * replaces them with new connections, maintaining diversity.
     */
    void replaceDeletedNeighbors(int level, int node, BitSet toDelete, NodeArray candidates);

    /**
     * Signals that all mutations are complete and the graph will not be mutated further.
     * Should be called by the builder after cleanup.
     */
    void setAllMutationsCompleted();

    /** @return true if {@link #setAllMutationsCompleted()} has been called */
    boolean allMutationsCompleted();
}
