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
 * An {@link ImmutableGraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>The base layer (layer 0) contains all nodes, while higher layers are stored in sparse maps.
 * For searching, use a view obtained from {@link #getView()} which supports level–aware operations.
 */
interface MutableGraphIndex extends ImmutableGraphIndex {
    /**
     * Add the given node ordinal with an empty set of neighbors.
     *
     * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
     * inserted out of order are eventually added.
     *
     * <p>Actually populating the neighbors, and establishing bidirectional links, is the
     * responsibility of the caller.
     *
     * <p>It is also the responsibility of the caller to ensure that each node is only added once.
     */
    void addNode(NodeAtLevel nodeLevel);

    /**
     * Add the given node ordinal with an empty set of neighbors.
     *
     * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
     * inserted out of order are eventually added.
     *
     * <p>Actually populating the neighbors, and establishing bidirectional links, is the
     * responsibility of the caller.
     *
     * <p>It is also the responsibility of the caller to ensure that each node is only added once.
     */
    void addNode(int level, int node);

    /**
     * Whether the given node is present in the graph.
     */
    boolean contains(NodeAtLevel nodeLevel);

    /**
     * Whether the given node is present in the given layer of the graph.
     */
    boolean contains(int level, int node);

    /**
     * Add the given node ordinal with an empty set of neighbors.
     *
     * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
     * inserted out of order are eventually added.
     *
     * <p>Actually populating the neighbors, and establishing bidirectional links, is the
     * responsibility of the caller.
     *
     * <p>It is also the responsibility of the caller to ensure that each node is only added once.
     */
    void connectNode(int level, int node, NodeArray nodes);

    /**
     * Use with extreme caution. Used by Builder to load a saved graph and for rescoring.
     */
    void connectNode(NodeAtLevel nodeLevel, NodeArray nodes);

    /**
     * Mark the given node deleted.  Does NOT remove the node from the graph.
     */
    void markDeleted(int node);

    /** must be called after addNode once neighbors are linked in all levels. */
    void markComplete(NodeAtLevel nodeLevel);

    void updateEntryNode(NodeAtLevel newEntry);

    /**
     * Returns an upper bound on the amount of memory used by a single node, in bytes.
     */
    long ramBytesUsedOneNode(int layer);

    ThreadSafeGrowableBitSet getDeletedNodes();

    void setDegrees(List<Integer> layerDegrees);

    /**
     * Enforce the degree of the given node in all layers.
     */
    void enforceDegree(int node);

    /**
     * Returns an iterator over the neighbors for the given node at the specified level, which can be empty if the node does not belong to that layer.
     */
    NodesIterator getNeighborsIterator(NodeAtLevel nodeLevel);

    /**
     * Returns an iterator over the neighbors for the given node at the specified level, which can be empty if the node does not belong to that layer.
     */
    NodesIterator getNeighborsIterator(int level, int node);

    /**
     * Removes the given node from all layers.
     *
     * @param node the node id to remove
     * @return the number of layers from which it was removed
     */
    int removeNode(int node);

    /**
     * Returns an Integer stream with the nodes contained in the specified level.
     */
    IntStream nodeStream(int level);

    /**
     * Returns the maximum (coarser) level that contains a vector in the graph or -1 if the node is not in the graph.
     */
    int getMaxLevelForNode(int node);

    /**
     * @return the node of the graph to start searches at
     */
    NodeAtLevel entryNode();

    /**
     * Add the given neighbors to the given node at the specified level, maintaining diversity
     * It also adds backlinks from the neighbors to the given node.
     * The edges will only be added if the out-degree of the node is less than overflowRatio times the max degree.
     */
    void addEdges(int level, int node, NodeArray candidates, float overflowRatio);

    /**
     * Remove edges to deleted nodes and add the new connections, maintaining diversity
     */
    void replaceDeletedNeighbors(int level, int node, BitSet toDelete, NodeArray candidates);

    /**
     * Signals that all mutations have been completed and the graph will not be mutated any further.
     * Should be called by the builder after all mutations are completed (during cleanup).
     */
    void setAllMutationsCompleted();

    /**
     * Returns true if all mutations have been completed. This is signaled by calling setAllMutationsCompleted.
     */
    boolean allMutationsCompleted();
}
