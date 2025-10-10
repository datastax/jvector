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

import io.github.jbellis.jvector.graph.NodeQueue.NodeConsumer;
import io.github.jbellis.jvector.util.ArrayUtil;

/**
 * NodesUnsorted contains scored node ids in insertion order.
 */
public class NodesUnsorted {
    /**
     * The current number of nodes stored in this collection.
     */
    protected int size;
    /**
     * Array of scores corresponding to each node, stored in insertion order.
     */
    float[] score;
    /**
     * Array of node IDs stored in insertion order.
     */
    int[] node;

    /**
     * Constructs a new NodesUnsorted collection with the specified initial capacity.
     * @param initialSize the initial capacity for storing nodes
     */
    public NodesUnsorted(int initialSize) {
        node = new int[initialSize];
        score = new float[initialSize];
    }

    /**
     * Adds a new node and its score to the collection in insertion order. Unlike NodeArray, this
     * collection does not maintain any sorting.
     * @param newNode the node ID to add
     * @param newScore the score associated with this node
     */
    public void add(int newNode, float newScore) {
        if (size == node.length) {
            growArrays();
        }
        node[size] = newNode;
        score[size] = newScore;
        ++size;
    }

    /**
     * Grows the internal storage arrays when capacity is exhausted, using the default growth strategy.
     */
    protected final void growArrays() {
        node = ArrayUtil.grow(node);
        score = ArrayUtil.growExact(score, node.length);
    }

    /**
     * Returns the number of nodes currently stored in this collection.
     * @return the count of nodes
     */
    public int size() {
        return size;
    }

    /**
     * Removes all nodes from the collection, resetting the size to zero without releasing memory.
     */
    public void clear() {
        size = 0;
    }

    /**
     * Iterates over all node-score pairs in insertion order, calling the consumer for each pair.
     * @param consumer the function to call for each node-score pair
     */
    public void foreach(NodeConsumer consumer) {
        for (int i = 0; i < size; i++) {
            consumer.accept(node[i], score[i]);
        }
    }

    @Override
    public String toString() {
        return "NodesUnsorted[" + size + "]";
    }
}
