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
    protected int size;
    float[] score;
    int[] node;

    public NodesUnsorted(int initialSize) {
        node = new int[initialSize];
        score = new float[initialSize];
    }

    /**
     * Add a new node to the NodeArray. The new node must be worse than all previously stored
     * nodes.
     */
    public void add(int newNode, float newScore) {
        if (size == node.length) {
            growArrays();
        }
        node[size] = newNode;
        score[size] = newScore;
        ++size;
    }

    protected final void growArrays() {
        node = ArrayUtil.grow(node);
        score = ArrayUtil.growExact(score, node.length);
    }

    public int size() {
        return size;
    }

    public void clear() {
        size = 0;
    }

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
