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
import io.github.jbellis.jvector.util.AbstractLongHeap;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.NumericUtils;
import org.agrona.collections.Int2ObjectHashMap;

import java.util.Arrays;

import static java.lang.Math.min;

/**
 * NodeQueue uses a {@link io.github.jbellis.jvector.util.AbstractLongHeap} to store lists of nodes in a graph,
 * represented as a node id with an associated score packed together as a sortable long, which is sorted
 * primarily by score. The queue {@link #push(int, float)} operation provides either fixed-size
 * or unbounded operations, depending on the implementation subclasses, and either maxheap or minheap behavior.
 */
public class NodeQueue {
    public enum Order {
        /** Smallest values at the top of the heap */
        MIN_HEAP {
            @Override
            long apply(long v) {
                return v;
            }
        },
        /** Largest values at the top of the heap */
        MAX_HEAP {
            @Override
            long apply(long v) {
                // This cannot be just `-v` since Long.MIN_VALUE doesn't have a positive counterpart. It
                // needs a function that returns MAX_VALUE for MIN_VALUE and vice-versa.
                return -1 - v;
            }
        };

        abstract long apply(long v);
    }

    private final AbstractLongHeap heap;
    private final Order order;

    public NodeQueue(AbstractLongHeap heap, Order order) {
        this.heap = heap;
        this.order = order;
    }

    /**
     * @return the number of elements in the heap
     */
    public int size() {
        return heap.size();
    }

    /**
     * Adds a new graph node to the heap.  Will extend storage or replace the worst element
     * depending on the type of heap it is.
     *
     * @param newNode  the node id
     * @param newScore the relative similarity score to the node of the owner
     *
     * @return true if the new value was added.
     */
    public boolean push(int newNode, float newScore) {
        return heap.push(encode(newNode, newScore));
    }

    /**
     * Encodes the node ID and its similarity score as long.  If two scores are equals,
     * the smaller node ID wins.
     *
     * <p>The most significant 32 bits represent the float score, encoded as a sortable int.
     *
     * <p>The less significant 32 bits represent the node ID.
     *
     * <p>The bits representing the node ID are complemented to guarantee the win for the smaller node
     * ID.
     *
     * <p>The AND with 0xFFFFFFFFL (a long with first 32 bit as 1) is necessary to obtain a long that
     * has
     *
     * <p>The most significant 32 bits to 0
     *
     * <p>The less significant 32 bits represent the node ID.
     *
     * @param node  the node ID
     * @param score the node score
     * @return the encoded score, node ID
     */
    private long encode(int node, float score) {
        assert node >= 0 : node;
        return order.apply(
                (((long) NumericUtils.floatToSortableInt(score)) << 32) | (0xFFFFFFFFL & ~node));
    }

    private float decodeScore(long heapValue) {
        return NumericUtils.sortableIntToFloat((int) (order.apply(heapValue) >> 32));
    }

    private int decodeNodeId(long heapValue) {
        return (int) ~(order.apply(heapValue));
    }

    /** Removes the top element and returns its node id. */
    public int pop() {
        return decodeNodeId(heap.pop());
    }

    /** Returns a copy of the internal nodes array. Not sorted by score! */
    public int[] nodesCopy() {
        int size = size();
        int[] nodes = new int[size];
        for (int i = 0; i < size; i++) {
            nodes[i] = decodeNodeId(heap.get(i + 1));
        }
        return nodes;
    }

    /**
     * Rerank results and return the worst approximate score that made it into the topK.
     * The topK results will be placed into `reranked`, and the remainder into `unused`.
     * <p>
     * Only the best result or results whose approximate score is at least `rerankFloor` will be reranked.
     */
    public float rerank(int topK, ScoreFunction.ExactScoreFunction reranker, float rerankFloor, NodeQueue reranked, NodesUnsorted unused) {
        // Rescore the nodes whose approximate score meets the floor.  Nodes that do not will be marked as -1
        int[] ids = new int[size()];
        float[] exactScores = new float[size()];
        var approximateScoresById = new Int2ObjectHashMap<Float>();
        float bestScore = Float.NEGATIVE_INFINITY;
        int bestIndex = -1;
        int scoresAboveFloor = 0;
        for (int i = 0; i < size(); i++) {
            long heapValue = heap.get(i + 1);
            float score = decodeScore(heapValue);
            var nodeId = decodeNodeId(heapValue);
            // track the best score found so far in case nothing is above the floor
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }

            if (score >= rerankFloor) {
                // rerank this one
                ids[i] = nodeId;
                exactScores[i] = reranker.similarityTo(ids[i]);
                approximateScoresById.put(ids[i], Float.valueOf(score));
                scoresAboveFloor++;
            } else {
                // mark it unranked
                ids[i] = -1;
            }
        }

        if (scoresAboveFloor == 0 && bestIndex >= 0) {
            // if nothing was above the floor, then rerank the best one found
            ids[bestIndex] = decodeNodeId(heap.get(bestIndex + 1));
            exactScores[bestIndex] = reranker.similarityTo(ids[bestIndex]);
            approximateScoresById.put(ids[bestIndex], Float.valueOf(bestScore));
        }

        // go through the entries and add to the appropriate collection
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] == -1) {
                unused.add(decodeNodeId(heap.get(i + 1)), decodeScore(heap.get(i + 1)));
                continue;
            }

            // if the reranked queue is full, then either this node, or the one it replaces on the heap, will be added
            // to the unused pile, but push() can't tell us what node was evicted when the queue was already full, so
            // we examine that manually
            if (reranked.size() < topK) {
                reranked.push(ids[i], exactScores[i]);
            } else if (exactScores[i] > reranked.topScore()) {
                int evictedNode = reranked.topNode();
                unused.add(evictedNode, approximateScoresById.get(evictedNode));
                reranked.push(ids[i], exactScores[i]);
            } else {
                unused.add(ids[i], decodeScore(heap.get(i + 1)));
            }
        }

        // final pass to find the worst approximate score in the topK
        // (we can't do this as part of the earlier loops because we don't know which nodes will be in the final topK)
        float worstApproximateInTopK = Float.POSITIVE_INFINITY;
        if (reranked.size() < topK) {
            return worstApproximateInTopK;
        }
        for (int i = 0; i < reranked.size(); i++) {
            int nodeId = decodeNodeId(reranked.heap.get(i + 1));
            worstApproximateInTopK = min(worstApproximateInTopK, approximateScoresById.get(nodeId));
        }

        return worstApproximateInTopK;
    }

    /** Returns the top element's node id. */
    public int topNode() {
        return decodeNodeId(heap.top());
    }

    /**
     * Returns the top element's node score. For the min heap this is the minimum score. For the max
     * heap this is the maximum score.
     */
    public float topScore() {
        return decodeScore(heap.top());
    }

    public void clear() {
        heap.clear();
    }

    /**
     * Set the max size of the underlying heap.  Only valid when NodeQueue was created with BoundedLongHeap.
     */
    public void setMaxSize(int maxSize) {
        ((BoundedLongHeap) heap).setMaxSize(maxSize);
    }

    @Override
    public String toString() {
        return "Nodes[" + heap.size() + "]";
    }

    public void foreach(NodeConsumer consumer) {
        for (int i = 0; i < heap.size(); i++) {
            long heapValue = heap.get(i + 1);
            consumer.accept(decodeNodeId(heapValue), decodeScore(heapValue));
        }
    }

    @FunctionalInterface
    public interface NodeConsumer {
        void accept(int node, float score);
    }

    /**
     * Copies the other NodeQueue to this one. If its order (MIN_HEAP or MAX_HEAP) is the same as this,
     * it is copied verbatim. If it differs, every lement is re-inserted into this.
     * @param other the other node queue.
     */
    public void copyFrom(NodeQueue other) {
        if (this.order == other.order) {
            this.heap.copyFrom(other.heap);
        } else {
            // can't avoid re-encoding since order influences it
            clear();
            other.foreach(this::push);
        }
    }
}
