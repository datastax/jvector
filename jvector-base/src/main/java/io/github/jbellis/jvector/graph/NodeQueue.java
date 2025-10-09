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
import java.util.PrimitiveIterator;
import org.agrona.collections.Int2ObjectHashMap;

import static java.lang.Math.min;

/**
 * NodeQueue uses a {@link io.github.jbellis.jvector.util.AbstractLongHeap} to store lists of nodes in a graph,
 * represented as a node id with an associated score packed together as a sortable long, which is sorted
 * primarily by score. The queue {@link #push(int, float)} operation provides either fixed-size
 * or unbounded operations, depending on the implementation subclasses, and either maxheap or minheap behavior.
 */
public class NodeQueue {
    /**
     * Ordering for the heap: MIN_HEAP keeps smallest values at the top, MAX_HEAP keeps largest values at the top.
     */
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

    /**
     * Constructs a NodeQueue with the specified heap and ordering.
     *
     * @param heap the underlying heap to store encoded node/score pairs
     * @param order the heap ordering (MIN_HEAP or MAX_HEAP)
     */
    public NodeQueue(AbstractLongHeap heap, Order order) {
        this.heap = heap;
        this.order = order;
    }

    /**
     * Returns the number of elements in the heap.
     *
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
     * Encodes then adds elements from the given iterator to this heap until elementsSize elements have been added or
     * the iterator is exhausted. The heap then re-heapifies in O(n) time (Floyd's build-heap).
     *
     * @param nodeScoreIterator the node/score pairs to add
     * @param count             the maximum number of elements to pull from the nodeScoreIterator
     */
    public void pushMany(NodeScoreIterator nodeScoreIterator, int count) {
        heap.pushMany(new NodeScoreIteratorConverter(nodeScoreIterator, this), count);
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

    /**
     * Decodes the score from the encoded heap value.
     *
     * @param heapValue the encoded long value from the heap
     * @return the decoded score
     */
    private float decodeScore(long heapValue) {
        return NumericUtils.sortableIntToFloat((int) (order.apply(heapValue) >> 32));
    }

    /**
     * Decodes the node ID from the encoded heap value.
     *
     * @param heapValue the encoded long value from the heap
     * @return the decoded node ID
     */
    private int decodeNodeId(long heapValue) {
        return (int) ~(order.apply(heapValue));
    }

    /**
     * Removes the top element and returns its node id.
     *
     * @return the node ID of the top element
     */
    public int pop() {
        return decodeNodeId(heap.pop());
    }

    /**
     * Returns a copy of the internal nodes array. Not sorted by score!
     *
     * @return an array of node IDs in heap order (not score order)
     */
    public int[] nodesCopy() {
        int size = size();
        int[] nodes = new int[size];
        for (int i = 0; i < size; i++) {
            nodes[i] = decodeNodeId(heap.get(i + 1));
        }
        return nodes;
    }

    /**
     * Reranks results and returns the worst approximate score that made it into the topK.
     * The topK results will be placed into {@code reranked}, and the remainder into {@code unused}.
     * <p>
     * Only the best result or results whose approximate score is at least {@code rerankFloor} will be reranked.
     *
     * @param topK the number of top results to rerank
     * @param reranker the exact score function to use for reranking
     * @param rerankFloor the minimum approximate score threshold for reranking
     * @param reranked the queue to receive the reranked top results
     * @param unused the collection to receive nodes that were not included in the top results
     * @return the worst approximate score among the topK results
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

    /**
     * Returns the top element's node id.
     *
     * @return the node ID of the top element
     */
    public int topNode() {
        return decodeNodeId(heap.top());
    }

    /**
     * Returns the top element's node score. For the min heap this is the minimum score. For the max
     * heap this is the maximum score.
     *
     * @return the score of the top element
     */
    public float topScore() {
        return decodeScore(heap.top());
    }

    /**
     * Removes all elements from this queue.
     */
    public void clear() {
        heap.clear();
    }

    /**
     * Sets the maximum size of the underlying heap.  Only valid when NodeQueue was created with BoundedLongHeap.
     *
     * @param maxSize the new maximum size for the heap
     * @throws ClassCastException if the underlying heap is not a BoundedLongHeap
     */
    public void setMaxSize(int maxSize) {
        ((BoundedLongHeap) heap).setMaxSize(maxSize);
    }

    @Override
    public String toString() {
        return "Nodes[" + heap.size() + "]";
    }

    /**
     * Applies the given consumer to each node/score pair in this queue.
     * The order of iteration is not guaranteed to be sorted by score.
     *
     * @param consumer the consumer to apply to each node/score pair
     */
    public void foreach(NodeConsumer consumer) {
        for (int i = 0; i < heap.size(); i++) {
            long heapValue = heap.get(i + 1);
            consumer.accept(decodeNodeId(heapValue), decodeScore(heapValue));
        }
    }

    /**
     * A consumer that accepts node ID and score pairs.
     */
    @FunctionalInterface
    public interface NodeConsumer {
        /**
         * Accepts a node ID and its associated score.
         *
         * @param node the node ID
         * @param score the score associated with the node
         */
        void accept(int node, float score);
    }

    /**
     * Iterator over node and score pairs.
     */
    public interface NodeScoreIterator {
        /**
         * Checks if there are more elements to iterate over.
         *
         * @return true if there are more elements, false otherwise
         */
        boolean hasNext();

        /**
         * Returns the next node ID and advances the iterator.
         *
         * @return the next node ID
         */
        int pop();

        /**
         * Returns the score of the next node without advancing the iterator.
         *
         * @return the next node score
         */
        float topScore();
    }

    /**
     * Copies the other NodeQueue to this one. If its order (MIN_HEAP or MAX_HEAP) is the same as this,
     * it is copied verbatim. If it differs, every element is re-inserted into this.
     *
     * @param other the other node queue to copy from
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

    /**
     * Converts a NodeScoreIterator to a PrimitiveIterator.OfLong by encoding the node and score as a long.
     */
    private static class NodeScoreIteratorConverter implements PrimitiveIterator.OfLong {
        private final NodeScoreIterator it;
        private final NodeQueue queue;

        /**
         * Constructs a converter that wraps the given iterator.
         *
         * @param it the node score iterator to wrap
         * @param queue the node queue used for encoding
         */
        public NodeScoreIteratorConverter(NodeScoreIterator it, NodeQueue queue) {
            this.it = it;
            this.queue = queue;
        }

        @Override
        public boolean hasNext() {
            return it.hasNext();
        }

        @Override
        public long nextLong() {
            // pop() advances the iterator
            float score = it.topScore();
            int node = it.pop();
            return queue.encode(node, score);
        }
    }
}
