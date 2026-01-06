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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.util.Arrays;
import java.util.stream.IntStream;

import static java.lang.Math.min;

/**
 * NodeArray encodes node IDs and their scores relative to some other element
 * (a query vector, or another graph node) as a pair of growable arrays.
 * Nodes are arranged in ascending order using the nodeIDs.
 */
public class NodeArray {
    public static final NodeArray EMPTY = new NodeArray(0);

    private int size;
    private float[] scores;
    private int[] nodes;
    private float minScore;

    public NodeArray(int initialSize) {
        nodes = new int[initialSize];
        scores = new float[initialSize];
        size = 0;
        minScore = Float.POSITIVE_INFINITY;
    }

    // this idiosyncratic constructor exists for the benefit of subclass ConcurrentNeighborMap
    protected NodeArray(NodeArray nodeArray) {
        this.size = nodeArray.size();
        this.nodes = nodeArray.nodes;
        this.scores = nodeArray.scores;
        minScore = nodeArray.minScore;
    }

    /** always creates a new NodeArray to return, even when a1 or a2 is empty.
     * If a node ID is present in both, the one from a1 will be added. */
    static NodeArray merge(NodeArray a1, NodeArray a2) {
        NodeArray merged = new NodeArray(a1.size() + a2.size());
        int i1 = 0, i2 = 0;

        // loop through both source arrays, adding the highest score element to the merged array,
        // until we reach the end of one of the sources
        while (i1 < a1.size() && i2 < a2.size()) {
            if (a1.nodes[i1] < a2.nodes[i2]) {
                // add from a1
                merged.addInOrder(a1.nodes[i1], a1.scores[i1]);
                i1++;
            } else if (a1.nodes[i1] > a2.nodes[i2]) {
                // add from a2
                merged.addInOrder(a2.nodes[i2], a2.scores[i2]);
                i2++;
            } else {
                // same node -- add from a1
                merged.addInOrder(a1.nodes[i1], a1.scores[i1]);
                i1++;
                i2++;
            }
        }

        // If elements remain in a1, add them
        if (i1 < a1.size()) {
            for (; i1 < a1.size; i1++) {
                merged.addInOrder(a1.nodes[i1], a1.scores[i1]);
            }
            merged.size += a1.size - i1;
        }

        // If elements remain in a2, add them
        if (i2 < a2.size()) {
            for (; i2 < a2.size; i2++) {
                merged.addInOrder(a2.nodes[i2], a2.scores[i2]);
            }
            merged.size += a2.size - i2;
        }
        return merged;
    }

    /**
     * Add a new node to the NodeArray. The new node must be worse than all previously stored
     * nodes.
     */
    public void addInOrder(int newNode, float newScore) {
        if (size == nodes.length) {
            growArrays();
        }
        if (size > 0) {
            int previousNode = nodes[size - 1];
            assert ((previousNode <= newNode))
                    : "Nodes are added in the incorrect order! Comparing "
                    + newNode
                    + " to "
                    + Arrays.toString(ArrayUtil.copyOfSubArray(nodes, 0, size));
        }
        nodes[size] = newNode;
        scores[size] = newScore;
        ++size;
        minScore = min(minScore, newScore);
    }

    /**
     * Returns the index at which the given node should be inserted to maintain sorted order,
     * or -1 if the node already exists in the array.
     */
    int insertionPoint(int newNode) {
        int insertionPoint = incSortFindRightMostInsertionPoint(newNode);
        return duplicateExists(insertionPoint, newNode) ? -1 : insertionPoint;
    }

    /**
     * Add a new node to the NodeArray into a correct sort position according to its score.
     * Duplicate node + score pairs are ignored.
     *
     * @return the insertion point of the new node, or -1 if it already existed
     */
    public int insertSorted(int newNode, float newScore) {
        if (size == nodes.length) {
            growArrays();
        }
        int insertionPoint = insertionPoint(newNode);
        if (insertionPoint == -1) {
            return -1;
        }

        return insertInternal(insertionPoint, newNode, newScore);
    }

    /**
     * Add a new node to the NodeArray into the specified insertion point.
     */
    void insertAt(int insertionPoint, int newNode, float newScore) {
        if (size == nodes.length) {
            growArrays();
        }
        insertInternal(insertionPoint, newNode, newScore);
    }

    private int insertInternal(int insertionPoint, int newNode, float newScore) {
        System.arraycopy(nodes, insertionPoint, nodes, insertionPoint + 1, size - insertionPoint);
        System.arraycopy(scores, insertionPoint, scores, insertionPoint + 1, size - insertionPoint);
        nodes[insertionPoint] = newNode;
        scores[insertionPoint] = newScore;
        ++size;
        return insertionPoint;
    }

    private boolean duplicateExists(int insertionPoint, int node) {
        if (insertionPoint < size && nodes[insertionPoint] == node) return true;
        if (insertionPoint + 1 < size && nodes[insertionPoint + 1] == node) return true;
        return insertionPoint - 1 >= 0 && nodes[insertionPoint - 1] == node;
    }

    /**
     * Retains only the elements in the current NodeArray whose corresponding index
     * is set in the given BitSet.
     * <p>
     * This modifies the array in place, preserving the relative order of the elements retained.
     * <p>
     *
     * @param selected A BitSet where the bit at index i is set if the i-th element should be retained.
     *                 (Thus, the elements of selected represent positions in the NodeArray, NOT node ids.)
     */
    public void retain(Bits selected) {
        minScore = Float.POSITIVE_INFINITY;

        int writeIdx = 0; // index for where to write the next retained element

        for (int readIdx = 0; readIdx < size; readIdx++) {
            if (selected.get(readIdx)) {
                if (writeIdx != readIdx) {
                    // Move the selected entries to the front while maintaining their relative order
                    nodes[writeIdx] = nodes[readIdx];
                    scores[writeIdx] = scores[readIdx];
                }
                // else { we haven't created any gaps in the backing arrays yet, so we don't need to move anything }
                writeIdx++;
                minScore = min(minScore, scores[readIdx]);
            }
        }

        size = writeIdx;
    }

    public NodeArray copy() {
        return copy(size);
    }

    public NodeArray copy(int newSize) {
        if (size > newSize) {
            throw new IllegalArgumentException(String.format("Cannot copy %d nodes to a smaller size %d", size, newSize));
        }

        NodeArray copy = new NodeArray(newSize);
        copy.size = size;
        System.arraycopy(nodes, 0, copy.nodes, 0, size);
        System.arraycopy(scores, 0, copy.scores, 0, size);
        return copy;
    }

    protected final void growArrays() {
        nodes = ArrayUtil.grow(nodes);
        scores = ArrayUtil.growExact(scores, nodes.length);
    }

    public int size() {
        return size;
    }

    public void clear() {
        size = 0;
    }

    public void removeLast() {
        size--;
    }

    public void removeIndex(int idx) {
        System.arraycopy(nodes, idx + 1, nodes, idx, size - idx - 1);
        System.arraycopy(scores, idx + 1, scores, idx, size - idx - 1);
        size--;
    }

    @Override
    public String toString() {
        var sb = new StringBuilder("NodeArray(");
        sb.append(size).append("/").append(nodes.length).append(") [");
        for (int i = 0; i < size; i++) {
            sb.append("(").append(nodes[i]).append(",").append(scores[i]).append(")").append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    protected final int incSortFindRightMostInsertionPoint(int newNode) {
        int start = 0;
        int end = size - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (nodes[mid] > newNode) end = mid - 1;
            else start = mid + 1;
        }
        return start;
    }

    public static long ramBytesUsed(int size) {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;

        return OH_BYTES
                + Integer.BYTES // size field
                + Float.BYTES // minScore field
                + REF_BYTES + AH_BYTES // nodes array
                + REF_BYTES + AH_BYTES // scores array
                + (long) size * (Integer.BYTES + Float.BYTES); // array contents
    }

    @VisibleForTesting
    boolean contains(int node) {
        int insertionPoint = incSortFindRightMostInsertionPoint(node);
        return duplicateExists(insertionPoint, node);
    }

    @VisibleForTesting
    int[] copyDenseNodes() {
        return Arrays.copyOf(nodes, size);
    }

    @VisibleForTesting
    float[] copyDenseScores() {
        return Arrays.copyOf(scores, size);
    }

    /**
     * Insert a new node, without growing the array.  If the array is full, drop the worst existing node to make room.
     * (Even if the worst existing one is better than newNode!)
     */
    protected int insertOrReplaceWorst(int newNode, float newScore) {
        size = min(size, nodes.length - 1);
        return insertSorted(newNode, newScore);
    }

    public float getScore(int i) {
        return scores[i];
    }

    public int getNode(int i) {
        return nodes[i];
    }

    protected int getArrayLength() {
        return nodes.length;
    }

    public float getMinScore() {
        return minScore;
    }

    public NodesIterator getIteratorSortedByScores() {
        return new ScoreSortedNeighborIterator(this);
    }

    private static class ScoreSortedNeighborIterator implements NodesIterator {
        private final NodeArray array;
        private final int[] sortedIndices;
        private int i;

        private ScoreSortedNeighborIterator(NodeArray array) {
            this.array = array;
            sortedIndices = IntStream.range(0, this.array.size())
                    .boxed().sorted((i, j) -> Float.compare(this.array.getScore(j), this.array.getScore(i)))
                    .mapToInt(ele -> ele).toArray();
            i = 0;
        }

        @Override
        public int size() {
            return array.size();
        }

        @Override
        public boolean hasNext() {
            return i < array.size();
        }

        @Override
        public int nextInt() {
            return sortedIndices[i++];
        }
    }
}