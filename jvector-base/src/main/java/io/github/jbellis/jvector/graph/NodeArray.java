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
import org.agrona.collections.IntHashSet;

import java.util.Arrays;

import static java.lang.Math.min;

/**
 * NodeArray encodes nodeids and their scores relative to some other element 
 * (a query vector, or another graph node) as a pair of growable arrays. 
 * Nodes are arranged in the sorted order of their scores in descending order,
 * i.e. the most-similar nodes are first.
 */
public class NodeArray {
    public static final NodeArray EMPTY = new NodeArray(0);

    private int size;
    private float[] scores;
    private int[] nodes;

    public NodeArray(int initialSize) {
        nodes = new int[initialSize];
        scores = new float[initialSize];
    }

    // this idiosyncratic constructor exists for the benefit of subclass ConcurrentNeighborMap
    protected NodeArray(NodeArray nodeArray) {
        this.size = nodeArray.size();
        this.nodes = nodeArray.nodes;
        this.scores = nodeArray.scores;
    }

    /** always creates a new NodeArray to return, even when a1 or a2 is empty */
    static NodeArray merge(NodeArray a1, NodeArray a2) {
        NodeArray merged = new NodeArray(a1.size() + a2.size());
        int i = 0, j = 0;

        // since nodes are only guaranteed to be sorted by score -- ties can appear in any node order --
        // we need to remember all the nodes with the current score to avoid adding duplicates
        var nodesWithLastScore = new IntHashSet();
        float lastAddedScore = Float.NaN;

        // loop through both source arrays, adding the highest score element to the merged array,
        // until we reach the end of one of the sources
        while (i < a1.size() && j < a2.size()) {
            if (a1.scores[i] < a2.scores[j]) {
                // add from a2
                if (a2.scores[j] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a2.scores[j];
                }
                if (nodesWithLastScore.add(a2.nodes[j])) {
                    merged.addInOrder(a2.nodes[j], a2.scores[j]);
                }
                j++;
            } else if (a1.scores[i] > a2.scores[j]) {
                // add from a1
                if (a1.scores[i] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a1.scores[i];
                }
                if (nodesWithLastScore.add(a1.nodes[i])) {
                    merged.addInOrder(a1.nodes[i], a1.scores[i]);
                }
                i++;
            } else {
                // same score -- add both
                if (a1.scores[i] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a1.scores[i];
                }
                if (nodesWithLastScore.add(a1.nodes[i])) {
                    merged.addInOrder(a1.nodes[i], a1.scores[i]);
                }
                if (nodesWithLastScore.add(a2.nodes[j])) {
                    merged.addInOrder(a2.nodes[j], a2.scores[j]);
                }
                i++;
                j++;
            }
        }

        // If elements remain in a1, add them
        if (i < a1.size()) {
            // avoid duplicates while adding nodes with the same score
            while (i < a1.size && a1.scores[i] == lastAddedScore) {
                if (!nodesWithLastScore.contains(a1.nodes[i])) {
                    merged.addInOrder(a1.nodes[i], a1.scores[i]);
                }
                i++;
            }
            // the remaining nodes have a different score, so we can bulk-add them
            System.arraycopy(a1.nodes, i, merged.nodes, merged.size, a1.size - i);
            System.arraycopy(a1.scores, i, merged.scores, merged.size, a1.size - i);
            merged.size += a1.size - i;
        }

        // If elements remain in a2, add them
        if (j < a2.size()) {
            // avoid duplicates while adding nodes with the same score
            while (j < a2.size && a2.scores[j] == lastAddedScore) {
                if (!nodesWithLastScore.contains(a2.nodes[j])) {
                    merged.addInOrder(a2.nodes[j], a2.scores[j]);
                }
                j++;
            }
            // the remaining nodes have a different score, so we can bulk-add them
            System.arraycopy(a2.nodes, j, merged.nodes, merged.size, a2.size - j);
            System.arraycopy(a2.scores, j, merged.scores, merged.size, a2.size - j);
            merged.size += a2.size - j;
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
            float previousScore = scores[size - 1];
            assert ((previousScore >= newScore))
                    : "Nodes are added in the incorrect order! Comparing "
                    + newScore
                    + " to "
                    + Arrays.toString(ArrayUtil.copyOfSubArray(scores, 0, size));
        }
        nodes[size] = newNode;
        scores[size] = newScore;
        ++size;
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
        int insertionPoint = descSortFindRightMostInsertionPoint(newScore);
        if (duplicateExistsNear(insertionPoint, newNode, newScore)) {
            return -1;
        }

        System.arraycopy(nodes, insertionPoint, nodes, insertionPoint + 1, size - insertionPoint);
        System.arraycopy(scores, insertionPoint, scores, insertionPoint + 1, size - insertionPoint);
        nodes[insertionPoint] = newNode;
        scores[insertionPoint] = newScore;
        ++size;
        return insertionPoint;
    }

    private boolean duplicateExistsNear(int insertionPoint, int newNode, float newScore) {
        // Check to the left
        for (int i = insertionPoint - 1; i >= 0 && scores[i] == newScore; i--) {
            if (nodes[i] == newNode) {
                return true;
            }
        }

        // Check to the right
        for (int i = insertionPoint; i < size && scores[i] == newScore; i++) {
            if (nodes[i] == newNode) {
                return true;
            }
        }

        return false;
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

    protected final int descSortFindRightMostInsertionPoint(float newScore) {
        int start = 0;
        int end = size - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (scores[mid] < newScore) end = mid - 1;
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
                + REF_BYTES + AH_BYTES // nodes array
                + REF_BYTES + AH_BYTES // scores array
                + (long) size * (Integer.BYTES + Float.BYTES); // array contents
    }

    /**
     * Caution! This performs a linear scan.
     */
    @VisibleForTesting
    boolean contains(int node) {
        for (int i = 0; i < size; i++) {
            if (this.nodes[i] == node) {
                return true;
            }
        }
        return false;
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
}
