/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.IntMap;

import static java.lang.Math.min;

/**
 * Encapsulates operations on a graph's neighbors.
 */
public class ConcurrentNeighborMap {
    final IntMap<Neighbors> neighbors;

    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    final float alpha;

    /** used to compute diversity */
    final BuildScoreProvider scoreProvider;

    /** the maximum number of neighbors desired per node */
    public final int maxDegree;
    /** the maximum number of neighbors a node can have temporarily during construction */
    public final int maxOverflowDegree;

    public ConcurrentNeighborMap(BuildScoreProvider scoreProvider, int maxDegree, int maxOverflowDegree, float alpha) {
        this(new DenseIntMap<>(1024), scoreProvider, maxDegree, maxOverflowDegree, alpha);
    }

    public <T> ConcurrentNeighborMap(IntMap<Neighbors> neighbors, BuildScoreProvider scoreProvider, int maxDegree, int maxOverflowDegree, float alpha) {
        assert maxDegree <= maxOverflowDegree : String.format("maxDegree %d exceeds maxOverflowDegree %d", maxDegree, maxOverflowDegree);
        this.neighbors = neighbors;
        this.alpha = alpha;
        this.scoreProvider = scoreProvider;
        this.maxDegree = maxDegree;
        this.maxOverflowDegree = maxOverflowDegree;
    }

    public void insertEdge(int fromId, int toId, float score, float overflow) {
        while (true) {
            var old = neighbors.get(fromId);
            var next = old.insert(toId, score, overflow, this);
            if (next == old || neighbors.compareAndPut(fromId, old, next)) {
                break;
            }
        }
    }

    public void insertEdgeNotDiverse(int fromId, int toId, float score) {
        while (true) {
            var old = neighbors.get(fromId);
            var next = old.insertNotDiverse(toId, score, this);
            if (next == old || neighbors.compareAndPut(fromId, old, next)) {
                break;
            }
        }
    }

    /**
     * @return the fraction of short edges, i.e., neighbors within alpha=1.0
     */
    public double enforceDegree(int nodeId) {
        var old = neighbors.get(nodeId);
        if (old == null) {
            return Double.NaN;
        }

        while (true) {
            old = neighbors.get(nodeId);
            var nwse = old.enforceDegree(this);
            if (nwse.neighbors == old || neighbors.compareAndPut(nodeId, old, nwse.neighbors)) {
                return nwse.shortEdges;
            }
        }
    }

    public void replaceDeletedNeighbors(int nodeId, BitSet toDelete, NodeArray candidates) {
        while (true) {
            var old = neighbors.get(nodeId);
            var next = old.replaceDeletedNeighbors(toDelete, candidates, this);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                break;
            }
        }
    }

    public Neighbors insertDiverse(int nodeId, NodeArray candidates) {
        while (true) {
            var old = neighbors.get(nodeId);
            assert old != null : nodeId; // graph.addNode should always be called before this method
            var next = old.insertDiverse(candidates, this);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                return next;
            }
        }
    }

    public Neighbors get(int node) {
        return neighbors.get(node);
    }

    public int size() {
        return neighbors.size();
    }

    /**
     * Only for internal use and by Builder loading a saved graph
     */
    void addNode(int nodeId, NodeArray nodes) {
        var next = new Neighbors(nodeId, nodes);
        if (!neighbors.compareAndPut(nodeId, null, next)) {
            throw new IllegalStateException("Node " + nodeId + " already exists");
        }
    }

    public void addNode(int nodeId) {
        addNode(nodeId, new NodeArray(0));
    }

    public Neighbors remove(int node) {
        return neighbors.remove(node);
    }

    public boolean contains(int nodeId) {
        return neighbors.containsKey(nodeId);
    }

    public void forEach(DenseIntMap.IntBiConsumer<Neighbors> consumer) {
        neighbors.forEach(consumer);
    }

    int nodeArrayLength() {
        // one extra so that insert() against a full NodeArray doesn't invoke growArrays()
        return maxOverflowDegree + 1;
    }

    /**
     * Add a link from every node in the NodeArray to the target toId.
     * If overflow is > 1.0, allow the number of neighbors to exceed maxConnections temporarily.
     */
    public void backlink(NodeArray nodes, int toId, float overflow) {
        for (int i = 0; i < nodes.size(); i++) {
            int nbr = nodes.getNode(i);
            float nbrScore = nodes.getScore(i);
            insertEdge(nbr, toId, nbrScore, overflow);
        }
    }

    /**
     * A concurrent set of neighbors that encapsulates diversity/pruning mechanics.
     * <p>
     * Nothing is modified in place; all mutating methods return a new instance.  These methods
     * are private and should only be exposed through the parent ConcurrentNeighborMap, which
     * performs the appropriate CAS dance.
     * <p>
     * CNM is passed as an explicit parameter to these methods (instead of making this a non-static
     * inner class) to avoid the overhead on the heap of the CNM$this reference.  Similarly,
     * Neighbors extends NodeArray instead of composing with it to avoid the overhead of an extra
     * object header.
     */
    public static class Neighbors extends NodeArray {
        /** the node id whose neighbors we are storing */
        private final int nodeId;

        /** entries in `nodes` before this index are diverse and don't need to be checked again */
        private int diverseBefore;

        /**
         * uses the node and score references directly from `nodeArray`, without copying
         * `nodeArray` is assumed to have had diversity enforced already
         */
        private Neighbors(int nodeId, NodeArray nodeArray) {
            super(nodeArray);
            this.nodeId = nodeId;
            this.diverseBefore = size();
        }

        public NodesIterator iterator() {
            return new NeighborIterator(this);
        }

        @Override
        public Neighbors copy() {
            return copy(size());
        }

        @Override
        public Neighbors copy(int newSize) {
            var superCopy = new NodeArray(this).copy(newSize);
            return new Neighbors(nodeId, superCopy);
        }

        /**
         * Enforce maxConnections as a hard cap, since we allow it to be exceeded temporarily during construction
         * for efficiency.  This method is threadsafe, but if you call it concurrently with other inserts,
         * the limit may end up being exceeded again.
         */
        private NeighborWithShortEdges enforceDegree(ConcurrentNeighborMap map) {
            if (size() <= map.maxDegree) {
                return new NeighborWithShortEdges(this, Double.NaN);
            }
            var next = copy();
            double shortEdges = retainDiverse(next, diverseBefore, map);
            next.diverseBefore = next.size();
            return new NeighborWithShortEdges(next, shortEdges);
        }

        private Neighbors replaceDeletedNeighbors(Bits deletedNodes, NodeArray candidates, ConcurrentNeighborMap map) {
            // copy the non-deleted neighbors to a new NodeArray
            var liveNeighbors = new NodeArray(size());
            for (int i = 0; i < size(); i++) {
                int nodeId = getNode(i);
                if (!deletedNodes.get(nodeId)) {
                    liveNeighbors.addInOrder(nodeId, getScore(i));
                }
            }

            // merge the remaining neighbors with the candidates and keep the diverse results
            NodeArray merged = NodeArray.merge(liveNeighbors, candidates);
            retainDiverse(merged, 0, map);
            return new Neighbors(nodeId, merged);
        }

        /**
         * For each candidate (going from best to worst), select it only if it is closer to target than it
         * is to any of the already-selected candidates. This is maintained whether those other neighbors
         * were selected by this method, or were added as a "backlink" to a node inserted concurrently
         * that chose this one as a neighbor.
         */
        private Neighbors insertDiverse(NodeArray toMerge, ConcurrentNeighborMap map) {
            if (toMerge.size() == 0) {
                return this;
            }

            // merge all the candidates into a single array and compute the diverse ones to keep
            // from that.
            NodeArray merged;
            if (size() > 0) {
                merged = NodeArray.merge(this, toMerge);
                retainDiverse(merged, 0, map);
            } else {
                merged = toMerge.copy(); // still need to copy in case we lose the race
                retainDiverse(merged, 0, map);
            }
            // insertDiverse usually gets called with a LOT of candidates, so trim down the resulting NodeArray
            var nextNodes = merged.getArrayLength() <= map.nodeArrayLength()
                    ? merged
                    : merged.copy(map.nodeArrayLength());
            return new Neighbors(nodeId, nextNodes);
        }

        private Neighbors insertNotDiverse(int node, float score, ConcurrentNeighborMap map) {
            int maxDegree = map.maxDegree;
            assert size() <= maxDegree : "insertNotDiverse called before enforcing degree/diversity";
            var next = copy(maxDegree); // we are only called during cleanup -- use actual maxDegree not nodeArrayLength()
            int insertedAt = next.insertOrReplaceWorst(node, score);
            if (insertedAt == -1) {
                // node already existed in the array -- this is rare enough that we don't check up front
                return this;
            }
            next.diverseBefore = min(insertedAt, diverseBefore);
            return next;
        }

        /**
         * Retain the diverse neighbors, updating `neighbors` in place
         * @return post-diversity short edges fraction
         */
        private double retainDiverse(NodeArray neighbors, int diverseBefore, ConcurrentNeighborMap map) {
            BitSet selected = new FixedBitSet(neighbors.size());
            for (int i = 0; i < min(diverseBefore, map.maxDegree); i++) {
                selected.set(i);
            }

            double shortEdges = retainDiverseInternal(neighbors, diverseBefore, selected, map);
            neighbors.retain(selected);
            return shortEdges;
        }

        /**
         * update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
         * @return the fraction of short edges (neighbors within alpha=1.0)
         */
        private double retainDiverseInternal(NodeArray neighbors, int diverseBefore, BitSet selected, ConcurrentNeighborMap map) {
            int nSelected = diverseBefore;
            double shortEdges = Double.NaN;
            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            for (float a = 1.0f; a <= map.alpha + 1E-6 && nSelected < map.maxDegree; a += 0.2f) {
                for (int i = diverseBefore; i < neighbors.size() && nSelected < map.maxDegree; i++) {
                    if (selected.get(i)) {
                        continue;
                    }

                    int cNode = neighbors.getNode(i);
                    float cScore = neighbors.getScore(i);
                    var sf = map.scoreProvider.diversityProviderFor(cNode).scoreFunction();
                    if (isDiverse(cNode, cScore, neighbors, sf, selected, a)) {
                        selected.set(i);
                        nSelected++;
                    }
                }

                if (a == 1.0f) {
                    // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                    // when we don't have to worry about concurrent changes
                    shortEdges = nSelected / (float) map.maxDegree;
                }
            }
            return shortEdges;
        }

        // is the candidate node with the given score closer to the base node than it is to any of the
        // already-selected neighbors
        private boolean isDiverse(int node, float score, NodeArray others, ScoreFunction sf, BitSet selected, float alpha) {
            assert others.size() > 0;

            for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
                int otherNode = others.getNode(i);
                if (node == otherNode) {
                    break;
                }
                if (sf.similarityTo(otherNode) > score * alpha) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Insert a new neighbor, maintaining our size cap by removing the least diverse neighbor if
         * necessary. "Overflow" is the factor by which to allow going over the size cap temporarily.
         */
        private Neighbors insert(int neighborId, float score, float overflow, ConcurrentNeighborMap map) {
            assert neighborId != nodeId : "can't add self as neighbor at node " + nodeId;

            int hardMax = (int) (overflow * map.maxDegree);
            assert hardMax <= map.maxOverflowDegree
                    : String.format("overflow %s could exceed max overflow degree %d", overflow, map.maxOverflowDegree);

            var next = copy(map.nodeArrayLength());
            int insertionPoint = next.insertSorted(neighborId, score);
            if (insertionPoint == -1) {
                // "new" node already existed
                return this;
            }

            // batch up the enforcement of the max connection limit, since otherwise
            // we do a lot of duplicate work scanning nodes that we won't remove
            next.diverseBefore = min(insertionPoint, diverseBefore);
            if (next.size() > hardMax) {
                retainDiverse(next, next.diverseBefore, map);
                next.diverseBefore = next.size();
            }

            return next;
        }

        public static long ramBytesUsed(int count) {
            return NodeArray.ramBytesUsed(count) // includes our object header
                    + Integer.BYTES // nodeId
                    + Integer.BYTES; // diverseBefore
        }

        /** Only for testing; this is a linear search */
        @VisibleForTesting
        boolean contains(int i) {
            var it = this.iterator();
            while (it.hasNext()) {
                if (it.nextInt() == i) {
                    return true;
                }
            }
            return false;
        }
    }

    private static class NeighborWithShortEdges {
        public final Neighbors neighbors;
        public final double shortEdges;

        public NeighborWithShortEdges(Neighbors neighbors, double shortEdges) {
            this.neighbors = neighbors;
            this.shortEdges = shortEdges;
        }
    }

    private static class NeighborIterator implements NodesIterator {
        private final NodeArray neighbors;
        private int i;

        private NeighborIterator(NodeArray neighbors) {
            this.neighbors = neighbors;
            i = 0;
        }

        @Override
        public int size() {
            return neighbors.size();
        }

        @Override
        public boolean hasNext() {
            return i < neighbors.size();
        }

        @Override
        public int nextInt() {
            return neighbors.getNode(i++);
        }
    }
}
