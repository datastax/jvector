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
import io.github.jbellis.jvector.graph.diversity.DiversityPruner;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.IntMap;

import static java.lang.Math.min;

/**
 * Encapsulates operations on a graph's neighbors.
 */
public class ConcurrentNeighborMap {
    final IntMap<Neighbors> neighbors;

    public ConcurrentNeighborMap() {
        this(new DenseIntMap<>(1024));
    }

    public <T> ConcurrentNeighborMap(IntMap<Neighbors> neighbors) {
        this.neighbors = neighbors;
    }

    /**
     * @return the fraction of short edges, i.e., neighbors within alpha=1.0
     */
    public void update(int nodeId, NodeArray newNeighbors) {
        while (true) {
            var old = neighbors.get(nodeId);
            var next = new Neighbors(nodeId, newNeighbors);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                return;
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
        public final int nodeId;

        /** entries in `nodes` before this index are diverse and don't need to be checked again */
        public float diverseBefore;

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

        public static long ramBytesUsed(int count) {
            return NodeArray.ramBytesUsed(count) // includes our object header
                    + Integer.BYTES // nodeId
                    + Float.BYTES; // diverseBefore
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
