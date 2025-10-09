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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.ConcurrentNeighborMap.Neighbors;
import io.github.jbellis.jvector.graph.diversity.DiversityProvider;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.SparseIntMap;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;
import org.agrona.collections.IntArrayList;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.StampedLock;
import java.util.stream.IntStream;

/**
 * An {@link ImmutableGraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>The base layer (layer 0) contains all nodes, while higher layers are stored in sparse maps.
 * For searching, use a view obtained from {@link #getView()} which supports level–aware operations.
 */
public class OnHeapGraphIndex implements MutableGraphIndex {
    // Used for saving and loading OnHeapGraphIndex
    public static final int MAGIC = 0x75EC4012; // JVECTOR, with some imagination

    // The current entry node for searches
    private final AtomicReference<NodeAtLevel> entryPoint;

    // Layers of the graph, with layer 0 as the bottom (dense) layer containing all nodes.
    final List<ConcurrentNeighborMap> layers = new ArrayList<>();

    private final CompletionTracker completions;
    private final ThreadSafeGrowableBitSet deletedNodes = new ThreadSafeGrowableBitSet(0);
    private final AtomicInteger maxNodeId = new AtomicInteger(-1);

    // Maximum number of neighbors (edges) per node per layer
    final List<Integer> maxDegrees;
    // The ratio by which we can overflow the neighborhood of a node during construction. Since it is a multiplicative
    // ratio, i.e., the maximum allowable degree if maxDegree * overflowRatio, it should be higher than 1.
    private final double overflowRatio;

    private volatile boolean allMutationsCompleted = false;

    OnHeapGraphIndex(List<Integer> maxDegrees, double overflowRatio, DiversityProvider diversityProvider) {
        this.overflowRatio = overflowRatio;
        this.maxDegrees = new IntArrayList();
        setDegrees(maxDegrees);
        entryPoint = new AtomicReference<>();
        this.completions = new CompletionTracker(1024);
        // Initialize the base layer (layer 0) with a dense map.
        this.layers.add(new ConcurrentNeighborMap(
                new DenseIntMap<>(1024),
                diversityProvider,
                getDegree(0),
                (int) (getDegree(0) * overflowRatio))
        );
    }

    /**
     * Returns the neighbors for the given node at the specified level, or null if the node does not exist.
     *
     * @param level the layer
     * @param node  the node id
     * @return the Neighbors structure or null
     */
    Neighbors getNeighbors(int level, int node) {
        if (level >= layers.size()) {
            return null;
        }
        return layers.get(level).get(node);
    }

    @Override
    public NodesIterator getNeighborsIterator(NodeAtLevel nodeAtLevel) {
        return getNeighborsIterator(nodeAtLevel.level, nodeAtLevel.node);
    }

    @Override
    public NodesIterator getNeighborsIterator(int level, int node) {
        if (level >= layers.size()) {
            return NodesIterator.EMPTY_NODE_ITERATOR;
        }
        var neighs = layers.get(level).get(node);
        if (neighs == null) {
            return NodesIterator.EMPTY_NODE_ITERATOR;
        } else {
            return neighs.iterator();
        }
    }

    @Override
    public int getMaxLevelForNode(int node) {
        int maxLayer = -1;
        for (int lvl = 0; lvl < layers.size(); lvl++) {
            if (getNeighbors(lvl, node) == null) {
                break;
            }
            maxLayer = lvl;
        }
        return maxLayer;
    }

    @Override
    public int size(int level) {
        return layers.get(level).size();
    }

    public void addNode(NodeAtLevel nodeLevel) {
        addNode(nodeLevel.level, nodeLevel.node);
    }

    public void addNode(int level, int node) {
        ensureLayersExist(level);

        // add the node to each layer
        for (int i = 0; i <= level; i++) {
            layers.get(i).addNode(node);
        }
        maxNodeId.accumulateAndGet(node, Math::max);
    }

    @Override
    public boolean contains(NodeAtLevel nodeLevel) {
        return contains(nodeLevel.level, nodeLevel.node);
    }

    @Override
    public boolean contains(int level, int node) {
        return layers.get(level).contains(node);
    }

    private void ensureLayersExist(int level) {
        for (int i = layers.size(); i <= level; i++) {
            synchronized (layers) {
                if (i == layers.size()) { // doublecheck after locking
                    var denseMap = layers.get(0);
                    var map = new ConcurrentNeighborMap(new SparseIntMap<>(),
                                                        denseMap.diversityProvider,
                                                        getDegree(level),
                                                        (int) (getDegree(level) * overflowRatio));
                    layers.add(map);
                }
            }
        }
    }

    public void connectNode(NodeAtLevel nodeLevel, NodeArray nodes) {
        connectNode(nodeLevel.level, nodeLevel.node, nodes);
    }

    public void connectNode(int level, int node, NodeArray nodes) {
        assert nodes != null;
        ensureLayersExist(level);
        this.layers.get(level).addNode(node, nodes);
        maxNodeId.accumulateAndGet(node, Math::max);
    }

    /**
     * Mark the given node deleted.  Does NOT remove the node from the graph.
     */
    public void markDeleted(int node) {
        deletedNodes.set(node);
    }

    public void markComplete(NodeAtLevel nodeLevel) {
        entryPoint.accumulateAndGet(
                nodeLevel,
                (oldEntry, newEntry) -> {
                    if (oldEntry == null || newEntry.level > oldEntry.level) {
                        return newEntry;
                    } else {
                        return oldEntry;
                    }
                });
        completions.markComplete(nodeLevel.node);
    }

    public void updateEntryNode(NodeAtLevel newEntry) {
        entryPoint.set(newEntry);
    }

    @Override
    public NodeAtLevel entryNode() {
        return entryPoint.get();
    }

    @Override
    public NodesIterator getNodes(int level) {
        return NodesIterator.fromPrimitiveIterator(nodeStream(level).iterator(),
                                                   layers.get(level).size());
    }

    @Override
    public IntStream nodeStream(int level) {
        var layer = layers.get(level);
        return level == 0
                ? IntStream.range(0, getIdUpperBound()).filter(i -> layer.get(i) != null)
                : ((SparseIntMap<Neighbors>) layer.neighbors).keysStream();
    }

    @Override
    public long ramBytesUsed() {
        var graphBytesUsed = IntStream.range(0, layers.size()).mapToLong(this::ramBytesUsedOneLayer).sum();
        return graphBytesUsed + completions.ramBytesUsed();
    }

    private long ramBytesUsedOneLayer(int level) {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        var REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        var AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long neighborSize = ramBytesUsedOneNode(level) * layers.get(level).size();
        return OH_BYTES + REF_BYTES * 2L + AH_BYTES + neighborSize;
    }

    public long ramBytesUsedOneNode(int level) {
        // we include the REF_BYTES for the CNS reference here to make it self-contained for addGraphNode()
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return REF_BYTES + Neighbors.ramBytesUsed(layers.get(level).nodeArrayLength());
    }

    @Override
    public void enforceDegree(int node) {
        for (int level = 0; level <= getMaxLevel(); level++) {
            layers.get(level).enforceDegree(node);
        }
    }

    @Override
    public void addEdges(int level, int node, NodeArray candidates, float overflowRatio) {
        var newNeighbors = layers.get(level).insertDiverse(node, candidates);
        layers.get(level).backlink(newNeighbors, node, overflowRatio);
    }

    @Override
    public void replaceDeletedNeighbors(int level, int node, BitSet toDelete, NodeArray candidates) {
        layers.get(level).replaceDeletedNeighbors(node, toDelete, candidates);
    }

    @Override
    public String toString() {
        return String.format("OnHeapGraphIndex(size=%d, entryPoint=%s)", size(0), entryPoint.get());
    }

    @Override
    public void close() {
        // No resources to close.
    }

    @Override
    public View getView() {
        // Before all completions are completed, we need a View that is thread-safe and allows concurrent mutations in the graph.
        // Once all completions are completed, we can freeze the graph and just need  a View that is thread-safe.
        if (allMutationsCompleted) {
            return new FrozenView();
        } else {
            return new ConcurrentGraphIndexView();
        }
    }

    public ThreadSafeGrowableBitSet getDeletedNodes() {
        return deletedNodes;
    }

    @Override
    public int removeNode(int node) {
        int found = 0;
        for (var layer : layers) {
            if (layer.remove(node) != null) {
                found++;
            }
        }
        deletedNodes.clear(node);
        return found;
    }

    @Override
    public int getIdUpperBound() {
        return maxNodeId.get() + 1;
    }

    @Override
    public boolean containsNode(int nodeId) {
        return layers.get(0).contains(nodeId);
    }

    /**
     * Returns the average degree computed over nodes in the specified layer.
     *
     * @param level the level of interest.
     * @return the average degree or NaN if no nodes are present.
     */
    public double getAverageDegree(int level) {
        return nodeStream(level)
                .mapToDouble(i -> getNeighbors(level, i).size())
                .average()
                .orElse(Double.NaN);
    }

    @Override
    public int getMaxLevel() {
        for (int lvl = 0; lvl < layers.size(); lvl++) {
            if (layers.get(lvl).size() == 0) {
                return lvl - 1;
            }
        }
        return layers.size() - 1;
    }

    @Override
    public int getDegree(int level) {
        if (level >= maxDegrees.size()) {
            return maxDegrees.get(maxDegrees.size() - 1);
        }
        return maxDegrees.get(level);
    }

    @Override
    public int maxDegree() {
        return maxDegrees.stream().mapToInt(i -> i).max().orElseThrow();
    }

    @Override
    public List<Integer> maxDegrees() {
        return maxDegrees;
    }

    @Override
    public void setDegrees(List<Integer> layerDegrees) {
        maxDegrees.clear();
        maxDegrees.addAll(layerDegrees);
    }

    @Override
    public void setAllMutationsCompleted() {
        allMutationsCompleted = true;
    }

    @Override
    public boolean allMutationsCompleted() {
        return allMutationsCompleted;
    }

    /**
     * A concurrent View of the graph that is safe to search concurrently with updates and with other
     * searches. The View provides a limited kind of snapshot isolation: only nodes completely added
     * to the graph at the time the View was created will be visible (but the connections between them
     * are allowed to change, so you could potentially get different top K results from the same query
     * if concurrent updates are in progress.)
     */
    public class ConcurrentGraphIndexView extends FrozenView {
        // It is tempting, but incorrect, to try to provide "adequate" isolation by
        // (1) keeping a bitset of complete nodes and giving that to the searcher as nodes to
        // accept -- but we need to keep incomplete nodes out of the search path entirely,
        // not just out of the result set, or
        // (2) keeping a bitset of complete nodes and restricting the View to those nodes
        // -- but we needs to consider neighbor diversity separately for concurrent
        // inserts and completed nodes; this allows us to keep the former out of the latter,
        // but not the latter out of the former (when a node completes while we are working,
        // that was in-progress when we started.)
        // The only really foolproof solution is to implement snapshot isolation as
        // we have done here.
        private final int timestamp = completions.clock();

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            NodesIterator it = OnHeapGraphIndex.this.getNeighborsIterator(level, node);

            return new NodesIterator() {
                int nextNode = advance();

                private int advance() {
                    while (it.hasNext()) {
                        int n = it.nextInt();
                        if (completions.completedAt(n) < timestamp) {
                            return n;
                        }
                    }
                    return Integer.MIN_VALUE;
                }

                @Override
                public int size() {
                    NodesIterator it = OnHeapGraphIndex.this.getNeighborsIterator(level, node);
                    int size = 0;
                    while (it.hasNext()) {
                        int n = it.nextInt();
                        if (completions.completedAt(n) < timestamp) {
                            size++;
                        }
                    }
                    return size;
                }

                @Override
                public int nextInt() {
                    int current = nextNode;
                    if (current == Integer.MIN_VALUE) {
                        throw new NoSuchElementException();
                    }
                    nextNode = advance();
                    return current;
                }

                @Override
                public boolean hasNext() {
                    return nextNode != Integer.MIN_VALUE;
                }
            };
        }
    }

    private class FrozenView implements View {
        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            return OnHeapGraphIndex.this.getNeighborsIterator(level, node);

        }

        @Override
        public int size() {
            return OnHeapGraphIndex.this.size(0);
        }

        @Override
        public NodeAtLevel entryNode() {
            return entryPoint.get();
        }

        @Override
        public Bits liveNodes() {
            // this Bits will return true for node ids that no longer exist in the graph after being purged,
            // but we defined the method contract so that that is okay
            return deletedNodes.cardinality() == 0 ? Bits.ALL : Bits.inverseOf(deletedNodes);
        }

        @Override
        public int getIdUpperBound() {
            return OnHeapGraphIndex.this.getIdUpperBound();
        }

        @Override
        public boolean contains(int level, int node) {
            return OnHeapGraphIndex.this.contains(level, node);
        }

        @Override
        public void close() {
            // No resources to close
        }

        @Override
        public String toString() {
            NodeAtLevel entry = entryNode();
            return String.format("%s(size=%d, entryNode=%s)", getClass().getSimpleName(), size(), entry);
        }
    }

    /**
     * Saves the graph to the given DataOutput for reloading into memory later
     */
    @Experimental
    @Deprecated
    public void save(DataOutput out) throws IOException {
        if (!allMutationsCompleted()) {
            throw new IllegalStateException("Cannot save a graph with pending mutations. Call cleanup() first");
        }

        out.writeInt(OnHeapGraphIndex.MAGIC); // the magic number
        out.writeInt(4); // The version

        // Write graph-level properties.
        out.writeInt(layers.size());
        for (int level = 0; level < layers.size(); level++) {
            out.writeInt(getDegree(level));
        }

        var entryNode = entryPoint.get();
        assert entryNode.level == getMaxLevel();
        out.writeInt(entryNode.node);

        for (int level = 0; level < layers.size(); level++) {
            out.writeInt(size(level));

            // Save neighbors from the layer.
            var it = nodeStream(level).iterator();
            while (it.hasNext()) {
                int nodeId = it.nextInt();
                var neighbors = layers.get(level).get(nodeId);
                out.writeInt(nodeId);
                out.writeInt(neighbors.size());

                for (int n = 0; n < neighbors.size(); n++) {
                    out.writeInt(neighbors.getNode(n));
                    out.writeFloat(neighbors.getScore(n));
                }
            }
        }
    }

    /**
     * Saves the graph to the given DataOutput for reloading into memory later
     */
    @Experimental
    @Deprecated
    public static OnHeapGraphIndex load(RandomAccessReader in, double overflowRatio, DiversityProvider diversityProvider) throws IOException {
        int magic = in.readInt(); // the magic number
        if (magic != OnHeapGraphIndex.MAGIC) {
            throw new IOException("Unsupported magic number: " + magic);
        }

        int version = in.readInt(); // The version
        if (version != 4) {
            throw new IOException("Unsupported version: " + version);
        }

        // Write graph-level properties.
        int layerCount = in.readInt();
        var layerDegrees = new ArrayList<Integer>(layerCount);
        for (int level = 0; level < layerCount; level++) {
            layerDegrees.add(in.readInt());
        }

        int entryNode = in.readInt();

        var graph = new OnHeapGraphIndex(layerDegrees, overflowRatio, diversityProvider);

        Map<Integer, Integer> nodeLevelMap = new HashMap<>();

        for (int level = 0; level < layerCount; level++) {
            int layerSize = in.readInt();

            for (int i = 0; i < layerSize; i++) {
                int nodeId = in.readInt();
                int nNeighbors = in.readInt();

                var ca = new NodeArray(nNeighbors);
                for (int j = 0; j < nNeighbors; j++) {
                    int neighbor = in.readInt();
                    float score = in.readFloat();
                    ca.addInOrder(neighbor, score);
                }
                graph.connectNode(level, nodeId, ca);
                nodeLevelMap.put(nodeId, level);
            }
        }

        for (var k : nodeLevelMap.keySet()) {
            NodeAtLevel nal = new NodeAtLevel(nodeLevelMap.get(k), k);
            graph.markComplete(nal);
        }

        graph.setDegrees(layerDegrees);
        graph.updateEntryNode(new NodeAtLevel(graph.getMaxLevel(), entryNode));

        return graph;
    }

    /**
     * A helper class that tracks completion times for nodes.
     */
    static final class CompletionTracker implements Accountable {
        private final AtomicInteger logicalClock = new AtomicInteger();
        private volatile AtomicIntegerArray completionTimes;
        private final StampedLock sl = new StampedLock();

        public CompletionTracker(int initialSize) {
            completionTimes = new AtomicIntegerArray(initialSize);
            for (int i = 0; i < initialSize; i++) {
                completionTimes.set(i, Integer.MAX_VALUE);
            }
        }

        void markComplete(int node) {
            int completionClock = logicalClock.getAndIncrement();
            ensureCapacity(node);
            long stamp;
            do {
                stamp = sl.tryOptimisticRead();
                completionTimes.set(node, completionClock);
            } while (!sl.validate(stamp));
        }

        int clock() {
            return logicalClock.get();
        }

        public int completedAt(int node) {
            AtomicIntegerArray ct = completionTimes;
            if (node >= ct.length()) {
                return Integer.MAX_VALUE;
            }
            return ct.get(node);
        }

        @Override
        public long ramBytesUsed() {
            int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
            return REF_BYTES + Integer.BYTES + REF_BYTES +
                    (long) Integer.BYTES * completionTimes.length();
        }

        private void ensureCapacity(int node) {
            if (node < completionTimes.length()) {
                return;
            }
            long stamp = sl.writeLock();
            try {
                AtomicIntegerArray oldArray = completionTimes;
                if (node >= oldArray.length()) {
                    int newSize = (node + 1) * 2;
                    AtomicIntegerArray newArr = new AtomicIntegerArray(newSize);
                    for (int i = 0; i < newSize; i++) {
                        if (i < oldArray.length()) {
                            newArr.set(i, oldArray.get(i));
                        } else {
                            newArr.set(i, Integer.MAX_VALUE);
                        }
                    }
                    completionTimes = newArr;
                }
            } finally {
                sl.unlockWrite(stamp);
            }
        }
    }
}
