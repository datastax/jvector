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

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.GrowableBitSet;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import org.jctools.maps.NonBlockingHashMapLong;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>To search this graph, you should use a View obtained from {@link #getView()} to perform `seek`
 * and `nextNeighbor` operations.
 */
public class OnHeapGraphIndex<T> implements GraphIndex<T>, Accountable {

  // the current graph entry node on the top level. -1 if not set
  private final AtomicLong entryPoint = new AtomicLong(-1);

  private final NonBlockingHashMapLong<ConcurrentNeighborSet> nodes;
  private final BitSet deletedNodes = new GrowableBitSet(0);
  private final AtomicInteger maxNodeId = new AtomicInteger(-1);

  // max neighbors/edges per node
  final int maxDegree;
  private final BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory;
  private boolean hasPurgedNodes;

  OnHeapGraphIndex(
      int M, BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory) {
    this.neighborFactory = neighborFactory;
    this.maxDegree = 2 * M;

    this.nodes = new NonBlockingHashMapLong<>(1024);
  }

  /**
   * Returns the neighbors connected to the given node, or null if the node does not exist.
   *
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  ConcurrentNeighborSet getNeighbors(int node) {
    return nodes.get(node);
  }

  @Override
  public int size() {
    return nodes.size();
  }

  /**
   * Add node on the given level with an empty set of neighbors.
   *
   * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
   * inserted out of order are eventually added.
   *
   * <p>Actually populating the neighbors, and establishing bidirectional links, is the
   * responsibility of the caller.
   *
   * <p>It is also the responsibility of the caller to ensure that each node is only added once.
   *
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int node) {
    nodes.put(node, neighborFactory.apply(node, maxDegree()));
    maxNodeId.accumulateAndGet(node, Math::max);
  }

  /**
   * Mark the given node deleted.  Does NOT remove the node from the graph.
   */
  public void markDeleted(int node) {
    deletedNodes.set(node);
  }

  /** must be called after addNode once neighbors are linked in all levels. */
  void markComplete(int node) {
    entryPoint.accumulateAndGet(
        node,
        (oldEntry, newEntry) -> {
          if (oldEntry >= 0) {
            return oldEntry;
          } else {
            return newEntry;
          }
        });
  }

  void updateEntryNode(int node) {
    entryPoint.set(node);
  }

  @Override
  public int maxDegree() {
    return maxDegree;
  }

  int entry() {
    return (int) entryPoint.get();
  }

  @Override
  public NodesIterator getNodes() {
    long[] keys = nodes.keySetLong();
    var keysInts = Arrays.stream(keys).mapToInt(i -> (int) i).iterator();
    return NodesIterator.fromPrimitiveIterator(keysInts, keys.length);
  }

  @Override
  public long ramBytesUsed() {
    // the main graph structure
    long total = concurrentHashMapRamUsed(size());
    long chmSize = concurrentHashMapRamUsed(size());
    long neighborSize = neighborsRamUsed(maxDegree()) * size();

    total += chmSize + neighborSize;

    return total;
  }

  long ramBytesUsedOneNode(int nodeLevel) {
    int entryCount = (int) (nodeLevel / CHM_LOAD_FACTOR);
    var graphBytesUsed =
        chmEntriesRamUsed(entryCount)
            + neighborsRamUsed(maxDegree())
            + nodeLevel * neighborsRamUsed(maxDegree());
    var clockBytesUsed = Integer.BYTES;
    return graphBytesUsed + clockBytesUsed;
  }

  private static long neighborsRamUsed(int count) {
    long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    long AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
    long neighborSetBytes =
        REF_BYTES // atomicreference
            + Integer.BYTES
            + Integer.BYTES
            + REF_BYTES // NeighborArray
            + AH_BYTES * 2 // NeighborArray internals
            + REF_BYTES * 2
            + Integer.BYTES
            + 1;
    return neighborSetBytes + (long) count * (Integer.BYTES + Float.BYTES);
  }

  private static final float CHM_LOAD_FACTOR = 0.75f; // this is hardcoded inside ConcurrentHashMap

  /**
   * caller's responsibility to divide number of entries by load factor to get internal node count
   */
  private static long chmEntriesRamUsed(int internalEntryCount) {
    long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    long chmNodeBytes =
        REF_BYTES // node itself in Node[]
            + 3L * REF_BYTES
            + Integer.BYTES; // node internals

    return internalEntryCount * chmNodeBytes;
  }

  private static long concurrentHashMapRamUsed(int externalSize) {
    long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    long AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
    long CORES = Runtime.getRuntime().availableProcessors();

    // CHM has a striped counter Cell implementation, we expect at most one per core
    long chmCounters = AH_BYTES + CORES * (REF_BYTES + Long.BYTES);

    int nodeCount = (int) (externalSize / CHM_LOAD_FACTOR);

    long chmSize =
        chmEntriesRamUsed(nodeCount) // nodes
            + nodeCount * REF_BYTES
            + AH_BYTES // nodes array
            + Long.BYTES
            + 3 * Integer.BYTES
            + 3 * REF_BYTES // extra internal fields
            + chmCounters
            + REF_BYTES; // the Map reference itself
    return chmSize;
  }

  @Override
  public String toString() {
    return String.format("OnHeapGraphIndex(size=%d, entryPoint=%d)", size(), entryPoint.get());
  }

  @Override
  public void close() {
    // no-op
  }

  /**
   * Returns a view of the graph that is safe to use concurrently with updates performed on the
   * underlying graph.
   *
   * <p>Multiple Views may be searched concurrently.
   */
  @Override
  public GraphIndex.View<T> getView() {
    return new ConcurrentGraphIndexView();
  }

  void validateEntryNode() {
    if (size() == 0) {
      return;
    }
    var en = entryPoint.get();
    if (!(en >= 0 && nodes.containsKey(en))) {
      throw new IllegalStateException("Entry node was incompletely added! " + en);
    }
  }

  public BitSet getDeletedNodes() {
    return deletedNodes;
  }

  void removeNode(int node) {
    nodes.remove(node);
    hasPurgedNodes = true;
  }

  @Override
  public int getMaxNodeId() {
    return maxNodeId.get();
  }

  int[] rawNodes() {
    return nodes.keySet().stream().mapToInt(i -> (int) (long) i).toArray();
  }

  public boolean containsNode(int nodeId) {
    return nodes.containsKey(nodeId);
  }

  public double getAverageShortEdges() {
    return IntStream.range(0, getMaxNodeId())
            .filter(this::containsNode)
            .mapToDouble(i -> getNeighbors(i).getShortEdges())
            .average()
            .orElse(Double.NaN);
  }

  private class ConcurrentGraphIndexView implements GraphIndex.View<T> {
    @Override
    public T getVector(int node) {
      throw new UnsupportedOperationException("All searches done with OnHeapGraphIndex should be exact");
    }

    public NodesIterator getNeighborsIterator(int node) {
      return getNeighbors(node).iterator();
    }

    @Override
    public int size() {
      return OnHeapGraphIndex.this.size();
    }

    @Override
    public int entryNode() {
      return (int) entryPoint.get();
    }

    @Override
    public String toString() {
      return "OnHeapGraphIndexView(size=" + size() + ", entryPoint=" + entryPoint.get();
    }

    @Override
    public Bits liveNodes() {
      // this Bits will return true for node ids that no longer exist in the graph after being purged,
      // but we defined the method contract so that that is okay
      return deletedNodes.cardinality() == 0 ? Bits.ALL : Bits.inverseOf(deletedNodes);
    }

    @Override
    public int getMaxNodeId() {
      return OnHeapGraphIndex.this.getMaxNodeId();
    }

    @Override
    public void close() {
      // no-op
    }
  }
}
