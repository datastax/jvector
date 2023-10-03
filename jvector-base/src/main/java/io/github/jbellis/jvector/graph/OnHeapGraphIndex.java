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
import io.github.jbellis.jvector.util.RamUsageEstimator;
import org.jctools.maps.NonBlockingHashMap;
import org.jctools.maps.NonBlockingHashMapLong;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLongFieldUpdater;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>To search this graph, you should use a View obtained from {@link #getView()} to perform `seek`
 * and `nextNeighbor` operations.
 */
public final class OnHeapGraphIndex<T> implements GraphIndex<T>, Accountable {
  private static final AtomicLongFieldUpdater<OnHeapGraphIndex> entryPointUpdater = AtomicLongFieldUpdater.newUpdater(OnHeapGraphIndex.class, "entryPointVal");

  // the current graph entry node on the top level. -1 if not set
  private volatile long entryPointVal;

  private final NonBlockingHashMapLong<ConcurrentNeighborSet> nodes;

  // max neighbors/edges per node
  final int nsize0;
  private final BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory;

  OnHeapGraphIndex(
      int M, BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory) {
    this.neighborFactory = neighborFactory;
    this.entryPointVal = -1; // Entry node should be negative until a node is added
    this.nsize0 = 2 * M;

    this.nodes = new NonBlockingHashMapLong<>(1024);
  }

  /**
   * Returns the neighbors connected to the given node.
   *
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public ConcurrentNeighborSet getNeighbors(int node) {
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
  }

  /** must be called after addNode once neighbors are linked in all levels. */
  void markComplete(int node) {
    entryPointUpdater.accumulateAndGet(this,
        node,
        (oldEntry, newEntry) -> {
          if (oldEntry >= 0) {
            return oldEntry;
          } else {
            return newEntry;
          }
        });
  }

  public void updateEntryNode(int node) {
    entryPointUpdater.set(this, node);
  }

  @Override
  public int maxDegree() {
    return nsize0;
  }

  int entry() {
    return (int) entryPointUpdater.get(this);
  }

  @Override
  public NodesIterator getNodes() {
    // We avoid the temptation to optimize this by using ArrayNodesIterator.
    // This is because, while the graph will contain sequential ordinals once the graph is complete,
    // we should not assume that that is the only time it will be called.
    var keysInts = Arrays.stream(nodes.keySetLong()).iterator();
    return new NodesIterator(nodes.size()) {
      @Override
      public int nextInt() {
        return keysInts.next().intValue();
      }

      @Override
      public boolean hasNext() {
        return keysInts.hasNext();
      }
    };
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

  public long ramBytesUsedOneNode(int nodeLevel) {
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
    return String.format("OnHeapGraphIndex(size=%d, entryPoint=%d)", size(), entryPointUpdater.get(this));
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
  public GraphIndex.View<T> getView() {
    return new ConcurrentGraphIndexView();
  }

  void validateEntryNode() {
    if (size() == 0) {
      return;
    }
    var en = entryPointUpdater.get(this);
    if (!(en >= 0 && nodes.containsKey(en))) {
      throw new IllegalStateException("Entry node was incompletely added! " + en);
    }
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
      return (int) entryPointUpdater.get(OnHeapGraphIndex.this);
    }

    @Override
    public String toString() {
      return "OnHeapGraphIndexView(size=" + size() + ", entryPoint=" + entryPointUpdater.get(OnHeapGraphIndex.this);
    }

    @Override
    public void close() {
      // no-op
    }
  }
}
