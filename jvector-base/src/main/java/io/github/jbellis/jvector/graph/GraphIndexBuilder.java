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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.util.AtomicFixedBitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntArrayList;
import org.agrona.collections.IntArrayQueue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.OnHeapGraphIndex.NO_ENTRY_POINT;
import static io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;
import static io.github.jbellis.jvector.vector.VectorUtil.dotProduct;

/**
 * Builder for Concurrent GraphIndex. See {@link GraphIndex} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 * <p>
 * GIB allocates scratch space and copies of the RandomAccessVectorValues for each thread
 * that calls `addGraphNode`.  These allocations are retained until the GIB itself is no longer referenced.
 * Under most conditions this is not something you need to worry about, but it does mean
 * that spawning a new Thread per call is not advisable.  This includes virtual threads.
 */
public class GraphIndexBuilder {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int beamWidth;
    private final ExplicitThreadLocal<NodeArray> naturalScratch;
    private final ExplicitThreadLocal<NodeArray> concurrentScratch;

    private final VectorSimilarityFunction similarityFunction;
    private final float neighborOverflow;
    private final float alpha;

    @VisibleForTesting
    final OnHeapGraphIndex graph;
    private final ConcurrentSkipListSet<Integer> insertionsInProgress = new ConcurrentSkipListSet<>();

    // We need two sources of vectors in order to perform diversity check comparisons without
    // colliding.  Usually it's obvious because you can see the different sources being used
    // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
    // and `vectorsCopy` later on when defining the ScoreFunction for search.
    private final Supplier<RandomAccessVectorValues> vectors;
    private final Supplier<RandomAccessVectorValues> vectorsCopy;
    private final int dimension; // for convenience so we don't have to go to the pool for this
    private final NodeSimilarity similarity;

    private final ForkJoinPool simdExecutor;
    private final ForkJoinPool parallelExecutor;

    private final AtomicInteger updateEntryNodeIn = new AtomicInteger(10_000);

    /**
     * Reads all the vectors from vector values, builds a graph connecting them by their dense
     * ordinals, using the given hyperparameter settings, and returns the resulting graph.
     *
     * @param vectorValues     the vectors whose relations are represented by the graph - must provide a
     *                         different view over those vectors than the one used to add via addGraphNode.
     * @param M                – the maximum number of connections a node can have
     * @param beamWidth        the size of the beam search to use when finding nearest neighbors.
     * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
     *                         node. larger values will build more efficiently, but use more memory.
     * @param alpha            how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
     *                         allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
     *                         an HNSW graph will be created, which is usually not what you want.
     */
    public GraphIndexBuilder(
            RandomAccessVectorValues vectorValues,
            VectorSimilarityFunction similarityFunction,
            int M,
            int beamWidth,
            float neighborOverflow,
            float alpha) {
        this(vectorValues, similarityFunction, M, beamWidth, neighborOverflow, alpha,
                PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Reads all the vectors from vector values, builds a graph connecting them by their dense
     * ordinals, using the given hyperparameter settings, and returns the resulting graph.
     *
     * @param vectorValues     the vectors whose relations are represented by the graph - must provide a
     *                         different view over those vectors than the one used to add via addGraphNode.
     * @param M                – the maximum number of connections a node can have
     * @param beamWidth        the size of the beam search to use when finding nearest neighbors.
     * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
     *                         node. larger values will build more efficiently, but use more memory.
     * @param alpha            how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
     *                         allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
     *                         an HNSW graph will be created, which is usually not what you want.
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public GraphIndexBuilder(
            RandomAccessVectorValues vectorValues,
            VectorSimilarityFunction similarityFunction,
            int M,
            int beamWidth,
            float neighborOverflow,
            float alpha,
            ForkJoinPool simdExecutor,
            ForkJoinPool parallelExecutor) {
        vectors = vectorValues.threadLocalSupplier();
        vectorsCopy = vectorValues.threadLocalSupplier();
        dimension = vectorValues.dimension();
        this.similarityFunction = Objects.requireNonNull(similarityFunction);
        this.neighborOverflow = neighborOverflow;
        this.alpha = alpha;
        if (M <= 0) {
            throw new IllegalArgumentException("maxConn must be positive");
        }
        if (beamWidth <= 0) {
            throw new IllegalArgumentException("beamWidth must be positive");
        }
        this.beamWidth = beamWidth;
        this.simdExecutor = simdExecutor;
        this.parallelExecutor = parallelExecutor;

        similarity = node1 -> {
            var v = vectors.get();
            var vc = vectorsCopy.get();
            VectorFloat<?> v1 = v.vectorValue(node1);
            return (NodeSimilarity.ExactScoreFunction) node2 -> similarityFunction.compare(v1, vc.vectorValue(node2));
        };
        this.graph = new OnHeapGraphIndex(M, (node, m) -> new ConcurrentNeighborSet(node, m, similarity, alpha));

        // in scratch we store candidates in reverse order: worse candidates are first
        this.naturalScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
        this.concurrentScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
    }

    /**
     * Convenience method for calling addGraphNode + cleanup.  addGraphNode is invoked concurrently.
     */
    public OnHeapGraphIndex build() {
        var v = vectors.get();
        int size = v.size();

        simdExecutor.submit(() -> {
            IntStream.range(0, size).parallel().forEach(i -> {
                addGraphNode(i, vectors.get());
            });
        }).join();

        cleanup();
        return graph;
    }

    /**
     * Cleanup the graph by completing removal of marked-for-delete nodes, trimming
     * neighbor sets to the advertised degree, and updating the entry node.
     * <p>
     * Uses default threadpool to process nodes in parallel.  There is currently no way to restrict this to a single thread.
     * <p>
     * Must be called before writing to disk.
     * <p>
     * May be called multiple times, but should not be called during concurrent modifications to the graph.
     */
    public void cleanup() {
        if (graph.size() == 0) {
            return;
        }
        graph.validateEntryNode(); // sanity check before we start

        // purge deleted nodes.
        // backlinks can cause neighbors to soft-overflow, so do this before neighbors cleanup
        removeDeletedNodes();

        if (graph.size() == 0) {
            // After removing all the deleted nodes, we might end up with an empty graph.
            // The calls below expect a valid entry node, but we do not have one right now.
            return;
        }

        // clean up overflowed neighbor lists
        parallelExecutor.submit(() -> IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
            var neighbors = graph.getNeighbors(i);
            if (neighbors != null) {
                neighbors.enforceDegree();
            }
        })).join();

        // reconnect any orphaned nodes.  this will maintain neighbors size
        reconnectOrphanedNodes();

        // optimize entry node
        updateEntryPoint();
    }

    private void reconnectOrphanedNodes() {
        var searchPathNeighbors = new ConcurrentHashMap<Integer, NodeArray>();
        // It's possible that reconnecting one node will result in disconnecting another, since we are maintaining
        // the maxConnections invariant.  In an extreme case, reconnecting node X disconnects Y, and reconnecting
        // Y disconnects X again.  So we do a best effort of 3 loops.
        for (int i = 0; i < 3; i++) {
            // find all nodes reachable from the entry node
            var connectedNodes = new AtomicFixedBitSet(graph.getIdUpperBound());
            connectedNodes.set(graph.entry());
            var entryNeighbors = graph.getNeighbors(graph.entry()).getCurrent();
            parallelExecutor.submit(() -> IntStream.range(0, entryNeighbors.size).parallel().forEach(node -> findConnected(connectedNodes, entryNeighbors.node[node]))).join();

            // reconnect unreachable nodes
            var nReconnected = new AtomicInteger();
            var connectionTargets = ConcurrentHashMap.<Integer>newKeySet();
            simdExecutor.submit(() -> IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(node -> {
                if (connectedNodes.get(node) || !graph.containsNode(node)) {
                    return;
                }
                nReconnected.incrementAndGet();

                // first, attempt to connect one of our own neighbors to us
                var neighbors = graph.getNeighbors(node).getCurrent();
                if (connectToClosestNeighbor(node, neighbors, connectionTargets)) {
                    return;
                }

                // no unused candidate found -- search for more neighbors and try again
                neighbors = searchPathNeighbors.get(node);
                if (neighbors == null) {
                    SearchResult result;
                    try (var gs = createSearcher()) {
                        var v1 = vectors.get();
                        var v2 = vectorsCopy.get();

                        var notSelfBits = createNotSelfBits(node);
                        var value = v1.vectorValue(node);
                        NodeSimilarity.ExactScoreFunction scoreFunction = i1 -> similarityFunction.compare(v2.vectorValue(i1), value);
                        int ep = graph.entry();
                        result = gs.searchInternal(scoreFunction, null, beamWidth, 0.0f, 0.0f, ep, notSelfBits);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    neighbors = new NodeArray(result.getNodes().length);
                    toScratchCandidates(result.getNodes(), neighbors);
                    searchPathNeighbors.put(node, neighbors);
                }
                connectToClosestNeighbor(node, neighbors, connectionTargets);
            })).join();
            if (nReconnected.get() == 0) {
                break;
            }
        }
    }

    /**
     * Connect `node` to the closest neighbor that is not already a connection target.
     * @return true if such a neighbor was found.
     */
    private boolean connectToClosestNeighbor(int node, NodeArray neighbors, Set<Integer> connectionTargets) {
        // connect this node to the closest neighbor that hasn't already been used as a connection target
        // (since this edge is likely to be the "worst" one in that target's neighborhood, it's likely to be
        // overwritten by the next node to need reconnection if we don't choose a unique target)
        for (int i = 0; i < neighbors.size; i++) {
            var neighborNode = neighbors.node[i];
            var neighborScore = neighbors.score[i];
            if (connectionTargets.add(neighborNode)) {
                graph.getNeighbors(neighborNode).insertNotDiverse(node, neighborScore);
                return true;
            }
        }
        return false;
    }

    private void findConnected(AtomicFixedBitSet connectedNodes, int start) {
        var queue = new IntArrayQueue();
        queue.add(start);
        try (var view = graph.getView()) {
            while (!queue.isEmpty()) {
                // DFS should result in less contention across findConnected threads than BFS
                int next = queue.pollInt();
                if (connectedNodes.getAndSet(next)) {
                    continue;
                }
                for (var it = view.getNeighborsIterator(next); it.hasNext(); ) {
                    queue.addInt(it.nextInt());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public OnHeapGraphIndex getGraph() {
        return graph;
    }

    /**
     * Number of inserts in progress, across all threads.
     */
    public int insertsInProgress() {
        return insertionsInProgress.size();
    }

    /**
     * Inserts a node with the given vector value to the graph.
     *
     * <p>To allow correctness under concurrency, we track in-progress updates in a
     * ConcurrentSkipListSet. After adding ourselves, we take a snapshot of this set, and consider all
     * other in-progress updates as neighbor candidates.
     *
     * @param node    the node ID to add
     * @param vectors the set of vectors
     * @return an estimate of the number of extra bytes used by the graph after adding the given node
     */
    public long addGraphNode(int node, RandomAccessVectorValues vectors) {
        final VectorFloat<?> value = vectors.vectorValue(node);

        // do this before adding to in-progress, so a concurrent writer checking
        // the in-progress set doesn't have to worry about uninitialized neighbor sets
        var newNodeNeighbors = graph.addNode(node);

        insertionsInProgress.add(node);
        ConcurrentSkipListSet<Integer> inProgressBefore = insertionsInProgress.clone();
        try (var gs = createSearcher()) {
            var vc = vectorsCopy.get();
            var naturalScratchPooled = naturalScratch.get();
            var concurrentScratchPooled = concurrentScratch.get();
            // find ANN of the new node by searching the graph
            int ep = graph.entry();
            NodeSimilarity.ExactScoreFunction scoreFunction = i -> similarityFunction.compare(vc.vectorValue(i), value);

            var bits = new ExcludingBits(node);
            // find best "natural" candidates with a beam search
            var result = gs.searchInternal(scoreFunction, null, beamWidth, 0.0f, 0.0f, ep, bits);

            // Update neighbors with these candidates.
            // The DiskANN paper calls for using the entire set of visited nodes along the search path as
            // potential candidates, but in practice we observe neighbor lists being completely filled using
            // just the topK results.  (Since the Robust Prune algorithm prioritizes closer neighbors,
            // this means that considering additional nodes from the search path, that are by definition
            // farther away than the ones in the topK, would not change the result.)
            // TODO if we made NeighborArray an interface we could wrap the NodeScore[] directly instead of copying
            var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
            var concurrent = getConcurrentCandidates(node, inProgressBefore, concurrentScratchPooled, vectors, vc);
            newNodeNeighbors.insertDiverse(natural, concurrent);
            newNodeNeighbors.backlink(graph::getNeighbors, neighborOverflow);

            maybeUpdateEntryPoint(node);
            maybeImproveOlderNode();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            insertionsInProgress.remove(node);
        }

        return graph.ramBytesUsedOneNode();
    }

    /**
     * Improve edge quality on very low-d indexes.  This makes a big difference
     * in the ability of search to escape local maxima to find better options.
     * <p>
     * This has negligible effect on ML embedding-sized vectors, starting at least with GloVe-25, so we don't bother.
     * (Dimensions between 4 and 25 are untested but they get left out too.)
     * For 2D vectors, this takes us to over 99% recall up to at least 4M nodes.  (Higher counts untested.)
    */
    private void maybeImproveOlderNode() {
        // pick a node added earlier at random to improve its connections
        // 20k threshold chosen because that's where recall starts to fall off from 100% for 2D vectors
        if (dimension <= 3 && graph.size() > 20_000) {
            // if we can't find a candidate in 3 tries, the graph is too sparse,
            // we'll have to wait for more nodes to be added (this threshold has been tested w/ parallel build,
            // which generates very sparse ids due to how spliterator works)
            for (int i = 0; i < 3; i++) {
                var olderNode = ThreadLocalRandom.current().nextInt(graph.size());
                if (graph.containsNode(olderNode) && !graph.getDeletedNodes().get(olderNode)) {
                    improveConnections(olderNode);
                    break;
                }
            }
        }
    }

    private void maybeUpdateEntryPoint(int node) {
        graph.maybeSetInitialEntryNode(node); // TODO it seems silly to call this long after we've set it the first time

        if (updateEntryNodeIn.decrementAndGet() == 0) {
            updateEntryPoint();
        }
    }

    private void updateEntryPoint() {
        int newEntryNode = approximateMedioid();
        graph.updateEntryNode(newEntryNode);
        if (newEntryNode >= 0) {
            improveConnections(newEntryNode);
            updateEntryNodeIn.addAndGet(graph.size());
        } else {
            updateEntryNodeIn.addAndGet(10_000);
        }
    }

    public void improveConnections(int node) {
        var pv = vectors.get();
        NodeArray naturalScratchPooled;
        SearchResult result;
        try (var gs = createSearcher()) {
            var vc = vectorsCopy.get();
            naturalScratchPooled = naturalScratch.get();
            final VectorFloat<?> value = pv.vectorValue(node);
            int ep = graph.entry();
            NodeSimilarity.ExactScoreFunction scoreFunction = i -> similarityFunction.compare(vc.vectorValue(i), value);
            var bits = new ExcludingBits(node);
            result = gs.searchInternal(scoreFunction, null, beamWidth, 0.0f, 0.0f, ep, bits);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
        ConcurrentNeighborSet neighbors = graph.getNeighbors(node);
        neighbors.insertDiverse(natural, NodeArray.EMPTY);
        // no overflow -- this method gets called from cleanup
        neighbors.backlink(graph::getNeighbors, 1.0f);
    }

    public void markNodeDeleted(int node) {
        graph.markDeleted(node);
    }

    /**
     * Remove nodes marked for deletion from the graph, and update neighbor lists
     * to maintain connectivity.  Threadsafe with respect to inserts.
     *
     * @return approximate size of memory no longer used
     */
    public synchronized long removeDeletedNodes() {
        // Take a snapshot of the nodes to delete
        var toDelete = graph.getDeletedNodes().copy();
        var nRemoved = toDelete.cardinality();
        if (nRemoved == 0) {
            return 0;
        }
        // make a list of remaining live nodes
        var liveNodes = new IntArrayList();
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (graph.containsNode(i) && !toDelete.get(i)) {
                liveNodes.add(i);
            }
        }

        // wait for any in-progress insertions to complete -- after this, we are guaranteed
        // that any new insertions will see all the nodes in toDelete as deleted (and will
        // not add connections to them)
        busyWaitForInProgressInserts();

        // Compute new edges to insert.  If node j is deleted, we add edges (i, k)
        // whenever (i, j) and (j, k) are directed edges in the current graph.  This
        // strategy is proposed in "FreshDiskANN: A Fast and Accurate Graph-Based
        // ANN Index for Streaming Similarity Search" section 4.2.
        var newEdges = new ConcurrentHashMap<Integer, Set<Integer>>(); // new edges for key k are values v
        parallelExecutor.submit(() -> {
            IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
                var neighbors = graph.getNeighbors(i);
                if (neighbors == null || toDelete.get(i)) {
                    return;
                }
                for (var it = neighbors.iterator(); it.hasNext(); ) {
                    var j = it.nextInt();
                    if (toDelete.get(j)) {
                        var newEdgesForI = newEdges.computeIfAbsent(i, __ -> ConcurrentHashMap.newKeySet());
                        for (var jt = graph.getNeighbors(j).iterator(); jt.hasNext(); ) {
                            int k = jt.nextInt();
                            if (i != k && !toDelete.get(k)) {
                                newEdgesForI.add(k);
                            }
                        }
                    }
                }
            });
        }).join();

        // Remove deleted nodes from neighbors lists;
        // Score the new edges, and connect the most diverse ones as neighbors
        parallelExecutor.submit(() -> {
            newEdges.entrySet().stream().parallel().forEach(e -> {
                // turn the new edges into a NodeArray
                int node = e.getKey();
                var v1 = vectors.get().vectorValue(node);
                var neighbors = graph.getNeighbors(node);
                var candidates = new NodeArray(graph.maxDegree);
                for (var k : e.getValue()) {
                    candidates.insertSorted(k, similarityFunction.compare(v1, vectorsCopy.get().vectorValue(k)));
                }

                // it's unlikely, but possible, that all the potential replacement edges were to nodes that have also
                // been deleted.  if that happens, keep the graph connected by adding random edges.
                // (this is overly conservative -- really what we care about is that the end result of
                // replaceDeletedNeighbors not be empty -- but we want to avoid having the node temporarily
                // neighborless while concurrent searches run.  empirically, this only results in a little extra work.)
                if (candidates.size() == 0) {
                    var R = ThreadLocalRandom.current();
                    var vc = vectorsCopy.get();
                    // doing actual sampling-without-replacement is expensive so we'll loop a fixed number of times instead
                    for (int i = 0; i < 2 * graph.maxDegree(); i++) {
                        int randomNode = liveNodes.get(R.nextInt(liveNodes.size()));
                        if (randomNode != node && !candidates.contains(randomNode)) {
                            float score = similarityFunction.compare(v1, vc.vectorValue(randomNode));
                            candidates.insertSorted(randomNode, score);
                        }
                        if (candidates.size == graph.maxDegree) {
                            break;
                        }
                    }
                }

                // remove edges to deleted nodes and add the new connections, maintaining diversity
                neighbors.replaceDeletedNeighbors(toDelete, candidates);
            });
        }).join();

        // Generally we want to keep entryPoint update and node removal distinct, because both can be expensive,
        // but if the entry point was deleted then we have no choice
        if (toDelete.get(graph.entry())) {
            updateEntryPoint();
        }

        // wait for in-progress searches (including for concurrent inserts) to complete
        // before removing the deleted nodes, so that we don't invalidate their candidate queues
        var view = graph.getView();
        view.close();
        graph.waitForViewsBefore(view.stamp);

        // Remove the deleted nodes from the graph
        assert toDelete.cardinality() == nRemoved : "cardinality changed";
        for (int i = toDelete.nextSetBit(0); i != NO_MORE_DOCS; i = toDelete.nextSetBit(i + 1)) {
            graph.removeNode(i);
        }

        return nRemoved * graph.ramBytesUsedOneNode();
    }

    @SuppressWarnings("BusyWait")
    private void busyWaitForInProgressInserts() {
        var inProgress = insertionsInProgress.clone();
        while (!inProgress.isEmpty()) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            inProgress.retainAll(insertionsInProgress);
        }
    }

    private static Bits createNotSelfBits(int node) {
        return new Bits() {
            @Override
            public boolean get(int index) {
                return index != node;
            }

            @Override
            public int length() {
                // length is max node id, which could be larger than size after deletes
                throw new UnsupportedOperationException();
            }
        };
    }

    /**
     * Returns the ordinal of the node that is closest to the centroid of the graph,
     * or NO_ENTRY_POINT if there are no live nodes in the graph.
     */
    private int approximateMedioid() {
        if (graph.size() == 0) {
            return NO_ENTRY_POINT;
        }

        SearchResult result;
        try (var gs = createSearcher()) {
            var vc = vectorsCopy.get();
            var centroid = vectorTypeSupport.createFloatVector(dimension);
            for (var it = graph.getNodes(); it.hasNext(); ) {
                var node = it.nextInt();
                VectorUtil.addInPlace(centroid, vc.vectorValue(node));
            }
            VectorUtil.scale(centroid, 1.0f / graph.size());

            // if the centroid is the zero vector, we can't use cosine similarity in our search
            VectorSimilarityFunction sf;
            if (similarityFunction == VectorSimilarityFunction.COSINE) {
                sf = dotProduct(centroid, centroid) < 1E-6 ? VectorSimilarityFunction.EUCLIDEAN : similarityFunction;
            } else {
                sf = similarityFunction;
            }
            NodeSimilarity.ExactScoreFunction scoreFunction = i -> sf.compare(vc.vectorValue(i), centroid);
            int ep = graph.entry();
            result = gs.searchInternal(scoreFunction, null, beamWidth, 0.0f, 0.0f, ep, Bits.ALL);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return result.getNodes().length > 0 ? result.getNodes()[0].node : NO_ENTRY_POINT;
    }

    private GraphSearcher createSearcher() {
        return new GraphSearcher.Builder(graph.getView()).withConcurrentUpdates().build();
    }

    private static NodeArray toScratchCandidates(SearchResult.NodeScore[] candidates, NodeArray scratch) {
        scratch.clear();
        for (var candidate : candidates) {
            scratch.addInOrder(candidate.node, candidate.score);
        }
        return scratch;
    }

    private NodeArray getConcurrentCandidates(int newNode,
                                              Set<Integer> inProgress,
                                              NodeArray scratch,
                                              RandomAccessVectorValues values,
                                              RandomAccessVectorValues valuesCopy)
    {
        scratch.clear();
        for (var n : inProgress) {
            if (n == newNode) {
                continue;
            }
            scratch.insertSorted(n, similarityFunction.compare(values.vectorValue(newNode), valuesCopy.vectorValue(n)));
        }
        return scratch;
    }

    @VisibleForTesting
    int randomLiveNode() {
        var R = ThreadLocalRandom.current();

        // first, try doing it quickly by just picking a random node
        for (int i = 0; i < 3; i++) {
            var idUpperBound = graph.getIdUpperBound();
            if (idUpperBound == 0) {
                return NO_ENTRY_POINT;
            }
            int n = R.nextInt(idUpperBound);
            if (graph.containsNode(n) && !graph.getDeletedNodes().get(n)) {
                return n;
            }
        }

        // lots of deletions and/or sparse node ids, so we do it the slow way
        var L = new ArrayList<Integer>();
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (graph.containsNode(i) && !graph.getDeletedNodes().get(i)) {
                L.add(i);
            }
        }
        if (L.isEmpty()) {
            return NO_ENTRY_POINT;
        }
        return L.get(R.nextInt(L.size()));
    }

    @VisibleForTesting
    void validateAllNodesLive() {
        assert graph.getDeletedNodes().cardinality() == 0;
        // all edges should be valid, live nodes
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (!graph.containsNode(i)) {
                continue; // holes are tolerated
            }
            var neighbors = graph.getNeighbors(i);
            for (var it = neighbors.iterator(); it.hasNext(); ) {
                var j = it.nextInt();
                assert graph.containsNode(j) : String.format("Edge %d -> %d is invalid", i, j);
            }
        }
    }

    private static class ExcludingBits implements Bits {
        private final int excluded;

        public ExcludingBits(int excluded) {
            this.excluded = excluded;
        }

        @Override
        public boolean get(int index) {
            return index != excluded;
        }

        @Override
        public int length() {
            throw new UnsupportedOperationException();
        }
    }

    public void load(RandomAccessReader in) throws IOException {
        if (graph.size() != 0) {
            throw new IllegalStateException("Cannot load into a non-empty graph");
        }

        int size = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();

        for (int i = 0; i < size; i++) {
            int node = in.readInt();
            int nNeighbors = in.readInt();
            var ca = new NodeArray(maxDegree);
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = in.readInt();
                ca.addInOrder(neighbor, similarity.score(node, neighbor));
            }
            graph.addNode(node, new ConcurrentNeighborSet(node, maxDegree, similarity, alpha, ca));
        }

        graph.updateEntryNode(entryNode);
    }
}
