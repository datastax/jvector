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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.index.Index;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * A graph/HNSW index that is still being built: nodes are added one at a time (from any number of
 * threads), may be marked deleted, and the graph can be searched while that is happening. Obtain one
 * from {@link HnswIndexBuilder#buildMutable()}.
 * <p>
 * This is the incremental counterpart of {@link HnswIndexBuilder#build()}, for callers that don't
 * have every vector up front &mdash; e.g. an in-memory index that grows as rows are written, or a
 * compaction that streams vectors in &mdash; and wraps the {@link GraphIndexBuilder} such callers used
 * directly before. A typical lifecycle:
 * <pre>{@code
 * try (MutableHnswIndex index = Indexes.hnswBuilder()
 *         .withScoreProvider(bsp).withDimension(dim)
 *         .withMaxDegree(32).withBeamWidth(100).withNeighborOverflow(1.2f).withAlpha(1.2f)
 *         .withAddHierarchy(true)
 *         .buildMutable()) {
 *     index.addNode(ordinal, vector);     // concurrently, as vectors arrive
 *     index.markDeleted(otherOrdinal);
 *     ...                                 // index.searcher() may be used throughout
 *     index.cleanup();                    // once, after the last addNode, before writing
 *     index.graph().getWriterBuilder(path)...
 * }
 * }</pre>
 * <p>
 * Thread safety: {@link #addNode}, {@link #markDeleted} and searches may all run concurrently.
 * {@link #cleanup}, {@link #removeDeletedNodes} and {@link #rescore} are exclusive: they wait for
 * in-flight {@code addNode}/{@code markDeleted} calls to finish and block new ones until they
 * complete. Searches are not blocked by any of these. Inserts may be issued from any threads,
 * including the workers of the ForkJoinPools passed to the builder: an insert waiting for an
 * exclusive operation blocks cooperatively, so the pool can still run that operation's parallel work.
 */
public final class MutableHnswIndex implements Index {
    // Read lock: operations that may run concurrently with each other (addNode, markDeleted).
    // Write lock: operations GraphIndexBuilder documents as unsafe during concurrent modification
    // (cleanup, removeDeletedNodes), and rescore, which replaces the builder.
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private volatile GraphIndexBuilder builder;

    MutableHnswIndex(GraphIndexBuilder builder) {
        this.builder = builder;
    }

    /**
     * Inserts a node with the given ordinal and vector. Safe to call from multiple threads at once,
     * and while the graph is being searched. Ordinals need not be dense or arrive in order, but each
     * may only be added once.
     *
     * @return an estimate of the number of heap bytes the graph grew by
     */
    public long addNode(int ordinal, VectorFloat<?> vector) {
        lockForInsert();
        try {
            return builder.addGraphNode(ordinal, vector);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Marks a node as deleted. It is excluded from search results immediately, and physically
     * removed (with its neighbors reconnected) by the next {@link #removeDeletedNodes} or
     * {@link #cleanup}. Safe to call concurrently with {@link #addNode}.
     */
    public void markDeleted(int ordinal) {
        lockForInsert();
        try {
            builder.markNodeDeleted(ordinal);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Removes nodes marked deleted and repairs their neighbors' connections. Blocks concurrent
     * {@link #addNode}/{@link #markDeleted} calls while it runs. {@link #cleanup} does this too, so
     * calling it separately is only needed to reclaim memory before cleanup.
     *
     * @return approximate number of heap bytes freed
     */
    public long removeDeletedNodes() {
        lock.writeLock().lock();
        try {
            return builder.removeDeletedNodes();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Finishes construction: removes deleted nodes, optionally refines connections (see
     * {@link HnswIndexBuilder#withRefineFinalGraph}), and trims every neighbor list to the max
     * degree. Must be called before the graph is written to disk. Blocks concurrent
     * {@link #addNode}/{@link #markDeleted} calls while it runs; nodes may still be added afterwards,
     * but then {@code cleanup()} must be called again before writing.
     * <p>
     * Cleanup freezes the graph: searchers created after it use a cheaper view that assumes no
     * further mutation. The next {@link #addNode} unfreezes the graph, so create new searchers after
     * adding more nodes rather than reusing ones from the frozen period.
     */
    public void cleanup() {
        lock.writeLock().lock();
        try {
            builder.cleanup();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Switches construction to a new score provider, keeping every existing edge but recomputing its
     * score with the new provider. The typical use is swapping in a refined PQ codebook partway
     * through a build.
     * <p>
     * {@code newScoreProvider} is invoked with all {@link #addNode}/{@link #markDeleted} calls locked
     * out, so it may safely replace whatever state the old provider reads (e.g. re-encode the
     * compressed vectors added so far) before returning the provider that reads the new state; no
     * insert can observe the old provider with half-updated state. Searches are not locked out, and
     * searchers created before this call keep searching the pre-rescore graph: afterwards
     * {@link #graph()} returns a new graph instance, and existing searchers should be replaced.
     */
    public void rescore(Supplier<BuildScoreProvider> newScoreProvider) {
        lock.writeLock().lock();
        try {
            GraphIndexBuilder old = builder;
            builder = GraphIndexBuilder.rescore(old, newScoreProvider.get());
            closeQuietly(old);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Convenience for {@link #rescore(Supplier)} when no other state needs to change under the lock. */
    public void rescore(BuildScoreProvider newScoreProvider) {
        rescore(() -> newScoreProvider);
    }

    /**
     * The graph under construction. It can be searched at any time, including concurrently with
     * {@link #addNode}, and written to disk (via {@link PersistableGraphIndex}'s writer builders) after
     * {@link #cleanup}. {@link #rescore} replaces it, so don't cache it across a rescore.
     */
    public PersistableGraphIndex graph() {
        return builder.graph;
    }

    /** A searcher over the current {@link #graph()}. */
    @Override
    public GraphSearcher searcher() {
        return graph().searcher();
    }

    /**
     * Number of {@link #addNode} calls in progress across all threads. For sanity checks only; it
     * cannot be used to prevent races.
     */
    public int insertsInProgress() {
        return builder.insertsInProgress();
    }

    @Override
    public long ramBytesUsed() {
        return builder.ramBytesUsed();
    }

    /**
     * Releases per-thread construction scratch space. The graph returned by {@link #graph()} stays
     * usable after this.
     */
    @Override
    public void close() throws IOException {
        builder.close();
    }

    /**
     * Adds ordinals {@code [from, vectors.size())} in parallel on {@code executor}, then cleans up.
     * The batch path shared by {@link HnswIndexBuilder#build()}.
     */
    void addAllAndCleanup(RandomAccessVectorValues vectors, int from, ForkJoinPool executor) {
        var vv = vectors.threadLocalSupplier();
        executor.submit(() -> IntStream.range(from, vectors.size()).parallel()
                .forEach(node -> addNode(node, vv.get().getVector(node)))).join();
        cleanup();
    }

    /**
     * Acquires the read lock, cooperating with {@link ForkJoinPool}s while waiting. The exclusive
     * operations hold the write lock while running parallel work on the builder's executors (and, for
     * {@link #rescore}, whatever the caller's supplier runs, e.g. {@code ProductQuantization.refine}).
     * If callers insert from a pool's own workers and those workers simply parked on the lock, the
     * exclusive operation's tasks could have no worker left to run on and deadlock. managedBlock tells
     * the pool the worker is blocked, so it can start a spare to run those tasks; outside a pool it
     * behaves like a plain lock().
     */
    private void lockForInsert() {
        Lock readLock = lock.readLock();
        if (readLock.tryLock()) {
            return;
        }
        try {
            ForkJoinPool.managedBlock(new ForkJoinPool.ManagedBlocker() {
                private boolean acquired;

                @Override
                public boolean block() {
                    readLock.lock();
                    acquired = true;
                    return true;
                }

                @Override
                public boolean isReleasable() {
                    return acquired || (acquired = readLock.tryLock());
                }
            });
        } catch (InterruptedException e) {
            // block() uses the uninterruptible lock(), so this is not expected
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting to insert", e);
        }
    }

    private static void closeQuietly(GraphIndexBuilder builder) {
        try {
            builder.close();
        } catch (IOException e) {
            // only thread-local scratch space; nothing to recover
        }
    }
}
