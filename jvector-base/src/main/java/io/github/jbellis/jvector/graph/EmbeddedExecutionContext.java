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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.PQRetrainer;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * A single carrier for the execution resources an embedder supplies to jvector, so the pool is
 * passed <b>once</b> and the "no parallel work escapes to {@link io.github.jbellis.jvector.util.PhysicalCoreExecutor#pool()}
 * or {@link ForkJoinPool#commonPool()}" guarantee lives in <b>one</b> place. Construct the context
 * with a bounded pool (or {@link #callerRuns()}), then obtain builders/compactors/writers and run
 * quantization through it; every operation routes <i>only</i> through the carried executors and never
 * through a default overload.
 *
 * <p><b>Two modes.</b> {@link #of(ForkJoinPool)} bounds all work to a supplied pool (the compaction
 * use case). {@link #callerRuns()} runs everything synchronously on the calling thread — no worker
 * threads, no pool — which lets a memtable flush run graph build and PQ/NVQ train/encode entirely on
 * its own flush-writer thread.
 *
 * <p><b>Three executor roles.</b> {@code compute} (a {@link ParallelExecutor}) drives graph build /
 * cleanup and all PQ/NVQ quantization; {@code merge} (an {@link Executor}) drives the compaction
 * merge; {@code io} (an {@link ExecutorService}) drives the optional IO-bound parallel graph writer.
 * The factories derive all three consistently from one input, so "pass one thing" stays the simple,
 * correct default; the general constructor accepts them separately for embedders that size an IO pool
 * distinctly.
 *
 * <p><b>Lifecycle.</b> The context neither owns nor shuts down the embedder's pool(s); the embedder
 * supplies and disposes them.
 *
 * <p><b>Throttling is per-operation, not stored here.</b> The compaction merge also accepts a
 * {@link io.github.jbellis.jvector.util.work.ProgressLimiter} (progress + throttle), but that is
 * scoped to a single host operation. Set it on the compactor returned by {@link #newCompactor}
 * per call — do not expect the context to carry it.
 */
@Experimental
public final class EmbeddedExecutionContext {
    private final ParallelExecutor compute;
    private final Executor merge;
    private final ExecutorService io;

    /**
     * @param compute drives graph build/cleanup and all PQ/NVQ work
     * @param merge   drives the compaction merge (a {@code ForkJoinPool} or {@code Runnable::run} for
     *                caller-runs)
     * @param io      drives the IO-bound parallel graph writer
     */
    public EmbeddedExecutionContext(ParallelExecutor compute, Executor merge, ExecutorService io) {
        this.compute = Objects.requireNonNull(compute, "compute");
        this.merge = Objects.requireNonNull(merge, "merge");
        this.io = Objects.requireNonNull(io, "io");
    }

    /** One pool for everything (compute, merge, and IO). */
    public static EmbeddedExecutionContext of(ForkJoinPool pool) {
        return new EmbeddedExecutionContext(ParallelExecutor.forkJoin(pool), pool, pool);
    }

    /** A compute pool plus a distinct IO pool; the compute pool also drives the merge. */
    public static EmbeddedExecutionContext of(ForkJoinPool computePool, ExecutorService io) {
        return new EmbeddedExecutionContext(ParallelExecutor.forkJoin(computePool), computePool, io);
    }

    /**
     * Runs every operation synchronously on the calling thread — no worker threads, no pool, common
     * pool untouched. Suitable for a memtable flush that wants graph build + PQ/NVQ encode on its own
     * thread. Encoding is byte-identical to the pool-backed path; PQ training is quality-equivalent
     * (its k-means seeds are drawn from {@code ThreadLocalRandom} either way).
     */
    public static EmbeddedExecutionContext callerRuns() {
        return new EmbeddedExecutionContext(ParallelExecutor.callerRuns(), Runnable::run, directExecutorService());
    }

    /** The compute executor (graph build + PQ/NVQ). */
    public ParallelExecutor parallelExecutor() {
        return compute;
    }

    /** The merge executor (compaction). */
    public Executor mergeExecutor() {
        return merge;
    }

    /** The IO executor (parallel graph writer). */
    public ExecutorService ioExecutor() {
        return io;
    }

    // ---- graph construction ----

    /**
     * A {@link GraphIndexBuilder} wired to the compute executor for both its executor roles.
     */
    public GraphIndexBuilder newBuilder(BuildScoreProvider scoreProvider,
                                        int dimension,
                                        int M,
                                        int beamWidth,
                                        float neighborOverflow,
                                        float alpha,
                                        boolean addHierarchy,
                                        boolean refineFinalGraph) {
        return new GraphIndexBuilder(scoreProvider, dimension, M, beamWidth, neighborOverflow, alpha,
                addHierarchy, refineFinalGraph, compute, compute);
    }

    // ---- compaction ----

    /**
     * An {@link OnDiskGraphIndexCompactor} wired to the merge executor. The caller sets a
     * {@link io.github.jbellis.jvector.util.work.ProgressLimiter} on the returned compactor per
     * operation if throttling/progress is wanted (see the class note).
     */
    public OnDiskGraphIndexCompactor newCompactor(List<OnDiskGraphIndex> sources,
                                                  List<FixedBitSet> liveNodes,
                                                  List<OrdinalMapper> remappers,
                                                  VectorSimilarityFunction similarityFunction,
                                                  int taskWindowSize) {
        return new OnDiskGraphIndexCompactor(sources, liveNodes, remappers, similarityFunction, merge, taskWindowSize);
    }

    // ---- parallel (IO-bound) graph writer ----

    /**
     * A parallel graph-writer builder wired to the IO executor. Chain feature/offset configuration on
     * the returned builder and call {@code build()}.
     */
    public OnDiskParallelGraphIndexWriter.Builder newParallelWriter(ImmutableGraphIndex graph, Path outputPath) throws IOException {
        return new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath).withExecutor(io);
    }

    // ---- product quantization ----

    /** Trains PQ on the compute executor (isotropic / unweighted). */
    public ProductQuantization trainPQ(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        return trainPQ(ravv, M, clusterCount, globallyCenter, UNWEIGHTED);
    }

    /** Trains PQ on the compute executor with an explicit anisotropic threshold. */
    public ProductQuantization trainPQ(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter, float anisotropicThreshold) {
        return ProductQuantization.compute(ravv, M, clusterCount, globallyCenter, anisotropicThreshold, compute, compute);
    }

    /** Refines an existing PQ codebook on the compute executor (one Lloyd's round, unweighted). */
    public ProductQuantization refinePQ(ProductQuantization base, RandomAccessVectorValues ravv) {
        return base.refine(ravv, 1, UNWEIGHTED, compute, compute);
    }

    /**
     * Retrains PQ across compaction sources on the compute executor — the direct-call counterpart to
     * the internal compaction retrain, and the entry point that closes the historical retrain leak.
     */
    public ProductQuantization retrainPQ(PQRetrainer retrainer, VectorSimilarityFunction similarityFunction) {
        return retrainer.retrain(similarityFunction, compute, compute);
    }

    /** As {@link #retrainPQ(PQRetrainer, VectorSimilarityFunction)} with an explicit base PQ. */
    public ProductQuantization retrainPQ(PQRetrainer retrainer, VectorSimilarityFunction similarityFunction, ProductQuantization basePQ) {
        return retrainer.retrain(similarityFunction, basePQ, compute, compute);
    }

    /** Encodes all vectors with PQ on the compute executor. */
    public PQVectors encodePQ(ProductQuantization pq, RandomAccessVectorValues ravv) {
        return pq.encodeAll(ravv, compute);
    }

    // ---- non-uniform vector quantization ----

    /** Encodes all vectors with NVQ on the compute executor. */
    public NVQVectors encodeNVQ(NVQuantization nvq, RandomAccessVectorValues ravv) {
        return nvq.encodeAll(ravv, compute);
    }

    /** An {@link ExecutorService} that runs every submitted task synchronously on the calling thread. */
    private static ExecutorService directExecutorService() {
        return new AbstractExecutorService() {
            @Override
            public void execute(Runnable command) {
                command.run();
            }

            @Override
            public void shutdown() {
            }

            @Override
            public List<Runnable> shutdownNow() {
                return Collections.emptyList();
            }

            @Override
            public boolean isShutdown() {
                return false;
            }

            @Override
            public boolean isTerminated() {
                return false;
            }

            @Override
            public boolean awaitTermination(long timeout, TimeUnit unit) {
                return true;
            }
        };
    }
}
