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
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * A single carrier for the execution resources an embedder supplies to jvector, so the pool is
 * passed <b>once</b> and the "no parallel work escapes to {@link io.github.jbellis.jvector.util.PhysicalCoreExecutor#pool()}
 * or {@link ForkJoinPool#commonPool()}" guarantee lives in <b>one</b> place. Construct the context
 * with a bounded pool, then obtain builders/compactors/writers and run quantization through it;
 * every operation routes <i>only</i> through the carried executors and never through a default
 * overload.
 *
 * <p><b>Two executors.</b> Graph and PQ/NVQ work is CPU-bound and runs on {@code compute}; the
 * optional parallel graph <i>write</i> is IO-bound and runs on {@code io}, which defaults to
 * {@code compute}. Passing one pool via {@link #of(ForkJoinPool)} is the simple, correct default;
 * an embedder that enables parallel writes can supply a separate IO pool without touching any
 * other call site.
 *
 * <p><b>Why a {@code ForkJoinPool}, not a {@link ParallelExecutor}.</b> Product/NV quantization is
 * expressed with the parallel-stream-on-a-pool idiom, which only confines work when the pool is a
 * {@code ForkJoinPool}. The context therefore carries a concrete compute {@code ForkJoinPool} so it
 * can serve PQ (needs a {@code ForkJoinPool}), the graph builder (via
 * {@link ParallelExecutor#forkJoin}), and the compactor (via {@link java.util.concurrent.Executor})
 * uniformly. A caller-runs, single-threaded mode is <i>not</i> offered here — PQ requires a real
 * pool; for a build-only caller-runs graph, use {@link ParallelExecutor#callerRuns()} on
 * {@link GraphIndexBuilder} directly.
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
    private final ForkJoinPool compute;
    private final ExecutorService io;
    private final ParallelExecutor computeParallel;

    /**
     * @param compute the pool for all CPU-bound build / cleanup / merge / PQ / NVQ work
     * @param io      the pool for the IO-bound parallel graph writer; {@code null} defaults to
     *                {@code compute}
     */
    public EmbeddedExecutionContext(ForkJoinPool compute, ExecutorService io) {
        this.compute = Objects.requireNonNull(compute, "compute");
        this.io = (io != null) ? io : compute;
        this.computeParallel = ParallelExecutor.forkJoin(compute);
    }

    /** One pool for everything (compute and IO). */
    public static EmbeddedExecutionContext of(ForkJoinPool pool) {
        return new EmbeddedExecutionContext(pool, null);
    }

    /** Separate compute and IO pools. */
    public static EmbeddedExecutionContext of(ForkJoinPool compute, ExecutorService io) {
        return new EmbeddedExecutionContext(compute, io);
    }

    /** The compute pool, as a {@link ParallelExecutor} (both simd and parallel roles). */
    public ParallelExecutor parallelExecutor() {
        return computeParallel;
    }

    /** The compute pool. */
    public ForkJoinPool computePool() {
        return compute;
    }

    /** The IO pool (equal to the compute pool unless a separate one was supplied). */
    public ExecutorService ioExecutor() {
        return io;
    }

    // ---- graph construction ----

    /**
     * A {@link GraphIndexBuilder} wired to the compute pool for both its executor roles. Equivalent
     * to the pool-taking builder constructor with {@code compute} passed for both executors.
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
                addHierarchy, refineFinalGraph, computeParallel, computeParallel);
    }

    // ---- compaction ----

    /**
     * An {@link OnDiskGraphIndexCompactor} wired to the compute pool. The caller sets a
     * {@link io.github.jbellis.jvector.util.work.ProgressLimiter} on the returned compactor per
     * operation if throttling/progress is wanted (see the class note).
     */
    public OnDiskGraphIndexCompactor newCompactor(List<OnDiskGraphIndex> sources,
                                                  List<FixedBitSet> liveNodes,
                                                  List<OrdinalMapper> remappers,
                                                  VectorSimilarityFunction similarityFunction,
                                                  int taskWindowSize) {
        return new OnDiskGraphIndexCompactor(sources, liveNodes, remappers, similarityFunction, compute, taskWindowSize);
    }

    // ---- parallel (IO-bound) graph writer ----

    /**
     * A parallel graph-writer builder wired to the IO pool. Chain feature/offset configuration on
     * the returned builder and call {@code build()}.
     */
    public OnDiskParallelGraphIndexWriter.Builder newParallelWriter(ImmutableGraphIndex graph, Path outputPath) throws IOException {
        return new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath).withExecutor(io);
    }

    // ---- product quantization ----

    /** Trains PQ on the compute pool (isotropic / unweighted). */
    public ProductQuantization trainPQ(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        return trainPQ(ravv, M, clusterCount, globallyCenter, UNWEIGHTED);
    }

    /** Trains PQ on the compute pool with an explicit anisotropic threshold. */
    public ProductQuantization trainPQ(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter, float anisotropicThreshold) {
        return ProductQuantization.compute(ravv, M, clusterCount, globallyCenter, anisotropicThreshold, compute, compute);
    }

    /** Refines an existing PQ codebook on the compute pool (one Lloyd's round, unweighted). */
    public ProductQuantization refinePQ(ProductQuantization base, RandomAccessVectorValues ravv) {
        return base.refine(ravv, 1, UNWEIGHTED, compute, compute);
    }

    /**
     * Retrains PQ across compaction sources on the compute pool — the direct-call counterpart to
     * the internal compaction retrain, and the entry point that closes the historical retrain leak.
     */
    public ProductQuantization retrainPQ(PQRetrainer retrainer, VectorSimilarityFunction similarityFunction) {
        return retrainer.retrain(similarityFunction, compute, compute);
    }

    /** As {@link #retrainPQ(PQRetrainer, VectorSimilarityFunction)} with an explicit base PQ. */
    public ProductQuantization retrainPQ(PQRetrainer retrainer, VectorSimilarityFunction similarityFunction, ProductQuantization basePQ) {
        return retrainer.retrain(similarityFunction, basePQ, compute, compute);
    }

    /** Encodes all vectors with PQ on the compute pool. */
    public PQVectors encodePQ(ProductQuantization pq, RandomAccessVectorValues ravv) {
        return pq.encodeAll(ravv, compute);
    }

    // ---- non-uniform vector quantization ----

    /** Encodes all vectors with NVQ on the compute pool. */
    public NVQVectors encodeNVQ(NVQuantization nvq, RandomAccessVectorValues ravv) {
        return nvq.encodeAll(ravv, compute);
    }
}
