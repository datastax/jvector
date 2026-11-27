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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.StreamingASHIndex;
import io.github.jbellis.jvector.graph.VectorMean;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Builds an in-memory index the way a streaming index is built: base vectors are added in arrival order, from
 * one thread per construction worker, to an index that keeps no copy of the vectors. How the index starts and
 * what each insert does depend on the construction compression:
 * <ul>
 *   <li>ASH: a {@link StreamingASHIndex}, which encodes each vector as it arrives. With {@code coldStartSize} zero
 *   it starts with the construction compressor's model, standing in for a model inherited from an earlier index (a
 *   training window can stand in for that index's data). Otherwise it starts cold: it builds with exact scores until
 *   its first model is ready, trained on its sample with {@code trainWhileStreaming}, else a random projection
 *   fixed after {@code coldStartSize} vectors.</li>
 *   <li>PQ: starts with the construction compressor's model; each arriving vector is encoded into mutable PQ codes
 *   and then added to the graph, as an index under construction in a database does.</li>
 *   <li>None: each arriving vector is added to the graph with exact scores.</li>
 * </ul>
 * Searches then use the index's own codes, reranked from the dataset's vectors. An {@code onDisk} streaming build
 * uses {@link #stream} and then writes and reloads the index, as a flush does.
 */
final class StreamingBuild implements IndexBuildStrategy {
    private final CompressorParameters params;
    private final Supplier<VectorCompressor<?>> model;

    StreamingBuild(CompressorParameters params, Supplier<VectorCompressor<?>> model) {
        this.params = params;
        this.model = model;
    }

    @Override
    public BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) {
        long start = System.nanoTime();
        var index = stream(build, ds, params, model, false);
        double buildS = (System.nanoTime() - start) / 1e9;
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = Map.of(Set.of(), index.graph());
        return new BuiltIndexes(indexes, Map.of(), buildS, index.codes() == null ? Map.of()
                : Map.of(Set.of(), StoredCodes.Loaded.built(index.codes())));
    }

    /** A streaming index: how it starts is its codec's, and each arriving vector is added with {@link #add}. */
    interface Index {
        /** Adds the vector at {@code ordinal}, read from the caller's vectors. */
        void add(int ordinal);

        /** Readies the index to be written, as a flush does; call only while no vectors are being added. */
        default void prepareForFlush() {}

        GraphIndexBuilder builder();

        OnHeapGraphIndex graph();

        /** The codes the index built as vectors arrived, or null for an index scored exactly. */
        CompressedVectors codes();

        /** The codes to write beside the index, or null for an index scored exactly. */
        default CompressedVectors writtenCodes() {
            return codes();
        }

        /** The mean of the vectors added, kept as they arrived, or null when the index was not asked to keep it. */
        VectorFloat<?> mean();

        long ramBytesUsed();

        /** How the index started and scores, for the console. */
        String start();

        /** What the flush preparation did, for the console; null when it does nothing. */
        default String flushReport() {
            return null;
        }

        /** True while the index trains a model while vectors arrive; see {@code trainWhileStreaming}. */
        default boolean isTraining() {
            return false;
        }

        /** True while the index moves its codes to a newer model. */
        default boolean isChangingModel() {
            return false;
        }
    }

    /**
     * Streams the dataset's base vectors into a new streaming index, readies it as a flush does and cleans up its
     * graph, so it can be searched in memory or written. Reports the streaming timeline, timings and memory.
     *
     * @param model    the inherited model for a warm start; not called for a cold start or without compression
     * @param keepMean whether the index keeps the mean of the vectors as they arrive, as NVQ needs to write it; an
     *                 ASH index always does
     */
    static Index stream(BuildConfig build, DataSet ds, CompressorParameters params,
                        Supplier<VectorCompressor<?>> model, boolean keepMean) {
        var vectors = ds.getBaseRavv();
        int n = vectors.size();
        int dimension = ds.getDimension();
        boolean train = Boolean.TRUE.equals(build.trainWhileStreaming);
        Index index = start(build, ds, params, model, keepMean);

        // Inserts come from their own threads, one per construction worker, as a database's writes do, so the
        // library's pool stays free for training. Each thread takes the next ordinal: arrival order is file order.
        int threads = GraphBuilderFactory.constructionThreads(build);
        var next = new AtomicInteger();
        var failure = new AtomicReference<Throwable>();
        long start = System.nanoTime();
        // Samples when training starts and ends and when the model change completes, in vectors and seconds.
        var timeline = new StringBuilder();
        var monitor = !train ? null : new Thread(() -> {
            boolean wasTraining = false, wasChanging = false;
            while (!Thread.currentThread().isInterrupted()) {
                boolean training = index.isTraining(), changing = index.isChangingModel();
                String at = String.format(Locale.ROOT, "%d vectors, %.2f s",
                        Math.min(next.get(), n), (System.nanoTime() - start) / 1e9);
                if (training && !wasTraining) timeline.append("; training started at ").append(at);
                if (!training && wasTraining) timeline.append("; training finished at ").append(at);
                if (changing && !wasChanging) timeline.append("; model change started at ").append(at);
                if (!changing && wasChanging) timeline.append("; model change finished at ").append(at);
                wasTraining = training;
                wasChanging = changing;
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "streaming-monitor");
        if (monitor != null) {
            monitor.setDaemon(true);
            monitor.start();
        }
        try (var progress = PhaseProgress.counting("Streaming insertion", n, () -> Math.min(next.get(), n))) {
            var workers = new ArrayList<Thread>();
            for (int t = 0; t < threads; t++) {
                var worker = new Thread(() -> {
                    try {
                        for (int i; (i = next.getAndIncrement()) < n; ) index.add(i);
                    } catch (Throwable e) {
                        failure.compareAndSet(null, e);
                        next.set(n);
                    }
                }, "streaming-insert-" + t);
                workers.add(worker);
                worker.start();
            }
            for (var worker : workers) worker.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        if (failure.get() != null) throw new RuntimeException("Streaming insertion failed", failure.get());
        double insertS = (System.nanoTime() - start) / 1e9;
        // The flush: an ASH index ends every vector on a trained model, as a written index needs.
        long flushStart = System.nanoTime();
        index.prepareForFlush();
        double flushS = (System.nanoTime() - flushStart) / 1e9;
        if (monitor != null) {
            monitor.interrupt();
            try {
                monitor.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            System.out.format("%s: streaming timeline: ingest ended at %.2f s%s%n", ds.getName(), insertS,
                    timeline.length() == 0 ? "; no training seen" : timeline);
        }
        var flushReport = index.flushReport();
        if (flushReport != null) {
            System.out.format(Locale.ROOT, "%s: flush preparation took %.2f s; %s%n", ds.getName(), flushS, flushReport);
        }
        try (var progress = PhaseProgress.timed("Graph cleanup")) {
            index.builder().cleanup();
        }
        var codes = index.codes();
        System.out.format(Locale.ROOT,
                "%s: streamed %d vectors in %.1f s (%.0f vectors/s, %d threads) with %s%n",
                ds.getName(), n, insertS, n / insertS, threads, index.start());
        System.out.format(Locale.ROOT, "%s: index holds %.1f MB (codes %.1f MB, graph %.1f MB); the vectors, read in place, are %.1f MB%n",
                ds.getName(), index.ramBytesUsed() / 1048576.0, codes == null ? 0 : codes.ramBytesUsed() / 1048576.0,
                index.graph().ramBytesUsed() / 1048576.0, (double) n * dimension * Float.BYTES / 1048576.0);
        return index;
    }

    /** Starts the index the construction compression streams with; see the class comment. */
    private static Index start(BuildConfig build, DataSet ds, CompressorParameters params,
                               Supplier<VectorCompressor<?>> model, boolean keepMean) {
        var vectors = ds.getBaseRavv();
        int dimension = ds.getDimension();
        int coldStartSize = build.coldStartSize == null ? 0 : build.coldStartSize;
        if (params instanceof CompressorParameters.ASHParameters) {
            var ash = (CompressorParameters.ASHParameters) params;
            Function<BuildScoreProvider, GraphIndexBuilder> builders =
                    scores -> GraphBuilderFactory.forCompressed(scores, dimension, build);
            var options = new StreamingASHIndex.Options();
            options.reservoirSize = 20 * dimension;
            if (coldStartSize > 0) options.coldStartSize = coldStartSize;
            options.train = Boolean.TRUE.equals(build.trainWhileStreaming);
            StreamingASHIndex index = coldStartSize > 0
                    ? StreamingASHIndex.coldStart(vectors, ds.getSimilarityFunction(), ash.encodedBits(),
                            ash.bitsPerDimension(), builders, options)
                    : StreamingASHIndex.withModel(vectors, (AsymmetricHashing) model.get(), builders, options);
            return new ASHIndex(index, coldStartSize);
        }
        if (coldStartSize > 0 || Boolean.TRUE.equals(build.trainWhileStreaming)) {
            throw new IllegalArgumentException("Only ASH streaming builds start cold or train while streaming");
        }
        if (params == CompressorParameters.NONE) {
            return new ArrivalIndex(vectors, null,
                    GraphBuilderFactory.forFullPrecision(vectors, ds.getSimilarityFunction(), build), keepMean);
        }
        var compressor = model.get();
        if (!(compressor instanceof ProductQuantization)) {
            throw new IllegalArgumentException("Streaming builds support ASH, PQ or no construction compression");
        }
        var codes = new MutablePQVectors((ProductQuantization) compressor);
        var scores = GraphBuilderFactory.compressedScores(codes, ds.getSimilarityFunction(), null);
        return new ArrivalIndex(vectors, codes, GraphBuilderFactory.forCompressed(scores, dimension, build), keepMean);
    }

    /** A {@link StreamingASHIndex}, which starts warm or cold and may train while vectors arrive. */
    private static final class ASHIndex implements Index {
        private final StreamingASHIndex index;
        private final int coldStartSize;

        ASHIndex(StreamingASHIndex index, int coldStartSize) {
            this.index = index;
            this.coldStartSize = coldStartSize;
        }

        @Override public void add(int ordinal) { index.add(ordinal); }
        @Override public void prepareForFlush() { index.prepareForFlush(); }
        @Override public GraphIndexBuilder builder() { return index.builder(); }
        @Override public OnHeapGraphIndex graph() { return index.graph(); }
        @Override public CompressedVectors codes() { return index.codes(); }
        @Override public CompressedVectors writtenCodes() { return index.codes().snapshot(); }
        @Override public VectorFloat<?> mean() { return index.mean(); }
        @Override public long ramBytesUsed() { return index.ramBytesUsed(); }
        @Override public boolean isTraining() { return index.isTraining(); }
        @Override public boolean isChangingModel() { return index.isChangingModel(); }

        @Override
        public String start() {
            return (coldStartSize > 0 ? "a cold-start (after " + coldStartSize + " vectors)" : "an inherited")
                    + " model " + index.model();
        }

        @Override
        public String flushReport() {
            return String.format(Locale.ROOT, "model %s trained at %d vectors; "
                            + "%d zero codes for missing vectors; the first model held inserts for %.3f s",
                    index.model(), index.vectorsAtLastTraining(), index.codes().reencodedMissing(),
                    index.firstModelPauseNanos() / 1e9);
        }
    }

    /**
     * An index whose model, if any, is fixed before the first vector arrives: each vector is encoded into the
     * mutable codes on arrival, then added to the graph, which scores with those codes, or exactly without them.
     */
    private static final class ArrivalIndex implements Index {
        private final Supplier<RandomAccessVectorValues> vectors;
        private final MutablePQVectors codes;
        private final GraphIndexBuilder builder;
        private final VectorMean.Running mean;

        ArrivalIndex(RandomAccessVectorValues vectors, MutablePQVectors codes, GraphIndexBuilder builder,
                     boolean keepMean) {
            this.vectors = vectors.threadLocalSupplier();
            this.codes = codes;
            this.builder = builder;
            this.mean = keepMean ? new VectorMean.Running(vectors.dimension()) : null;
        }

        @Override
        public void add(int ordinal) {
            VectorFloat<?> v = vectors.get().getVector(ordinal);
            if (mean != null) mean.accept(v);
            if (codes != null) codes.encodeAndSet(ordinal, v);
            builder.addGraphNode(ordinal, v);
        }

        @Override public GraphIndexBuilder builder() { return builder; }
        @Override public OnHeapGraphIndex graph() { return (OnHeapGraphIndex) builder.getGraph(); }
        @Override public CompressedVectors codes() { return codes; }
        @Override public VectorFloat<?> mean() { return mean == null ? null : mean.mean(); }

        @Override
        public long ramBytesUsed() {
            return graph().ramBytesUsed() + (codes == null ? 0 : codes.ramBytesUsed());
        }

        @Override
        public String start() {
            return codes == null ? "exact scores" : "an inherited model " + codes.getCompressor();
        }
    }
}
