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
import io.github.jbellis.jvector.graph.StreamingASHIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.VectorCompressor;

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
 * one thread per construction worker, to a {@link StreamingASHIndex} that encodes each one as it arrives and keeps
 * no copy of the vectors. Searches then use the index's own codes, reranked from the dataset's vectors. An
 * {@code onDisk} streaming build uses {@link #stream} and then writes and reloads the index, as a flush does.
 *
 * <p>With {@code coldStartSize} zero the index starts with the construction compressor's model, standing in for
 * a model inherited from an earlier index (a training window can stand in for that index's data). Otherwise it
 * starts cold: it builds with exact scores until its first model is ready, trained on its sample with
 * {@code trainWhileStreaming}, else a random projection fixed after {@code coldStartSize} vectors.</p>
 */
final class StreamingBuild implements IndexBuildStrategy {
    private final CompressorParameters.ASHParameters params;
    private final Supplier<VectorCompressor<?>> model;

    StreamingBuild(CompressorParameters.ASHParameters params, Supplier<VectorCompressor<?>> model) {
        this.params = params;
        this.model = model;
    }

    @Override
    public BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) {
        long start = System.nanoTime();
        var index = stream(build, ds, params, model);
        double buildS = (System.nanoTime() - start) / 1e9;
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = Map.of(Set.of(), index.graph());
        return new BuiltIndexes(indexes, Map.of(), buildS, Map.of(Set.of(), StoredCodes.Loaded.built(index.codes())));
    }

    /**
     * Streams the dataset's base vectors into a new {@link StreamingASHIndex}, readies it as a flush does and cleans
     * up its graph, so it can be searched in memory or written. Reports the streaming timeline, timings and memory.
     *
     * @param model the inherited model for a warm start ({@code coldStartSize} zero); not called for a cold start
     */
    static StreamingASHIndex stream(BuildConfig build, DataSet ds, CompressorParameters.ASHParameters params,
                                    Supplier<VectorCompressor<?>> model) {
        var vectors = ds.getBaseRavv();
        int n = vectors.size();
        int dimension = ds.getDimension();
        int reservoir = 20 * dimension;
        Function<BuildScoreProvider, GraphIndexBuilder> builders =
                scores -> GraphBuilderFactory.forCompressed(scores, dimension, build);
        int coldStartSize = build.coldStartSize == null ? 0 : build.coldStartSize;
        var options = new StreamingASHIndex.Options();
        options.reservoirSize = reservoir;
        if (coldStartSize > 0) options.coldStartSize = coldStartSize;
        boolean train = Boolean.TRUE.equals(build.trainWhileStreaming);
        options.train = train;
        StreamingASHIndex index = coldStartSize > 0
                ? StreamingASHIndex.coldStart(vectors, ds.getSimilarityFunction(), params.encodedBits(),
                        params.bitsPerDimension(), builders, options)
                : StreamingASHIndex.withModel(vectors, ds.getSimilarityFunction(), (AsymmetricHashing) model.get(),
                        builders, options);

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
        // The flush: every vector ends on a trained model, as a written index needs.
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
        System.out.format(Locale.ROOT, "%s: flush preparation took %.2f s; model %s trained at %d vectors; "
                        + "%d zero codes for missing vectors; the first model held inserts for %.3f s%n",
                ds.getName(), flushS, index.model(), index.vectorsAtLastTraining(), index.codes().reencodedMissing(),
                index.firstModelPauseNanos() / 1e9);
        try (var progress = PhaseProgress.timed("Graph cleanup")) {
            index.builder().cleanup();
        }
        var codes = index.codes();
        System.out.format(Locale.ROOT,
                "%s: streamed %d vectors in %.1f s (%.0f vectors/s, %d threads) with %s model %s%n",
                ds.getName(), n, insertS, n / insertS, threads,
                coldStartSize > 0 ? "a cold-start (after " + coldStartSize + " vectors)" : "an inherited",
                index.model());
        System.out.format(Locale.ROOT, "%s: index holds %.1f MB (codes %.1f MB, graph %.1f MB); the vectors, read in place, are %.1f MB%n",
                ds.getName(), index.ramBytesUsed() / 1048576.0, codes.ramBytesUsed() / 1048576.0,
                index.graph().ramBytesUsed() / 1048576.0, (double) n * dimension * Float.BYTES / 1048576.0);
        return index;
    }
}
