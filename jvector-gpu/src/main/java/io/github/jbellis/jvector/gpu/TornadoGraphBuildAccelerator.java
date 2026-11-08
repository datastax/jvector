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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphBuildAccelerator;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.NodeArray;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * Builds JVector graphs on an NVIDIA GPU with TornadoVM and NVIDIA cuVS. Select it with
 * {@code -Djvector.graph.accelerator=tornadovm} and run under the TornadoVM launcher.
 * <ol>
 * <li>Candidates: the approximate k-NN graph of all vectors, from cuVS NN-Descent. Datasets larger than device
 * memory are built in overlapping batches from host memory.</li>
 * <li>Diversity pruning with JVector's rule and the builder's alpha, on the GPU: the candidates' Gram matrix with
 * tensor cores (FP16 in, FP32 accumulate), then one work-group per node.</li>
 * <li>Reverse edges are merged, and the lists that grew are pruned again.</li>
 * <li>With a hierarchy, the upper layers are built the same way from exact candidates, with JVector's level
 * distribution.</li>
 * </ol>
 * Supports COSINE, DOT_PRODUCT and EUCLIDEAN, any dimension, and max degrees up to {@value #MAX_DEGREE}. For
 * datasets in a {@link MappedVectors} file, the vectors are read in place.
 */
public final class TornadoGraphBuildAccelerator implements GraphBuildAccelerator {

    private static final Logger LOG = Logger.getLogger(TornadoGraphBuildAccelerator.class.getName());

    public static final String NAME = "tornadovm";

    /** Largest max degree: lists hold {@link PruneKernels#SLOTS} entries before pruning. */
    static final int MAX_DEGREE = 64;
    /** NN-Descent candidates per node. */
    static final int CANDIDATES = 96;
    /**
     * A fixed number of candidates to prune from ({@code -Djvector.gpu.candidates}); by default {@link CandidateWidth}
     * chooses between {@value CandidateWidth#NARROW} and {@value #CANDIDATES} from the data.
     */
    static final Integer FIXED_WIDTH = Integer.getInteger("jvector.gpu.candidates");
    /** Device memory for the pruner besides the vectors: Gram matrices and lists of one batch, with slack. */
    private static final long PRUNE_WORKSPACE = 512L << 20;

    private static final boolean TRACE = Boolean.getBoolean("jvector.gpu.trace");

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public boolean supports(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int dim = vectors.dimension();
        if (builder.getMaxDegree() > MAX_DEGREE) {
            LOG.info(() -> "GPU build supports max degree up to " + MAX_DEGREE + ", not " + builder.getMaxDegree());
            return false;
        }
        if (n < 2 || !PackedVectors.fits(n, dim)) {
            return false;
        }
        long needed = PackedVectors.bytes(n, dim) + PRUNE_WORKSPACE;
        long budget = deviceMemoryBudget();
        if (needed > budget) {
            LOG.info(() -> String.format(Locale.ROOT, "GPU build of %d x %d vectors needs %.1f GB of device memory: run with -Dtornado.device.memory=%dGB",
                    n, dim, needed / 1e9, (needed >> 30) + 1));
            return false;
        }
        String host = HostMemory.shortfall(n, dim, builder.getMaxDegree(), vectors instanceof MappedVectors);
        if (host != null) {
            LOG.info(() -> String.format(Locale.ROOT, "GPU build of %d x %d vectors declined, building on the CPU: %s", n, dim, host));
            return false;
        }
        return runtimeAvailable();
    }

    private static boolean runtimeAvailable() {
        try {
            Class.forName("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime");
            return uk.ac.manchester.tornado.cuvs.CuVS.isAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    /** TornadoVM's device memory budget, {@code -Dtornado.device.memory} (default 4GB). */
    static long deviceMemoryBudget() {
        String size = System.getProperty("tornado.device.memory", "4GB").trim().toUpperCase(Locale.ROOT);
        long unit = size.endsWith("GB") ? 1L << 30 : size.endsWith("MB") ? 1L << 20 : size.endsWith("KB") ? 1L << 10 : 1L;
        String digits = size.replaceAll("[^0-9]", "");
        return digits.isEmpty() ? 0 : Long.parseLong(digits) * unit;
    }

    @Override
    public void build(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int dim = vectors.dimension();
        var timer = new PhaseTimer();
        try (Arena arena = Arena.ofShared()) {
            // 1. the vectors as contiguous float32 rows: the mapping itself, or a copy (normalised for COSINE)
            MemorySegment rows;
            boolean unit;
            if (vectors instanceof MappedVectors mapped) {
                rows = mapped.segment();
                unit = false;
            } else {
                unit = similarity == VectorSimilarityFunction.COSINE;
                rows = HostRows.copy(vectors, arena, unit);
            }
            Candidates.Metric metric = Candidates.Metric.of(similarity, unit);
            boolean normalise = similarity == VectorSimilarityFunction.COSINE && !unit;
            boolean euclidean = similarity == VectorSimilarityFunction.EUCLIDEAN;
            timer.lap("rows");

            // 2. while the GPU computes the candidates (step 3), the CPU works ahead in the background:
            //    - it copies out the vectors of the upper layers (the levels are drawn up front);
            //    - it prepares the pruner: the vectors packed to FP16, and its kernels compiled. Not when the dataset
            //      is larger than the device: then cuVS builds the candidates in batches, which needs the host memory
            int k = Math.min(FIXED_WIDTH != null ? FIXED_WIDTH : CANDIDATES, n - 1);
            var batching = Candidates.Batching.of(n, dim);
            int[] levels = builder.sampleGraphLevels(n);
            CompletableFuture<UpperLayers.Nodes> upperNodes = CompletableFuture.supplyAsync(() -> UpperLayers.gather(rows, levels, dim, arena));
            Supplier<GpuPrune> preparePruner = () ->
                    new GpuPrune(PackedVectors.of(rows, n, dim, normalise, euclidean), builder.getMaxDegree(), builder.getAlpha()).precompile();
            CompletableFuture<GpuPrune> preparedPruner = batching.clusters() == 1 ? CompletableFuture.supplyAsync(preparePruner) : null;
            CompletableFuture<List<UpperLayers.Layer>> upperLayers = null;
            CandidateWidth.Choice width;
            try {
                // 3. candidates: the approximate k-NN graph, from cuVS NN-Descent
                Candidates.TopK candidates;
                try {
                    candidates = Candidates.nnDescent(rows, n, dim, k, metric, batching);
                } catch (RuntimeException e) {
                    discard(preparedPruner);
                    throw e;
                }
                timer.lap("candidates");

                // 4. diversity pruning, from the short candidate lists when they are enough (tried on a sample of nodes)
                GpuGraphBuilder.Adjacency adjacency;
                try (GpuPrune pruner = preparedPruner != null ? preparedPruner.join() : preparePruner.get()) {
                    width = FIXED_WIDTH != null ? CandidateWidth.fixed(k) : CandidateWidth.choose(pruner, candidates, builder.getMaxDegree());
                    candidates = candidates.first(width.width()); // and the wide lists can be freed
                    adjacency = GpuGraphBuilder.build(pruner, n, candidates);
                }
                candidates = null; // free before the install
                timer.lap("prune");

                // 5. the GPU builds the upper layers while the CPU installs the base layer's lists into JVector's builder
                upperLayers = upperNodes.thenApplyAsync(nodes -> UpperLayers.build(nodes, dim, metric, normalise, euclidean,
                        Math.min(CANDIDATES, n - 1), builder.getMaxDegree(), builder.getAlpha(), arena));
                install(builder, 0, adjacency, null);
                adjacency = null;
                timer.lap("install");
                List<UpperLayers.Layer> layers = upperLayers.join();
                for (var layer : layers) {
                    install(builder, layer.level(), layer.adjacency(), layer.members());
                }
                // with a hierarchy the entry point is the node on the top layer; a single layer starts at the medoid
                builder.completeExternalBuild(layers.isEmpty() ? medoid(rows, n, dim, similarity) : 0);
                timer.lap("upper layers");
            } finally {
                // the background work reads the rows, which are released on return
                awaitQuietly(upperNodes);
                awaitQuietly(upperLayers);
            }
            if (TRACE) {
                System.out.printf(Locale.ROOT, "GPU build of %d x %d (%s): %s; %d batches (overlap %d), pruned from %d of %d candidates (short lists filled %.3f)%n",
                        n, dim, similarity, timer, batching.clusters(), batching.overlap(), width.width(), k, width.full());
            }
        }
    }

    /** Waits for a pruner being prepared in the background, if any, and releases it. */
    private static void discard(CompletableFuture<GpuPrune> preparedPruner) {
        if (preparedPruner == null) {
            return;
        }
        try {
            preparedPruner.join().close();
        } catch (RuntimeException ignored) {
            // the build has already failed; this is cleanup
        }
    }

    /** Waits for background work, if any, without failing: the build has either used its result or failed already. */
    private static void awaitQuietly(CompletableFuture<?> work) {
        if (work != null) {
            work.handle((result, failure) -> null).join();
        }
    }

    /** Wall-clock time of consecutive phases, for {@code -Djvector.gpu.trace}. */
    private static final class PhaseTimer {
        private final long start = System.nanoTime();
        private long last = start;
        private final StringBuilder laps = new StringBuilder();

        void lap(String phase) {
            long now = System.nanoTime();
            laps.append(String.format(Locale.ROOT, "%s %.2f s, ", phase, (now - last) / 1e9));
            last = now;
        }

        @Override
        public String toString() {
            return laps + String.format(Locale.ROOT, "total %.2f s", (last - start) / 1e9);
        }
    }

    /** Installs the lists of one layer; {@code members[x]} is the node of row {@code x}, or row x itself if null. */
    private static void install(GraphIndexBuilder builder, int level, GpuGraphBuilder.Adjacency adjacency, int[] members) {
        int slots = PruneKernels.SLOTS;
        IntStream.range(0, adjacency.sizes().length).parallel().forEach(x -> {
            int size = adjacency.sizes()[x];
            var list = new NodeArray(Math.max(size, 1));
            for (int c = 0; c < size; c++) {
                int id = adjacency.ids()[x * slots + c];
                list.addInOrder(members == null ? id : members[id], adjacency.scores()[x * slots + c]);
            }
            builder.addGraphNodeWithNeighbors(level, members == null ? x : members[x], list);
        });
    }

    /** The node most similar to the centroid, the usual entry point of a single-layer graph. */
    static int medoid(MemorySegment rows, int n, int dim, VectorSimilarityFunction similarity) {
        boolean cosine = similarity == VectorSimilarityFunction.COSINE;
        double[] centroid = IntStream.range(0, n).parallel().collect(() -> new double[dim], (acc, i) -> {
            double scale = cosine ? 1.0 / Math.sqrt(dot(rows, i, i, dim)) : 1.0;
            for (int d = 0; d < dim; d++) {
                acc[d] += value(rows, i, d, dim) * scale;
            }
        }, (a, b) -> {
            for (int d = 0; d < dim; d++) {
                a[d] += b[d];
            }
        });
        for (int d = 0; d < dim; d++) {
            centroid[d] /= n;
        }
        return IntStream.range(0, n).parallel().boxed().max((a, b) -> Double.compare(closeness(rows, a, centroid, dim, similarity),
                closeness(rows, b, centroid, dim, similarity))).orElse(0);
    }

    /** Larger is closer: cosine, dot product, or negative squared distance. */
    private static double closeness(MemorySegment rows, int i, double[] c, int dim, VectorSimilarityFunction similarity) {
        double dot = 0;
        double norm = 0;
        for (int d = 0; d < dim; d++) {
            double x = value(rows, i, d, dim);
            dot += x * c[d];
            norm += x * x;
        }
        switch (similarity) {
            case EUCLIDEAN:
                return 2 * dot - norm; // -|x - c|^2 + |c|^2
            case DOT_PRODUCT:
                return dot;
            default:
                return dot / Math.sqrt(norm);
        }
    }

    private static double dot(MemorySegment rows, int i, int j, int dim) {
        double s = 0;
        for (int d = 0; d < dim; d++) {
            s += (double) value(rows, i, d, dim) * value(rows, j, d, dim);
        }
        return s;
    }

    private static float value(MemorySegment rows, int i, int d, int dim) {
        return rows.get(ValueLayout.JAVA_FLOAT_UNALIGNED, ((long) i * dim + d) * Float.BYTES);
    }
}
