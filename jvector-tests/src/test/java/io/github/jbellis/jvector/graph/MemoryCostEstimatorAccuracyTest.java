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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.MemoryCostEstimator;
import io.github.jbellis.jvector.MemoryCostEstimator.IndexConfig;
import io.github.jbellis.jvector.MemoryCostEstimator.MemoryModel;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertTrue;

/**
 * Validates {@link MemoryCostEstimator} projections across multiple deployment styles and vector
 * dimensionalities (64&nbsp;through&nbsp;4096). Sample sizes scale with dimensionality to keep runtime
 * reasonable (see {@link #sampleSizeFor(int)}). Recent runs produced the following relative errors
 * (|estimate&nbsp;−&nbsp;measured|/measured):
 * <table border="1" cellpadding="2" cellspacing="0">
 *   <caption>Relative error (%) by configuration and dimensionality</caption>
 *   <tr>
 *     <th>Configuration</th>
 *     <th>64d</th><th>128d</th><th>256d</th><th>512d</th><th>1024d</th><th>2048d</th><th>4096d</th>
 *   </tr>
 *   <tr>
 *     <td>Hierarchical, no&nbsp;PQ (M=16)</td>
 *     <td>1.59</td><td>1.59</td><td>1.59</td><td>0.60</td><td>0.80</td><td>0.80</td><td>1.65</td>
 *   </tr>
 *   <tr>
 *     <td>Hierarchical, PQ (M=16, 16×256 ≤1024d, 32×256 ≥2048d)</td>
 *     <td>1.45</td><td>1.40</td><td>1.29</td><td>0.40</td><td>0.04</td><td>0.16</td><td>0.09</td>
 *   </tr>
 *   <tr>
 *     <td>Flat, no&nbsp;PQ (M=32)</td>
 *     <td>0.95</td><td>0.95</td><td>0.95</td><td>0.91</td><td>0.84</td><td>0.69</td><td>0.00</td>
 *   </tr>
 *   <tr>
 *     <td>Flat, PQ (M=24, 16×256 ≤1024d, 32×256 ≥2048d)</td>
 *     <td>1.21</td><td>1.17</td><td>1.10</td><td>0.88</td><td>0.55</td><td>0.23</td><td>0.00</td>
 *   </tr>
 *   <tr>
 *     <td>Hierarchical, high-M (M=48)</td>
 *     <td>0.10</td><td>0.21</td><td>0.51</td><td>0.19</td><td>0.42</td><td>0.60</td><td>0.96</td>
 *   </tr>
 *   <tr>
 *     <td>Hierarchical, cosine</td>
 *     <td>2.26</td><td>1.59</td><td>1.59</td><td>0.60</td><td>0.80</td><td>0.80</td><td>1.65</td>
 *   </tr>
 * </table>
 * All scenarios remain well within the ±5&nbsp;% tolerance enforced by the assertions. The full
 * sweep (six configurations × seven dimensionalities) completes in ~23&nbsp;s on an M2 Max laptop;
 * expect higher runtimes if CPU parallelism or vector acceleration is limited. If you instead
 * force every run to use the 10&nbsp;000 vector sample cap, the 64d scenarios finish in a few seconds
 * while the 4096d cases stretch past a minute apiece because Product Quantization dominates wall
 * time at the higher dimensionalities.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class MemoryCostEstimatorAccuracyTest extends LuceneTestCase {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int[] DIMENSIONS = {64, 128, 256, 512, 1024, 2048, 4096};
    private static final int BASE_SAMPLE_SIZE = 1_000;
    private static final int BASE_VERIFICATION_SIZE = 5_000;
    private static final double TOLERANCE = 0.05; // 5% relative error (roughly 2x worst observed)

    @Test
    public void testEstimateAccuracyHierarchicalWithPQ() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = new IndexConfig(
                dimension,
                16,
                1.5f,
                true,
                VectorSimilarityFunction.EUCLIDEAN,
                pqSubspacesFor(dimension),
                256,
                Boolean.TRUE
            );

            verifyEstimateAccuracy(config, dimension, "hierarchical+PQ");
        }
    }

    @Test
    public void testEstimateAccuracyHierarchicalWithoutPQ() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = IndexConfig.withoutPQ(dimension, 16, true);
            verifyEstimateAccuracy(config, dimension, "hierarchical");
        }
    }

    @Test
    public void testEstimateAccuracyFlatNoPQ() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = IndexConfig.withoutPQ(dimension, 32, false);
            verifyEstimateAccuracy(config, dimension, "flat");
        }
    }

    @Test
    public void testEstimateAccuracyFlatPQ() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = new IndexConfig(
                dimension,
                24,
                1.4f,
                false,
                VectorSimilarityFunction.EUCLIDEAN,
                pqSubspacesFor(dimension) / 2,
                256,
                Boolean.FALSE
            );

            verifyEstimateAccuracy(config, dimension, "flat+PQ");
        }
    }

    @Test
    public void testEstimateAccuracyHierarchicalHighDegree() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = new IndexConfig(
                dimension,
                48,
                1.5f,
                true,
                VectorSimilarityFunction.EUCLIDEAN,
                null,
                null,
                null
            );

            verifyEstimateAccuracy(config, dimension, "hierarchical-highM");
        }
    }

    @Test
    public void testEstimateAccuracyHierarchicalCosine() throws Exception {
        for (int dimension : DIMENSIONS) {
            IndexConfig config = new IndexConfig(
                dimension,
                16,
                1.5f,
                true,
                VectorSimilarityFunction.COSINE,
                null,
                null,
                null
            );

            verifyEstimateAccuracy(config, dimension, "hierarchical-cosine");
        }
    }

    private void verifyEstimateAccuracy(IndexConfig config, int dimension, String label) throws Exception {
        int sampleSize = Math.max(sampleSizeFor(dimension), minimumSampleSize(config));
        int verificationSize = verificationSizeFor(dimension);

        MemoryModel model = buildModel(config, sampleSize, 7L);
        Measurement measurement = measure(config, verificationSize, 13L);

        long estimatedBytes = model.estimateBytes(verificationSize).value();
        long measuredBytes = measurement.totalBytes();
        double relativeError = Math.abs(estimatedBytes - measuredBytes) / (double) measuredBytes;

        System.out.printf(
            "Memory estimator: dim=%d, %s -> estimated=%d, measured=%d, error=%.2f%%%n",
            dimension,
            label,
            estimatedBytes,
            measuredBytes,
            relativeError * 100.0
        );

        assertTrue(
            String.format(
                "Estimated bytes=%d, measured bytes=%d, relative error=%.2f%%",
                estimatedBytes,
                measuredBytes,
                relativeError * 100.0
            ),
            relativeError <= TOLERANCE
        );
    }

    private MemoryModel buildModel(IndexConfig config, int sampleSize, long seed) throws Exception {
        Measurement measurement = measure(config, sampleSize, seed);
        return new MemoryModel(
            config,
            sampleSize,
            measurement.bytesPerNodeGraph(),
            measurement.fixedGraphOverhead(),
            measurement.hierarchyFactor(),
            measurement.bytesPerNodePQ(),
            measurement.fixedCodebookBytes()
        );
    }

    private int sampleSizeFor(int dimension) {
        // Scaling the sample size keeps 64d runs sub-second while preventing 4096d + PQ
        // configurations from ballooning past a minute; the 10k cap is still available for
        // deeper diagnostics, but not used in this quick regression sweep.
        if (dimension >= 4096) {
            return 200;
        }
        if (dimension >= 2048) {
            return 300;
        }
        if (dimension >= 1024) {
            return 400;
        }
        if (dimension >= 512) {
            return 600;
        }
        return BASE_SAMPLE_SIZE;
    }

    private int verificationSizeFor(int dimension) {
        if (dimension >= 4096) {
            return 1_000;
        }
        if (dimension >= 2048) {
            return 2_000;
        }
        if (dimension >= 1024) {
            return 3_000;
        }
        if (dimension >= 512) {
            return 4_000;
        }
        return BASE_VERIFICATION_SIZE;
    }

    private int minimumSampleSize(IndexConfig config) {
        if (config.usesPQ()) {
            return config.pqClusters;
        }
        return 1;
    }

    private int pqSubspacesFor(int dimension) {
        if (dimension >= 4096) {
            return 64;
        }
        if (dimension >= 2048) {
            return 48;
        }
        if (dimension >= 1024) {
            return 32;
        }
        return 16;
    }

    private Measurement measure(IndexConfig config, int size, long seed) throws Exception {
        List<VectorFloat<?>> vectors = generateVectors(config.dimension, size, seed);
        var ravv = new ListRandomAccessVectorValues(vectors, config.dimension);

        long graphBytes;
        long pqBytes = 0L;
        long bytesPerNodeGraph;
        long fixedGraphOverhead;
        double hierarchyFactor;
        long bytesPerNodePQ = 0L;
        long fixedCodebookBytes = 0L;

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, config.similarityFunction);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
            bsp,
            ravv.dimension(),
            config.maxDegree,
            100,
            config.overflowRatio,
            1.2f,
            config.useHierarchy,
            true
        )) {
            ImmutableGraphIndex graph = builder.build(ravv);
            graphBytes = graph.ramBytesUsed();

            OnHeapGraphIndex onHeapGraph = (OnHeapGraphIndex) graph;
            long sumNodeBytes = 0L;
            long level0NodeBytes = 0L;
            for (int level = 0; level <= onHeapGraph.getMaxLevel(); level++) {
                long perNode = onHeapGraph.ramBytesUsedOneNode(level);
                long nodesAtLevel = onHeapGraph.size(level);
                long levelBytes = perNode * nodesAtLevel;
                sumNodeBytes += levelBytes;
                if (level == 0) {
                    level0NodeBytes = levelBytes;
                }
            }

            fixedGraphOverhead = Math.max(0L, graphBytes - sumNodeBytes);
            bytesPerNodeGraph = size == 0 ? 0L : level0NodeBytes / size;
            hierarchyFactor = level0NodeBytes == 0
                ? 1.0
                : Math.max(1.0, (double) sumNodeBytes / (double) level0NodeBytes);
            if (!config.useHierarchy) {
                hierarchyFactor = 1.0;
            }

            if (config.usesPQ()) {
                ProductQuantization pq = ProductQuantization.compute(
                    ravv,
                    config.pqSubspaces,
                    config.pqClusters,
                    config.pqCenter,
                    KMeansPlusPlusClusterer.UNWEIGHTED,
                    PhysicalCoreExecutor.pool(),
                    ForkJoinPool.commonPool()
                );

                PQVectors pqVectors = pq.encodeAll(ravv, PhysicalCoreExecutor.pool());
                pqBytes = pqVectors.ramBytesUsed();
                fixedCodebookBytes = pq.ramBytesUsed();
                bytesPerNodePQ = size == 0 ? 0L : Math.max(0L, (pqBytes - fixedCodebookBytes) / size);
            }
        }

        return new Measurement(
            graphBytes,
            pqBytes,
            bytesPerNodeGraph,
            fixedGraphOverhead,
            hierarchyFactor,
            bytesPerNodePQ,
            fixedCodebookBytes
        );
    }

    private List<VectorFloat<?>> generateVectors(int dimension, int count, long seed) {
        Random rng = new Random(seed);
        List<VectorFloat<?>> vectors = new ArrayList<>(count);

        for (int i = 0; i < count; i++) {
            VectorFloat<?> vec = VTS.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) {
                vec.set(d, rng.nextFloat() * 2f - 1f);
            }
            vectors.add(vec);
        }

        return vectors;
    }

    private static class Measurement {
        private final long graphBytes;
        private final long pqBytes;
        private final long bytesPerNodeGraph;
        private final long fixedGraphOverhead;
        private final double hierarchyFactor;
        private final long bytesPerNodePQ;
        private final long fixedCodebookBytes;

        private Measurement(long graphBytes,
                             long pqBytes,
                             long bytesPerNodeGraph,
                             long fixedGraphOverhead,
                             double hierarchyFactor,
                             long bytesPerNodePQ,
                             long fixedCodebookBytes) {
            this.graphBytes = graphBytes;
            this.pqBytes = pqBytes;
            this.bytesPerNodeGraph = bytesPerNodeGraph;
            this.fixedGraphOverhead = fixedGraphOverhead;
            this.hierarchyFactor = hierarchyFactor;
            this.bytesPerNodePQ = bytesPerNodePQ;
            this.fixedCodebookBytes = fixedCodebookBytes;
        }

        long bytesPerNodeGraph() {
            return bytesPerNodeGraph;
        }

        long fixedGraphOverhead() {
            return fixedGraphOverhead;
        }

        double hierarchyFactor() {
            return hierarchyFactor;
        }

        long bytesPerNodePQ() {
            return bytesPerNodePQ;
        }

        long fixedCodebookBytes() {
            return fixedCodebookBytes;
        }

        long totalBytes() {
            return graphBytes + pqBytes;
        }
    }
}
