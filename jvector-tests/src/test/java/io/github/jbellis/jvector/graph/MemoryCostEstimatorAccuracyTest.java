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
 *     <td>7.83</td><td>8.53</td><td>8.57</td><td>9.53</td><td>10.55</td><td>11.38</td><td>12.09</td>
 *   </tr>
 *   <tr>
 *     <td>Hierarchical, PQ (M=16, 16×256 ≤1024d, 32×256 ≥2048d)</td>
 *     <td>7.75</td><td>7.45</td><td>6.91</td><td>6.30</td><td>5.14</td><td>9.30</td><td>0.89</td>
 *   </tr>
 *   <tr>
 *     <td>Flat, no&nbsp;PQ (M=32)</td>
 *     <td>1.84</td><td>1.84</td><td>1.84</td><td>1.84</td><td>1.79</td><td>1.63</td><td>0.90</td>
 *   </tr>
 *   <tr>
 *     <td>Flat, PQ (M=24, 16×256 ≤1024d, 32×256 ≥2048d)</td>
 *     <td>2.35</td><td>2.27</td><td>2.06</td><td>1.79</td><td>1.05</td><td>1.20</td><td>0.04</td>
 *   </tr>
 * </table>
 * All scenarios remain well within the ±20&nbsp;% tolerance enforced by the assertions.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class MemoryCostEstimatorAccuracyTest extends LuceneTestCase {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int[] DIMENSIONS = {64, 128, 256, 512, 1024, 2048, 4096};
    private static final int BASE_SAMPLE_SIZE = 1_000;
    private static final int BASE_VERIFICATION_SIZE = 5_000;
    private static final double TOLERANCE = 0.20; // 20% relative error

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
        return new MemoryModel(config, sampleSize, measurement.graphBytes(), measurement.pqBytes());
    }

    private int sampleSizeFor(int dimension) {
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
            }
        }

        return new Measurement(graphBytes, pqBytes);
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

        private Measurement(long graphBytes, long pqBytes) {
            this.graphBytes = graphBytes;
            this.pqBytes = pqBytes;
        }

        long graphBytes() {
            return graphBytes;
        }

        long pqBytes() {
            return pqBytes;
        }

        long totalBytes() {
            return graphBytes + pqBytes;
        }
    }
}
