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

package io.github.jbellis.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Predictive sizing utility for JVector indexes. Unlike generic heap estimators such as {@code
 * RamUsageEstimator}, this class is configuration-aware: it constructs a small, representative
 * sample index using {@link GraphIndexBuilder} (and {@link ProductQuantization} when configured),
 * records the measured footprint of graph structures, PQ vectors/codebooks, and thread-local
 * buffers, then extrapolates the expected memory cost for larger datasets. Use this when planning
 * or capacity-sizing an index rather than when inspecting already-instantiated objects.
 */
public final class MemoryCostEstimator {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int MAX_SAMPLE_SIZE = 10_000;
    private static final double DEFAULT_MARGIN_FRACTION = 0.20; // 20%

    private MemoryCostEstimator() {
    }

    /**
     * Configuration describing the index build and optional PQ settings.
     */
    public static class IndexConfig {
        public final int dimension;
        public final int maxDegree;
        public final float overflowRatio;
        public final boolean useHierarchy;
        public final VectorSimilarityFunction similarityFunction;
        public final Integer pqSubspaces;
        public final Integer pqClusters;
        public final Boolean pqCenter;

        public IndexConfig(int dimension,
                           int maxDegree,
                           float overflowRatio,
                           boolean useHierarchy,
                           VectorSimilarityFunction similarityFunction,
                           Integer pqSubspaces,
                           Integer pqClusters,
                           Boolean pqCenter) {
            this.dimension = dimension;
            this.maxDegree = maxDegree;
            this.overflowRatio = overflowRatio;
            this.useHierarchy = useHierarchy;
            this.similarityFunction = similarityFunction;
            this.pqSubspaces = pqSubspaces;
            this.pqClusters = pqClusters;
            this.pqCenter = pqCenter;
        }

        public static IndexConfig defaultConfig(int dimension) {
            return new IndexConfig(
                dimension,
                16,
                1.5f,
                true,
                VectorSimilarityFunction.EUCLIDEAN,
                16,
                256,
                Boolean.TRUE
            );
        }

        public static IndexConfig withoutPQ(int dimension, int maxDegree, boolean useHierarchy) {
            return new IndexConfig(
                dimension,
                maxDegree,
                1.5f,
                useHierarchy,
                VectorSimilarityFunction.EUCLIDEAN,
                null,
                null,
                null
            );
        }

        public boolean usesPQ() {
            return pqSubspaces != null && pqClusters != null && pqCenter != null;
        }
    }

    /**
     * Captures per-vector and fixed costs derived from a sample build.
     */
    public static class MemoryModel {
        public final IndexConfig config;
        public final int sampleSize;

        public final long bytesPerNodeGraph;
        public final long bytesPerNodePQ;
        public final long fixedCodebookBytes;
        public final long fixedGraphOverhead;
        public final double hierarchyFactor;
        public final long bytesPerThreadIndexing;
        public final long bytesPerThreadSearch;
        private final double marginFraction;

        public MemoryModel(IndexConfig config,
                           int sampleSize,
                           long bytesPerNodeGraph,
                           long fixedGraphOverhead,
                           double hierarchyFactor,
                           long bytesPerNodePQ,
                           long fixedCodebookBytes) {
            this.config = config;
            this.sampleSize = sampleSize;
            this.bytesPerNodeGraph = Math.max(0L, bytesPerNodeGraph);
            this.fixedGraphOverhead = Math.max(0L, fixedGraphOverhead);
            this.hierarchyFactor = hierarchyFactor;

            if (config.usesPQ()) {
                this.fixedCodebookBytes = Math.max(0L, fixedCodebookBytes);
                this.bytesPerNodePQ = Math.max(0L, bytesPerNodePQ);
            } else {
                this.fixedCodebookBytes = 0L;
                this.bytesPerNodePQ = 0L;
            }

            this.bytesPerThreadIndexing = estimateThreadBuffersIndexing(config);
            this.bytesPerThreadSearch = estimateThreadBuffersSearch(config);
            this.marginFraction = DEFAULT_MARGIN_FRACTION;
        }

        private static long estimateThreadBuffersIndexing(IndexConfig config) {
            int beamWidth = 100;
            int scratchSize = Math.max(beamWidth, config.maxDegree + 1);

            int objectHeader = 16;
            int referenceBytes = 8;
            int arrayHeader = 16;

            long nodeArrayBytes = objectHeader + Integer.BYTES + (2L * referenceBytes) + (2L * arrayHeader)
                                  + (long) scratchSize * (Integer.BYTES + Float.BYTES);
            long twoNodeArrays = 2 * nodeArrayBytes;

            long nodeQueueBytes = objectHeader + referenceBytes + arrayHeader + 100L * Long.BYTES;
            long graphSearcherBytes = 3 * nodeQueueBytes + (objectHeader + 8192L);
            long ravvWrapperBytes = objectHeader + 2L * referenceBytes;

            return twoNodeArrays + graphSearcherBytes + ravvWrapperBytes;
        }

        private static long estimateThreadBuffersSearch(IndexConfig config) {
            int objectHeader = 16;
            int referenceBytes = 8;
            int arrayHeader = 16;

            long nodeQueueBytes = objectHeader + referenceBytes + arrayHeader + 100L * Long.BYTES;
            long graphSearcherBytes = 3 * nodeQueueBytes + (objectHeader + 4096L);

            long pqPartials = 0L;
            if (config.usesPQ()) {
                pqPartials = (long) config.pqSubspaces * 256 * Float.BYTES;
            }

            return graphSearcherBytes + pqPartials;
        }

        public Estimate estimateBytes(int numVectors) {
            long graphBytes = fixedGraphOverhead + (long) (bytesPerNodeGraph * numVectors * hierarchyFactor);
            long pqBytes = config.usesPQ() ? fixedCodebookBytes + bytesPerNodePQ * numVectors : 0L;
            return estimate(graphBytes + pqBytes);
        }

        public Estimate estimateBytesWithIndexingBuffers(int numVectors, int numThreads) {
            Estimate base = estimateBytes(numVectors);
            long adjusted = base.value + bytesPerThreadIndexing * (long) numThreads;
            return estimate(adjusted);
        }

        public Estimate estimateBytesWithSearchBuffers(int numVectors, int numThreads) {
            Estimate base = estimateBytes(numVectors);
            long adjusted = base.value + bytesPerThreadSearch * (long) numThreads;
            return estimate(adjusted);
        }

        private Estimate estimate(long value) {
            return new Estimate(value, marginFraction);
        }
    }

    /**
     * Build a sample index to derive a {@link MemoryModel} for the supplied configuration. The
     * sample size should remain modest (default cap: 10&nbsp;000) because this method actually
     * runs {@link GraphIndexBuilder} and (optionally) {@link ProductQuantization} to gather real
     * measurements. If you only need object-level metrics for already instantiated structures,
     * prefer {@link io.github.jbellis.jvector.util.RamUsageEstimator} instead.
     */
    public static MemoryModel createModel(IndexConfig config, int sampleSize) throws Exception {
        if (sampleSize <= 0) {
            throw new IllegalArgumentException("sampleSize must be positive; received " + sampleSize);
        }
        if (sampleSize > MAX_SAMPLE_SIZE) {
            throw new IllegalArgumentException(
                "sampleSize " + sampleSize + " exceeds the maximum " + MAX_SAMPLE_SIZE +
                    "; reduce it to keep MemoryCostEstimator runtime manageable.");
        }

        List<VectorFloat<?>> vectors = generateRandomVectors(config.dimension, sampleSize);
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, config.dimension);

        long graphBytes;
        long pqBytes = 0L;
        long fixedCodebookBytes = 0L;
        long bytesPerNodePQ = 0L;
        long bytesPerNodeGraph;
        long fixedGraphOverhead;
        double hierarchyFactor;

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

            if (!(graph instanceof OnHeapGraphIndex)) {
                throw new IllegalStateException("MemoryCostEstimator expects an OnHeapGraphIndex sample");
            }
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
            bytesPerNodeGraph = sampleSize == 0 ? 0L : level0NodeBytes / sampleSize;
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
                bytesPerNodePQ = sampleSize == 0 ? 0L : Math.max(0L, (pqBytes - fixedCodebookBytes) / sampleSize);
            }
        }

        return new MemoryModel(
            config,
            sampleSize,
            bytesPerNodeGraph,
            fixedGraphOverhead,
            hierarchyFactor,
            bytesPerNodePQ,
            fixedCodebookBytes
        );
    }

    /**
     * Represents a point estimate with a relative margin of error.
     */
    public static final class Estimate {
        private final long value;
        private final double marginFraction;

        private Estimate(long value, double marginFraction) {
            this.value = value;
            this.marginFraction = marginFraction;
        }

        /** Central estimate in bytes. */
        public long value() {
            return value;
        }

        /** Margin of error in bytes (computed as value * marginFraction, rounded). */
        public long marginBytes() {
            return Math.round(value * marginFraction);
        }

        /** Relative margin as a fraction (e.g., 0.20 == ±20%). */
        public double marginFraction() {
            return marginFraction;
        }

        /** Lower bound (value - margin). */
        public long lowerBound() {
            return Math.max(0L, value - marginBytes());
        }

        /** Upper bound (value + margin). */
        public long upperBound() {
            return value + marginBytes();
        }
    }

    private static List<VectorFloat<?>> generateRandomVectors(int dimension, int count) {
        List<VectorFloat<?>> vectors = new ArrayList<>(count);
        Random rng = ThreadLocalRandom.current();

        for (int i = 0; i < count; i++) {
            VectorFloat<?> vec = VTS.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) {
                vec.set(d, rng.nextFloat() * 2f - 1f);
            }
            vectors.add(vec);
        }

        return vectors;
    }
}
