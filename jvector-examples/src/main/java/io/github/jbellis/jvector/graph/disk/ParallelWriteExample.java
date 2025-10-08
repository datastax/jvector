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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Example demonstrating how to use parallel writes with OnDiskGraphIndexWriter.
 * <p>
 * Usage patterns:
 * <pre>
 * // Sequential (default):
 * var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .build();
 * writer.write(featureSuppliers);
 *
 * // Parallel:
 * var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .withParallelWrites(true)  // Enable parallel writes
 *     .build();
 * writer.write(featureSuppliers);
 * </pre>
 */
public class ParallelWriteExample {
    
    /**
     * Benchmark comparison between sequential and parallel writes using NVQ + FUSED_ADC features.
     * This matches the configuration used in Grid.buildOnDisk for realistic performance testing.
     */
    public static void benchmarkComparison(ImmutableGraphIndex graph,
                                          Path sequentialPath,
                                          Path parallelPath,
                                          RandomAccessVectorValues floatVectors,
                                          PQVectors pqVectors) throws IOException {

        int nSubVectors = floatVectors.dimension() == 2 ? 1 : 2;
        var nvq = NVQuantization.compute(floatVectors, nSubVectors);
        var pq = pqVectors.getCompressor();

        // Create features: NVQ + FUSED_ADC
        var nvqFeature = new NVQ(nvq);
        var fusedAdcFeature = new FusedADC(graph.maxDegree(), pq);

        // Build suppliers for inline features (NVQ only - FUSED_ADC needs neighbors)
        Map<FeatureId, IntFunction<Feature.State>> inlineSuppliers = new EnumMap<>(FeatureId.class);
        inlineSuppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(floatVectors.getVector(ordinal))));

        // FUSED_ADC supplier needs graph view, provided at write time
        var identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);

        // Sequential write
        System.out.printf("Writing with NVQ + FUSED_ADC features...%n");
        long sequentialStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, sequentialPath)
                .withParallelWrites(false)
                .with(nvqFeature)
                .with(fusedAdcFeature)
                .withMapper(identityMapper)
                .build()) {

            var view = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
            writeSuppliers.put(FeatureId.NVQ_VECTORS, inlineSuppliers.get(FeatureId.NVQ_VECTORS));
            writeSuppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(view, pqVectors, ordinal));

            writer.write(writeSuppliers);
            view.close();
        }
        long sequentialTime = System.nanoTime() - sequentialStart;
        System.out.printf("Sequential write: %.2f ms%n", sequentialTime / 1_000_000.0);

        // Parallel write
        long parallelStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, parallelPath)
                .withParallelWrites(true)
                .with(nvqFeature)
                .with(fusedAdcFeature)
                .withMapper(identityMapper)
                .build()) {

            var view = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
            writeSuppliers.put(FeatureId.NVQ_VECTORS, inlineSuppliers.get(FeatureId.NVQ_VECTORS));
            writeSuppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(view, pqVectors, ordinal));

            writer.write(writeSuppliers);
            view.close();
        }
        long parallelTime = System.nanoTime() - parallelStart;

        System.out.printf("Parallel write:   %.2f ms%n", parallelTime / 1_000_000.0);
        System.out.printf("Speedup:          %.2fx%n", (double) sequentialTime / parallelTime);
    }

    /**
     * Main method to run a benchmark test of sequential vs parallel writes.
     *
     * Usage: java ParallelWriteExample [dataset-name]
     *
     * Example: java ParallelWriteExample cohere-english-v3-100k
     *
     * If no dataset is provided, uses "cohere-english-v3-100k" by default.
     */
    public static void main(String[] args) throws IOException {
        String datasetName = args.length > 0 ? args[0] : "cohere-english-v3-100k";

        System.out.println("Loading dataset: " + datasetName);
        DataSet ds = DataSetLoader.loadDataSet(datasetName);
        System.out.printf("Loaded %d vectors of dimension %d%n", ds.baseVectors.size(), ds.getDimension());

        var floatVectors = ds.getBaseRavv();

        // Build PQ compression (matching Grid.buildOnDisk pattern)
        System.out.println("Computing PQ compression...");
        int pqM = floatVectors.dimension() / 8; // m = dimension / 8
        boolean centerData = ds.similarityFunction == io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
        var pq = ProductQuantization.compute(floatVectors, pqM, 256, centerData, UNWEIGHTED);
        var pqVectors = (PQVectors) pq.encodeAll(floatVectors);
        System.out.printf("PQ compression: %d subspaces, 256 clusters%n", pqM);

        // Build graph parameters (matching typical benchmark settings)
        int M = 32;
        int efConstruction = 100;
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = false;
        boolean refineFinalGraph = true;

        System.out.printf("Building graph with PQ-compressed vectors (M=%d, efConstruction=%d)...%n", M, efConstruction);
        long buildStart = System.nanoTime();

        var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.similarityFunction, pqVectors);
        var builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph);

        // Add all vectors to the graph
        for (int i = 0; i < floatVectors.size(); i++) {
            builder.addGraphNode(i, floatVectors.getVector(i));
        }
        builder.cleanup();

        var graph = builder.getGraph();
        long buildTime = System.nanoTime() - buildStart;
        System.out.printf("Graph built in %.2fs%n", buildTime / 1_000_000_000.0);
        System.out.printf("Graph has %d nodes%n", graph.size(0));

        // Create temporary paths for writing
        Path tempDir = Files.createTempDirectory("parallel-write-test");
        Path sequentialPath = tempDir.resolve("graph-sequential");
        Path parallelPath = tempDir.resolve("graph-parallel");

        try {
            System.out.println("\n=== Testing Write Performance ===");

            // Run benchmark comparison
            benchmarkComparison(graph, sequentialPath, parallelPath, floatVectors, pqVectors);

            // Report file sizes
            long seqSize = Files.size(sequentialPath);
            long parSize = Files.size(parallelPath);
            System.out.printf("%nFile sizes: Sequential=%.2f MB, Parallel=%.2f MB%n",
                    seqSize / 1024.0 / 1024.0,
                    parSize / 1024.0 / 1024.0);

        } finally {
            // Cleanup
            builder.close();
            Files.deleteIfExists(sequentialPath);
            Files.deleteIfExists(parallelPath);
            Files.deleteIfExists(tempDir);
        }

        System.out.println("\nTest complete!");
    }
}
