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
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

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
     * Example: Benchmark comparison between sequential and parallel writes.
     */
    public static void benchmarkComparison(ImmutableGraphIndex graph,
                                          Path sequentialPath,
                                          Path parallelPath,
                                          RandomAccessVectorValues floatVectors) throws IOException {

        var inlineVectors = new InlineVectors(floatVectors.dimension());
        Map<FeatureId, IntFunction<Feature.State>> suppliers =
            Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                ordinal -> new InlineVectors.State(floatVectors.getVector(ordinal))
            );

        // Sequential write
        long sequentialStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, sequentialPath)
                .withParallelWrites(false)
                .with(inlineVectors)
                .build()) {
            writer.write(suppliers);
        }
        long sequentialTime = System.nanoTime() - sequentialStart;
        System.out.printf("Sequential write: %.2f ms%n", sequentialTime / 1_000_000.0);

        // Parallel write
        long parallelStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, parallelPath)
                .withParallelWrites(true)
                .with(inlineVectors)
                .build()) {
            writer.write(suppliers);
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

        // Build graph parameters
        int M = 16;  // Max connections per node
        int efConstruction = 100;  // Size of dynamic candidate list during construction
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;  // Diversity pruning parameter
        boolean addHierarchy = false;  // Don't add HNSW hierarchy for simplicity
        boolean refineFinalGraph = true;  // Refine graph after construction

        System.out.printf("Building graph (M=%d, efConstruction=%d)...%n", M, efConstruction);
        long buildStart = System.nanoTime();

        var floatVectors = ds.getBaseRavv();
        var scoreProvider = BuildScoreProvider.randomAccessScoreProvider(floatVectors, ds.similarityFunction);
        var builder = new GraphIndexBuilder(scoreProvider, floatVectors.dimension(), M, efConstruction, neighborOverflow, alpha, addHierarchy, refineFinalGraph);

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
            benchmarkComparison(graph, sequentialPath, parallelPath, floatVectors);

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
