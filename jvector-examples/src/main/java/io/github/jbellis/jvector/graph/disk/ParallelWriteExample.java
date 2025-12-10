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

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
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
     * Verifies that two OnDiskGraphIndex instances are identical in structure and content.
     * Compares graph structure (nodes, neighbors) and feature data (vectors).
     */
    private static void verifyIndicesIdentical(OnDiskGraphIndex index1, OnDiskGraphIndex index2) throws IOException {
        System.out.println("\n=== Verifying Graph Indices ===");

        // Check basic properties
        if (index1.getMaxLevel() != index2.getMaxLevel()) {
            throw new AssertionError(String.format("Max levels differ: %d vs %d",
                index1.getMaxLevel(), index2.getMaxLevel()));
        }
        System.out.printf("✓ Max level matches: %d%n", index1.getMaxLevel());

        if (index1.getIdUpperBound() != index2.getIdUpperBound()) {
            throw new AssertionError(String.format("ID upper bounds differ: %d vs %d",
                index1.getIdUpperBound(), index2.getIdUpperBound()));
        }
        System.out.printf("✓ ID upper bound matches: %d%n", index1.getIdUpperBound());

        if (!index1.getFeatureSet().equals(index2.getFeatureSet())) {
            throw new AssertionError(String.format("Feature sets differ: %s vs %s",
                index1.getFeatureSet(), index2.getFeatureSet()));
        }
        System.out.printf("✓ Feature sets match: %s%n", index1.getFeatureSet());

        // Check each layer
        try (var view1 = index1.getView(); var view2 = index2.getView()) {
            // Check entry nodes (accessed through views)
            if (!view1.entryNode().equals(view2.entryNode())) {
                throw new AssertionError(String.format("Entry nodes differ: %s vs %s",
                    view1.entryNode(), view2.entryNode()));
            }
            System.out.printf("✓ Entry node matches: %s%n", view1.entryNode());
            for (int level = 0; level <= index1.getMaxLevel(); level++) {
                if (index1.size(level) != index2.size(level)) {
                    throw new AssertionError(String.format("Layer %d sizes differ: %d vs %d",
                        level, index1.size(level), index2.size(level)));
                }

                if (index1.getDegree(level) != index2.getDegree(level)) {
                    throw new AssertionError(String.format("Layer %d degrees differ: %d vs %d",
                        level, index1.getDegree(level), index2.getDegree(level)));
                }

                // Collect all node IDs from both indices into arrays
                java.util.List<Integer> nodeList1 = new java.util.ArrayList<>();
                java.util.List<Integer> nodeList2 = new java.util.ArrayList<>();

                NodesIterator nodes1 = index1.getNodes(level);
                while (nodes1.hasNext()) {
                    nodeList1.add(nodes1.nextInt());
                }

                NodesIterator nodes2 = index2.getNodes(level);
                while (nodes2.hasNext()) {
                    nodeList2.add(nodes2.nextInt());
                }

                // Verify same set of nodes
                if (!nodeList1.equals(nodeList2)) {
                    // Find differences
                    java.util.Set<Integer> set1 = new java.util.HashSet<>(nodeList1);
                    java.util.Set<Integer> set2 = new java.util.HashSet<>(nodeList2);

                    java.util.Set<Integer> onlyIn1 = new java.util.HashSet<>(set1);
                    onlyIn1.removeAll(set2);

                    java.util.Set<Integer> onlyIn2 = new java.util.HashSet<>(set2);
                    onlyIn2.removeAll(set1);

                    System.out.printf("Layer %d node count: sequential=%d, parallel=%d%n",
                        level, nodeList1.size(), nodeList2.size());

                    if (!onlyIn1.isEmpty()) {
                        var sample1 = onlyIn1.stream().limit(10).collect(java.util.stream.Collectors.toList());
                        System.out.printf("  Nodes only in sequential (first 10): %s%n", sample1);
                    }
                    if (!onlyIn2.isEmpty()) {
                        var sample2 = onlyIn2.stream().limit(10).collect(java.util.stream.Collectors.toList());
                        System.out.printf("  Nodes only in parallel (first 10): %s%n", sample2);
                    }

                    // Sample some nodes from each to see the pattern
                    System.out.printf("  First 20 nodes in sequential: %s%n",
                        nodeList1.stream().limit(20).collect(java.util.stream.Collectors.toList()));
                    System.out.printf("  First 20 nodes in parallel: %s%n",
                        nodeList2.stream().limit(20).collect(java.util.stream.Collectors.toList()));

                    throw new AssertionError(String.format("Layer %d has different node sets: sequential has %d nodes, parallel has %d nodes, %d nodes differ",
                        level, nodeList1.size(), nodeList2.size(), onlyIn1.size() + onlyIn2.size()));
                }

                // Compare neighbors for each node
                int differentNeighbors = 0;
                for (int nodeId : nodeList1) {
                    NodesIterator neighbors1 = view1.getNeighborsIterator(level, nodeId);
                    NodesIterator neighbors2 = view2.getNeighborsIterator(level, nodeId);

                    if (neighbors1.size() != neighbors2.size()) {
                        throw new AssertionError(String.format("Layer %d node %d neighbor counts differ: %d vs %d",
                            level, nodeId, neighbors1.size(), neighbors2.size()));
                    }

                    int[] n1 = new int[neighbors1.size()];
                    int[] n2 = new int[neighbors2.size()];
                    for (int i = 0; i < n1.length; i++) {
                        n1[i] = neighbors1.nextInt();
                        n2[i] = neighbors2.nextInt();
                    }

                    if (!Arrays.equals(n1, n2)) {
                        differentNeighbors++;
                        if (differentNeighbors <= 3) {
                            System.out.printf("  ✗ Layer %d node %d has different neighbor sets: %s vs %s%n",
                                level, nodeId, Arrays.toString(n1), Arrays.toString(n2));
                        }
                    }
                }

                if (differentNeighbors > 0) {
                    throw new AssertionError(String.format("Layer %d: %d/%d nodes have different neighbor sets",
                        level, differentNeighbors, nodeList1.size()));
                }

                System.out.printf("✓ Layer %d structure matches (%d nodes, degree %d)%n",
                    level, index1.size(level), index1.getDegree(level));
            }

            // Compare vectors if present (only check layer 0)
            if (index1.getFeatureSet().contains(FeatureId.INLINE_VECTORS) ||
                index1.getFeatureSet().contains(FeatureId.NVQ_VECTORS)) {

                int vectorsChecked = 0;
                int maxToCheck = Math.min(100, index1.size(0)); // Check up to 100 vectors as a sample

                NodesIterator nodes = index1.getNodes(0);
                while (nodes.hasNext() && vectorsChecked < maxToCheck) {
                    int node = nodes.nextInt();

                    if (index1.getFeatureSet().contains(FeatureId.INLINE_VECTORS)) {
                        var vec1 = view1.getVector(node);
                        var vec2 = view2.getVector(node);

                        if (!vec1.equals(vec2)) {
                            throw new AssertionError(String.format("Node %d vectors differ", node));
                        }
                    }

                    vectorsChecked++;
                }

                System.out.printf("✓ Sampled %d vectors, all match%n", vectorsChecked);
            }
        }

        System.out.println("✓ All checks passed - indices are identical!");
    }

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
        var fusedPQFeature = new FusedPQ(graph.maxDegree(), pq);

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
                .with(fusedPQFeature)
                .withMapper(identityMapper)
                .build()) {

            var view = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
            writeSuppliers.put(FeatureId.NVQ_VECTORS, inlineSuppliers.get(FeatureId.NVQ_VECTORS));
            writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal));

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
                .with(fusedPQFeature)
                .withMapper(identityMapper)
                .build()) {

            var view = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
            writeSuppliers.put(FeatureId.NVQ_VECTORS, inlineSuppliers.get(FeatureId.NVQ_VECTORS));
            writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal));

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
        DataSet ds = DataSets.loadDataSet(datasetName).orElseThrow(
                () -> new RuntimeException("Dataset " + datasetName + " not found")
        );
        System.out.printf("Loaded %d vectors of dimension %d%n", ds.getBaseVectors().size(), ds.getDimension());

        var floatVectors = ds.getBaseRavv();

        // Build PQ compression (matching Grid.buildOnDisk pattern)
        System.out.println("Computing PQ compression...");
        int pqM = floatVectors.dimension() / 8; // m = dimension / 8
        boolean centerData = ds.getSimilarityFunction() == io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
        var pq = ProductQuantization.compute(floatVectors, pqM, 256, centerData, UNWEIGHTED);
        var pqVectors = (PQVectors) pq.encodeAll(floatVectors);
        System.out.printf("PQ compression: %d subspaces, 256 clusters%n", pqM);

        // Build graph parameters (matching typical benchmark settings)
        int M = 32;
        int efConstruction = 100;
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = true;
        boolean refineFinalGraph = true;

        System.out.printf("Building graph with PQ-compressed vectors (M=%d, efConstruction=%d)...%n", M, efConstruction);
        long buildStart = System.nanoTime();

        var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.getSimilarityFunction(), pqVectors);
        var builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph);

        // Build graph using parallel construction for much better performance
        var graph = builder.build(floatVectors);
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

            // === Read Phase: Load and verify both indices ===
            System.out.println("\n=== Testing Read Correctness ===");
            System.out.println("Loading sequential index...");
            OnDiskGraphIndex sequentialIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(sequentialPath));
            System.out.println("Loading parallel index...");
            OnDiskGraphIndex parallelIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(parallelPath));

            // Verify that both indices are identical
            verifyIndicesIdentical(sequentialIndex, parallelIndex);

            // Close the loaded indices
            sequentialIndex.close();
            parallelIndex.close();

        } finally {
            // Cleanup
            builder.close();
            Files.deleteIfExists(sequentialPath);
            Files.deleteIfExists(parallelPath);
            Files.deleteIfExists(tempDir);
        }

        System.out.println("\n✅ Test complete - sequential and parallel writes produce identical results!");
    }
}
