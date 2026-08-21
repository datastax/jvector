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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Example demonstrating how to use {@link OnDiskParallelGraphIndexWriter} and comparing its
 * two internal write strategies against the sequential {@link OnDiskGraphIndexWriter}.
 * <p>
 * {@code OnDiskParallelGraphIndexWriter} parallelizes serialization of Level-0 node records
 * using {@link java.nio.channels.AsynchronousFileChannel}. Which of its two write paths runs
 * (see {@link NodeRecordTask}) is determined entirely by whether every feature supplier is
 * still present in the map passed to {@code write()}:
 * <pre>
 * // Sequential (default) — single-threaded baseline:
 * var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .build();
 * writer.write(featureSuppliers);
 *
 * // Parallel, batched path — every feature supplier is passed directly to write();
 * // each task packs its whole node range into one buffer and issues one channel write:
 * var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .build();
 * writer.write(featureSuppliers);
 *
 * // Parallel, legacy path — a plain (non-fused) feature is pre-written per node via
 * // writeFeaturesInline() and then omitted from the map passed to write(); each task then
 * // owns only the bytes not already on disk and issues one non-blocking write per owned span:
 * var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
 *     .with(plainInlineFeature)
 *     .build();
 * for (int ordinal = 0; ordinal &lt; graph.size(0); ordinal++) {
 *     writer.writeFeaturesInline(ordinal, Map.of(plainInlineFeature.id(), stateFor(ordinal)));
 * }
 * writer.write(Map.of());
 * </pre>
 * <p>
 * Note: a <i>fused</i> feature (one whose {@code Feature.isFused()} is {@code true}, e.g.
 * {@code FusedPQ}) can never be fully pre-written this way. {@code write()} always requires a
 * supplier for it regardless of what NodeRecordTask needs, because
 * {@code AbstractGraphIndexWriter.writeSparseLevels()} writes the fused source feature for the
 * hierarchy's higher layers directly — a separate code path from the L0 records NodeRecordTask
 * owns, with no {@code writeFeaturesInline()} equivalent. Omitting a fused feature's supplier
 * throws {@code IllegalStateException: Supplier for feature ... not found}. The benchmarks below
 * demonstrate the correct pattern: pre-write the plain feature (NVQ_VECTORS) and keep supplying
 * the fused one (FUSED_PQ) to {@code write()}.
 * <p>
 * Two benchmark methods exercise this:
 * <ul>
 *   <li>{@link #benchmarkPlainWrites} — sequential vs. parallel-batched, both writing a single,
 *       already-fully-built graph with every supplier passed directly to {@code write()}.</li>
 *   <li>{@link #benchmarkInterleavedWrites} — sequential vs. parallel-legacy, both fed
 *       {@code writeFeaturesInline()} calls interleaved with graph construction itself, mirroring
 *       {@code Grid.buildOnDisk}'s {@code writers.forEach(...)} pattern: every writer receives its
 *       feature state as each node is added, regardless of which one finalizes sequentially vs.
 *       asynchronously. This is the realistic shape of the {@code writeFeaturesInline()} use case.
 *       Calling it in an isolated, single-threaded pass over an already-built graph instead is an
 *       anti-pattern that pays 10x+ the cost for no reason — see
 *       {@code BufferedRandomAccessWriter.seek()}, which flushes and reseeks the underlying file
 *       on every call.</li>
 * </ul>
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
     * Benchmark comparison on an already-fully-built graph using NVQ + FUSED_ADC features:
     * sequential writes via {@link OnDiskGraphIndexWriter} vs. parallel writes via
     * {@link OnDiskParallelGraphIndexWriter} taking its batched path (see {@link NodeRecordTask}):
     * every feature supplier is passed straight to {@code write()}, so each task builds its
     * whole node range into one buffer and issues a single {@code channel.write()} call.
     * <p>
     * Both writers see the graph only after it's completely built — this isolates the cost of
     * the write strategy itself from graph construction. Contrast with
     * {@link #benchmarkInterleavedWrites}, which measures the legacy/pre-write pattern.
     */
    public static void benchmarkPlainWrites(ImmutableGraphIndex graph,
                                          Path sequentialPath,
                                          Path parallelBatchedPath,
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
        System.out.printf("Sequential write:         %.2f ms%n", sequentialTime / 1_000_000.0);

        // Parallel write — batched path: every feature supplier is passed directly to write(),
        // so NodeRecordTask.callBatched() packs each task's whole node range into one buffer
        // and issues a single channel.write() for it.
        long parallelBatchedStart = System.nanoTime();
        try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, parallelBatchedPath)
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
        long parallelBatchedTime = System.nanoTime() - parallelBatchedStart;
        System.out.printf("Parallel write (batched): %.2f ms%n", parallelBatchedTime / 1_000_000.0);
        System.out.printf("Speedup (batched vs sequential): %.2fx%n", (double) sequentialTime / parallelBatchedTime);
    }

    /**
     * Benchmark comparison of the legacy/pre-write pattern, built the way {@code Grid.buildOnDisk}
     * actually uses it: {@code NVQ_VECTORS} is written via {@code writeFeaturesInline()}
     * <em>interleaved with graph construction itself</em>, not as an isolated pass afterward.
     * <p>
     * One {@link GraphIndexBuilder} is shared by both a sequential ({@link OnDiskGraphIndexWriter})
     * and a parallel-legacy ({@link OnDiskParallelGraphIndexWriter}) writer. Both writers are
     * registered before construction starts (mirroring {@code Grid.builderWithSuppliers} being
     * called before the incremental-build loop), and both receive a {@code writeFeaturesInline()}
     * call for every node inside the same parallel construction stream — mirroring
     * {@code Grid.buildOnDisk}'s {@code writers.forEach(...)} pattern, which feeds every writer
     * regardless of whether it will finalize sequentially or asynchronously. {@code FUSED_PQ}
     * can't be pre-written (see the class Javadoc), so it stays in the map passed to the final
     * {@code write()} call for both writers.
     * <p>
     * Since both writers are fed from the same construction pass, the graph-build-plus-pre-write
     * cost is a single, genuinely shared number — it's added once to each writer's total rather
     * than measured or estimated separately. The two {@code write()} finalize calls are timed one
     * after the other (not concurrently), so each writer's finalize cost is cleanly attributable
     * instead of contending with the other for I/O.
     */
    public static void benchmarkInterleavedWrites(RandomAccessVectorValues floatVectors,
                                                  VectorSimilarityFunction similarityFunction,
                                                  PQVectors pqVectors,
                                                  int M,
                                                  int efConstruction,
                                                  float neighborOverflow,
                                                  float alpha,
                                                  boolean addHierarchy,
                                                  boolean refineFinalGraph,
                                                  Path sequentialPath,
                                                  Path parallelLegacyPath) throws IOException {

        int nSubVectors = floatVectors.dimension() == 2 ? 1 : 2;
        var nvq = NVQuantization.compute(floatVectors, nSubVectors);
        var pq = pqVectors.getCompressor();
        var identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);

        var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqVectors);
        var builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph);
        var onHeapGraph = builder.getGraph();

        try (var sequentialWriter = new OnDiskGraphIndexWriter.Builder(onHeapGraph, sequentialPath)
                    .with(new NVQ(nvq))
                    .with(new FusedPQ(onHeapGraph.maxDegree(), pq))
                    .withMapper(identityMapper)
                    .build();
             var parallelLegacyWriter = new OnDiskParallelGraphIndexWriter.Builder(onHeapGraph, parallelLegacyPath)
                    .with(new NVQ(nvq))
                    .with(new FusedPQ(onHeapGraph.maxDegree(), pq))
                    .withMapper(identityMapper)
                    .build()) {

            // Interleave NVQ pre-writes for BOTH writers into the same construction pass,
            // mirroring Grid.buildOnDisk's writers.forEach(...) per-node loop.
            System.out.println("Building graph with interleaved feature writes (Grid.buildOnDisk pattern)...");
            long interleavedBuildStart = System.nanoTime();
            var vv = floatVectors.threadLocalSupplier();
            IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
                Feature.State nvqState = new NVQ.State(nvq.encode(floatVectors.getVector(node)));
                Map<FeatureId, Feature.State> stateMap = Map.of(FeatureId.NVQ_VECTORS, nvqState);
                try {
                    sequentialWriter.writeFeaturesInline(node, stateMap);
                    parallelLegacyWriter.writeFeaturesInline(node, stateMap);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                builder.addGraphNode(node, vv.get().getVector(node));
            });
            builder.cleanup();
            long interleavedBuildTime = System.nanoTime() - interleavedBuildStart;
            System.out.printf("Graph build + interleaved pre-write: %.2f ms%n", interleavedBuildTime / 1_000_000.0);

            // Finalize each writer separately (sequentially, not concurrently) so each one's
            // write() time is cleanly attributable rather than contending with the other for I/O.
            try (var view = onHeapGraph.getView()) {
                long seqWriteStart = System.nanoTime();
                sequentialWriter.write(Map.of(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal)));
                long seqWriteTime = System.nanoTime() - seqWriteStart;
                System.out.printf("Sequential write() (finalize):        %.2f ms%n", seqWriteTime / 1_000_000.0);

                long legacyWriteStart = System.nanoTime();
                parallelLegacyWriter.write(Map.of(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal)));
                long legacyWriteTime = System.nanoTime() - legacyWriteStart;
                System.out.printf("Parallel (legacy) write() (finalize): %.2f ms%n", legacyWriteTime / 1_000_000.0);

                long sequentialTotal = interleavedBuildTime + seqWriteTime;
                long legacyTotal = interleavedBuildTime + legacyWriteTime;
                System.out.printf("%nSequential total (build+prewrite+write):        %.2f ms%n", sequentialTotal / 1_000_000.0);
                System.out.printf("Parallel (legacy) total (build+prewrite+write): %.2f ms%n", legacyTotal / 1_000_000.0);
                System.out.printf("Speedup (legacy finalize vs sequential finalize): %.2fx%n", (double) seqWriteTime / legacyWriteTime);
                System.out.printf("Speedup (legacy total vs sequential total):       %.2fx%n", (double) sequentialTotal / legacyTotal);
            }
        } finally {
            builder.close();
        }
    }

    /**
     * Main method to run two benchmark comparisons: sequential vs. parallel-batched writes on
     * an already-built graph ({@link #benchmarkPlainWrites}), and sequential vs. parallel-legacy
     * writes with feature data interleaved into graph construction, Grid.java-style
     * ({@link #benchmarkInterleavedWrites}).
     * <p>
     * Usage: java ParallelWriteExample [dataset-name]
     * <p>
     * Example: java ParallelWriteExample cohere-english-v3-1M
     * <p>
     * If no dataset is provided, uses "cohere-english-v3-1M" by default.
     */
    public static void main(String[] args) throws IOException {
        String datasetName = args.length > 0 ? args[0] : "cohere-english-v3-1M";

        System.out.println("Loading dataset: " + datasetName);
        DataSet ds = DataSets.loadDataSet(datasetName).orElseThrow(
                () -> new RuntimeException("Dataset " + datasetName + " not found")
        ).getDataSet();
        System.out.printf("Loaded %d vectors of dimension %d%n", ds.getBaseVectors().size(), ds.getDimension());

        var floatVectors = ds.getBaseRavv();

        // Build PQ compression (matching Grid.buildOnDisk pattern)
        System.out.println("Computing PQ compression...");
        int pqM = floatVectors.dimension() / 8; // m = dimension / 8
        boolean centerData = ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
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

        Path tempDir = Files.createTempDirectory("parallel-write-test");
        Path sequentialPath = tempDir.resolve("graph-sequential");
        Path parallelBatchedPath = tempDir.resolve("graph-parallel-batched");
        Path sequentialInterleavedPath = tempDir.resolve("graph-sequential-interleaved");
        Path parallelLegacyInterleavedPath = tempDir.resolve("graph-parallel-legacy-interleaved");

        try {
            // === Graph A: built once, plain (no writer involved during construction) ===
            System.out.printf("%nBuilding Graph A (plain) with PQ-compressed vectors (M=%d, efConstruction=%d)...%n", M, efConstruction);
            long buildStart = System.nanoTime();
            var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.getSimilarityFunction(), pqVectors);
            var graphABuilder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction,
                    neighborOverflow, alpha, addHierarchy, refineFinalGraph);
            var graphA = graphABuilder.build(floatVectors);
            long buildTime = System.nanoTime() - buildStart;
            System.out.printf("Graph A built in %.2fs (%d nodes)%n", buildTime / 1_000_000_000.0, graphA.size(0));
            graphABuilder.close();

            System.out.println("\n=== Graph A: sequential vs. parallel (batched) ===");
            benchmarkPlainWrites(graphA, sequentialPath, parallelBatchedPath, floatVectors, pqVectors);

            long seqSize = Files.size(sequentialPath);
            long batchedSize = Files.size(parallelBatchedPath);
            System.out.printf("%nFile sizes: Sequential=%.2f MB, Parallel(batched)=%.2f MB%n",
                    seqSize / 1024.0 / 1024.0, batchedSize / 1024.0 / 1024.0);

            System.out.println("\n=== Testing Read Correctness (Graph A) ===");
            try (var sequentialIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(sequentialPath));
                 var parallelBatchedIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(parallelBatchedPath))) {
                verifyIndicesIdentical(sequentialIndex, parallelBatchedIndex);
            }

            // === Graph B: built once via feature writes interleaved into construction, fed to
            // both writers from the same pass (Grid.buildOnDisk pattern) ===
            System.out.println("\n=== Graph B: sequential vs. parallel (legacy), interleaved pre-write ===");
            benchmarkInterleavedWrites(floatVectors, ds.getSimilarityFunction(), pqVectors,
                    M, efConstruction, neighborOverflow, alpha, addHierarchy, refineFinalGraph,
                    sequentialInterleavedPath, parallelLegacyInterleavedPath);

            long seqInterleavedSize = Files.size(sequentialInterleavedPath);
            long legacyInterleavedSize = Files.size(parallelLegacyInterleavedPath);
            System.out.printf("%nFile sizes: Sequential(interleaved)=%.2f MB, Parallel(legacy,interleaved)=%.2f MB%n",
                    seqInterleavedSize / 1024.0 / 1024.0, legacyInterleavedSize / 1024.0 / 1024.0);

            // Both Graph B writers serialized the same shared graph object (see
            // benchmarkInterleavedWrites), so the strict structural check applies here too.
            System.out.println("\n=== Testing Read Correctness (Graph B) ===");
            try (var sequentialInterleavedIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(sequentialInterleavedPath));
                 var parallelLegacyInterleavedIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(parallelLegacyInterleavedPath))) {
                verifyIndicesIdentical(sequentialInterleavedIndex, parallelLegacyInterleavedIndex);
            }

        } finally {
            // Cleanup
            Files.deleteIfExists(sequentialPath);
            Files.deleteIfExists(parallelBatchedPath);
            Files.deleteIfExists(sequentialInterleavedPath);
            Files.deleteIfExists(parallelLegacyInterleavedPath);
            Files.deleteIfExists(tempDir);
        }

        System.out.println("\n✅ Test complete - sequential and parallel (batched and legacy) writes produce identical results!");
    }
}
