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
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * JMH benchmark that mirrors the ParallelWriteExample: it builds a graph from vectors, then
 * writes the graph to disk sequentially and in parallel using NVQ + FUSED_PQ features,
 * and verifies that the outputs are identical.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=false"})
@Warmup(iterations = 1)
@Measurement(iterations = 2)
@Threads(1)
public class ParallelWriteBenchmark {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    
    @Param({"100000"})
    int numBaseVectors;

    @Param({"1024"})
    int dimension;

    @Param({"true", "false"})
    boolean addHierarchy;

    // Graph build parameters
    final int M = 32;
    final int efConstruction = 100;
    final float neighborOverflow = 1.2f;
    final float alpha = 1.2f;
    //final boolean addHierarchy = false;
    final boolean refineFinalGraph = true;

    // Dataset and index state
    private RandomAccessVectorValues floatVectors;
    private PQVectors pqVectors;
    private ImmutableGraphIndex graph;

    // Feature state reused between iterations
    private NVQ nvqFeature;
    private FusedPQ fusedPQFeature;
    private OrdinalMapper identityMapper;
    private Map<FeatureId, IntFunction<Feature.State>> inlineSuppliers;

    // Paths
    private Path tempDir;
    private final AtomicInteger fileCounter = new AtomicInteger();

    @Setup(Level.Trial)
    public void setup() throws IOException {
        // Generate random vectors
        final var baseVectors = new ArrayList<VectorFloat<?>>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            baseVectors.add(createRandomVector(dimension));
        }
        floatVectors = new ListRandomAccessVectorValues(baseVectors, dimension);

        // Compute PQ compression
        final int pqM = Math.max(1, dimension / 8);
        final boolean centerData = true; // for EUCLIDEAN
        final var pq = ProductQuantization.compute(floatVectors, pqM, 256, centerData, UNWEIGHTED);
        pqVectors = (PQVectors) pq.encodeAll(floatVectors);

        // Build graph using PQ build score provider
        final var bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);
        try (var builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph)) {
            graph = builder.build(floatVectors);
        }

        // Prepare features
        int nSubVectors = floatVectors.dimension() == 2 ? 1 : 2;
        var nvq = NVQuantization.compute(floatVectors, nSubVectors);
        nvqFeature = new NVQ(nvq);
        fusedPQFeature = new FusedPQ(graph.maxDegree(), pqVectors.getCompressor());

        inlineSuppliers = new EnumMap<>(FeatureId.class);
        inlineSuppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(floatVectors.getVector(ordinal))));

        identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);

        // Temp directory for outputs
        tempDir = Files.createTempDirectory("parallel-write-bench");
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (tempDir != null) {
            // Best-effort cleanup of all files created
            try (var stream = Files.list(tempDir)) {
                stream.forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (IOException ignored) {}
                });
            }
            Files.deleteIfExists(tempDir);
        }
    }

    @Benchmark
    public void writeSequentialThenParallelAndVerify(Blackhole blackhole) throws IOException {
        // Unique output files per invocation
        int idx = fileCounter.getAndIncrement();
        Path sequentialPath = tempDir.resolve("graph-sequential-" + idx);
        Path parallelPath = tempDir.resolve("graph-parallel-" + idx);

        long startSeq = System.nanoTime();
        writeGraph(graph, sequentialPath, false);
        long seqTime = System.nanoTime() - startSeq;

        long startPar = System.nanoTime();
        writeGraph(graph, parallelPath, true);
        long parTime = System.nanoTime() - startPar;

        // Report times and speedup for this invocation
        double seqMs = seqTime / 1_000_000.0;
        double parMs = parTime / 1_000_000.0;
        double speedup = parTime == 0 ? Double.NaN : seqTime / (double) parTime;
        System.out.printf("Sequential write: %.2f ms, Parallel write: %.2f ms, Speedup: %.2fx%n", seqMs, parMs, speedup);

        // Load and verify identical
        OnDiskGraphIndex sequentialIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(sequentialPath));
        OnDiskGraphIndex parallelIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(parallelPath));
        try {
            verifyIndicesIdentical(sequentialIndex, parallelIndex);
        } finally {
            sequentialIndex.close();
            parallelIndex.close();
        }

        // Consume sizes to prevent DCE
        blackhole.consume(Files.size(sequentialPath));
        blackhole.consume(Files.size(parallelPath));

        // Cleanup files after each invocation to limit disk usage
        Files.deleteIfExists(sequentialPath);
        Files.deleteIfExists(parallelPath);
    }

    private void writeGraph(ImmutableGraphIndex graph,
                            Path path,
                            boolean parallel) throws IOException {
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                .withParallelWrites(parallel)
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
    }

    private static void verifyIndicesIdentical(OnDiskGraphIndex index1, OnDiskGraphIndex index2) throws IOException {
        // Basic properties
        if (index1.getMaxLevel() != index2.getMaxLevel()) {
            throw new AssertionError("Max levels differ: " + index1.getMaxLevel() + " vs " + index2.getMaxLevel());
        }
        if (index1.getIdUpperBound() != index2.getIdUpperBound()) {
            throw new AssertionError("ID upper bounds differ: " + index1.getIdUpperBound() + " vs " + index2.getIdUpperBound());
        }
        if (!index1.getFeatureSet().equals(index2.getFeatureSet())) {
            throw new AssertionError("Feature sets differ: " + index1.getFeatureSet() + " vs " + index2.getFeatureSet());
        }

        try (var view1 = index1.getView(); var view2 = index2.getView()) {
            if (!view1.entryNode().equals(view2.entryNode())) {
                throw new AssertionError("Entry nodes differ: " + view1.entryNode() + " vs " + view2.entryNode());
            }
            for (int level = 0; level <= index1.getMaxLevel(); level++) {
                if (index1.size(level) != index2.size(level)) {
                    throw new AssertionError("Layer " + level + " sizes differ: " + index1.size(level) + " vs " + index2.size(level));
                }
                if (index1.getDegree(level) != index2.getDegree(level)) {
                    throw new AssertionError("Layer " + level + " degrees differ: " + index1.getDegree(level) + " vs " + index2.getDegree(level));
                }

                // Collect node IDs in arrays
                java.util.List<Integer> nodeList1 = new java.util.ArrayList<>();
                java.util.List<Integer> nodeList2 = new java.util.ArrayList<>();
                NodesIterator nodes1 = index1.getNodes(level);
                while (nodes1.hasNext()) nodeList1.add(nodes1.nextInt());
                NodesIterator nodes2 = index2.getNodes(level);
                while (nodes2.hasNext()) nodeList2.add(nodes2.nextInt());
                if (!nodeList1.equals(nodeList2)) {
                    throw new AssertionError("Layer " + level + " has different node sets");
                }

                // Compare neighbors
                for (int nodeId : nodeList1) {
                    NodesIterator neighbors1 = view1.getNeighborsIterator(level, nodeId);
                    NodesIterator neighbors2 = view2.getNeighborsIterator(level, nodeId);
                    if (neighbors1.size() != neighbors2.size()) {
                        throw new AssertionError("Layer " + level + " node " + nodeId + " neighbor counts differ: " + neighbors1.size() + " vs " + neighbors2.size());
                    }
                    int[] n1 = new int[neighbors1.size()];
                    int[] n2 = new int[neighbors2.size()];
                    for (int i = 0; i < n1.length; i++) {
                        n1[i] = neighbors1.nextInt();
                        n2[i] = neighbors2.nextInt();
                    }
                    if (!Arrays.equals(n1, n2)) {
                        throw new AssertionError("Layer " + level + " node " + nodeId + " has different neighbor sets");
                    }
                }
            }

            // Optional vector checks (layer 0)
            if (index1.getFeatureSet().contains(FeatureId.INLINE_VECTORS) ||
                index1.getFeatureSet().contains(FeatureId.NVQ_VECTORS)) {
                int vectorsChecked = 0;
                int maxToCheck = Math.min(100, index1.size(0));
                NodesIterator nodes = index1.getNodes(0);
                while (nodes.hasNext() && vectorsChecked < maxToCheck) {
                    int node = nodes.nextInt();
                    if (index1.getFeatureSet().contains(FeatureId.INLINE_VECTORS)) {
                        var vec1 = view1.getVector(node);
                        var vec2 = view2.getVector(node);
                        if (!vec1.equals(vec2)) {
                            throw new AssertionError("Node " + node + " vectors differ");
                        }
                    }
                    vectorsChecked++;
                }
            }
        }
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}
