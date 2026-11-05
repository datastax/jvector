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

import io.github.jbellis.jvector.disk.*;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.*;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import static org.junit.Assert.*;

public class TestBufferedGraphCompatibility {
    /// Every level's adjacency, one "level node: neighbors" line per node in node order.
    private static List<String> adjacency(ImmutableGraphIndex graph) throws Exception {
        var lines = new ArrayList<String>();
        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                var nodes = new ArrayList<Integer>();
                for (var it = graph.getNodes(level); it.hasNext(); ) nodes.add(it.nextInt());
                Collections.sort(nodes);
                for (int node : nodes) {
                    var neighbors = new StringBuilder();
                    for (var it = view.getNeighborsIterator(level, node); it.hasNext(); ) neighbors.append(it.nextInt()).append(' ');
                    lines.add(level + " " + node + ": " + neighbors);
                }
            }
        }
        return lines;
    }

    @Test(timeout = 180000)
    public void exactPqAndBinaryBuildsWithAllMetricsAndRefinementModes() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var pool = new ForkJoinPool(3);
        // One worker makes insertion order, level draws and cleanup order reproducible.
        var single = new ForkJoinPool(1);
        try {
            for (int dimension : new int[] {17, 768}) {
                int count = 257;
                var path = Files.createTempFile("graph-input", ".fvecs");
                var bytes = ByteBuffer.allocate(count * (dimension + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
                var vectors = new ArrayList<VectorFloat<?>>();
                var random = new Random(42);
                for (int i = 0; i < count; i++) {
                    var v = vts.createFloatVector(dimension);
                    for (int j = 0; j < dimension; j++) v.set(j, random.nextFloat() * 2 - 1);
                    VectorUtil.l2normalize(v);
                    bytes.putInt(dimension);
                    for (int j = 0; j < dimension; j++) bytes.putFloat(v.get(j));
                    vectors.add(v);
                }
                Files.write(path, bytes.array());
                var resident = new ListRandomAccessVectorValues(vectors, dimension);
                try (var source = FvecFileVectorValues.open(path)) {
                    var pq = ProductQuantization.compute(resident, Math.max(1, dimension / 8), 256, false,
                            io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED, pool, pool);
                    var pqv = pq.encodeAll(source, pool);
                    var bqv = (BQVectors) new BinaryQuantization(dimension).encodeAll(source, pool);
                    var nvq = NVQuantization.compute(source, 3);
                    for (var metric : VectorSimilarityFunction.values()) {
                        for (boolean refine : new boolean[] {false, true}) {
                            for (int mode = 0; mode < 3; mode++) {
                                var bsp = mode == 0 ? BuildScoreProvider.randomAccessScoreProvider(source, metric)
                                        : mode == 1 ? BuildScoreProvider.pqBuildScoreProvider(metric, pqv)
                                        : BuildScoreProvider.bqBuildScoreProvider(bqv);
                                var residentBsp = mode == 0 ? BuildScoreProvider.randomAccessScoreProvider(resident, metric) : bsp;
                                List<String> residentAdjacency;
                                try (var reference = new GraphIndexBuilder(residentBsp, dimension, 8, 64, 1.2f, 1.2f,
                                        true, refine, single, single)) {
                                    residentAdjacency = adjacency(reference.build(resident));
                                }
                                try (var builder = new GraphIndexBuilder(bsp, dimension, 8, 64, 1.2f, 1.2f,
                                        true, refine, single, single)) {
                                    var graph = builder.build(source);
                                    assertEquals(count, graph.size(0));
                                    // Deterministic builds: the file-backed source must give the resident graph.
                                    assertEquals("dimension=" + dimension + ", build=" + (mode == 0 ? "exact" : mode == 1 ? "PQ" : "BQ")
                                                    + ", metric=" + metric + ", refine=" + refine,
                                            residentAdjacency, adjacency(graph));
                                    // A file-backed shared source must not overwrite insertion/diversity queries.
                                    // Binary edges use Hamming distance, so exact-metric searches need more candidates.
                                    int searchBudget = mode == 2 ? 20 : 10;
                                    var result = GraphSearcher.search(vectors.get(5), 10, searchBudget, resident, metric, graph, Bits.ALL);
                                    assertTrue("dimension=" + dimension + ", build=" + (mode == 0 ? "exact" : mode == 1 ? "PQ" : "BQ")
                                                    + ", metric=" + metric + ", refine=" + refine + ", provider="
                                                    + VectorizationProvider.getInstance().getClass().getSimpleName()
                                                    + ", returned=" + Arrays.toString(Arrays.stream(result.getNodes()).mapToInt(n -> n.node).toArray()),
                                            Arrays.stream(result.getNodes()).anyMatch(n -> n.node == 5));
                                    // Reuse the SAME graph/books to isolate source and serialization behavior.
                                    for (boolean fused : new boolean[] {false, true}) {
                                        for (boolean quantized : new boolean[] {false, true}) {
                                            byte[] expected = null;
                                            SearchResult expectedAnswer = null;
                                            for (var values : List.of(resident, source)) {
                                                var graphFile = Files.createTempFile("graph-source", ".jvector");
                                                try {
                                                    var writerBuilder = new RandomAccessOnDiskGraphIndexWriter.Builder(graph, graphFile);
                                                    if (fused) writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
                                                    if (quantized) writerBuilder.with(new NVQ(nvq));
                                                    else writerBuilder.with(new InlineVectors(dimension));
                                                    var local = values.threadLocalSupplier();
                                                    Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
                                                    if (fused) suppliers.put(FeatureId.FUSED_PQ,
                                                            i -> new FusedPQ.State(graph.getView(), pqv, i));
                                                    if (quantized) suppliers.put(FeatureId.NVQ_VECTORS,
                                                            i -> new NVQ.State(nvq.encode(local.get().getVector(i))));
                                                    else suppliers.put(FeatureId.INLINE_VECTORS,
                                                            i -> new InlineVectors.State(local.get().getVector(i)));
                                                    try (var writer = writerBuilder.build()) { writer.write(suppliers); }
                                                    var actual = Files.readAllBytes(graphFile);
                                                    if (expected == null) expected = actual;
                                                    else assertArrayEquals(expected, actual);
                                                    try (var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphFile));
                                                         var view = index.getView(); var searcher = new GraphSearcher(index)) {
                                                        var reranker = view.rerankerFor(vectors.get(5), metric);
                                                        var scores = fused
                                                                ? new DefaultSearchScoreProvider(view.approximateScoreFunctionFor(vectors.get(5), metric), reranker)
                                                                : new DefaultSearchScoreProvider(reranker);
                                                        var answer = searcher.search(scores, 10, Bits.ALL);
                                                        if (expectedAnswer == null) expectedAnswer = answer;
                                                        else assertSameAnswer(expectedAnswer, answer);
                                                    }
                                                } finally { Files.delete(graphFile); }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } finally { Files.delete(path); }
            }
        } finally { pool.shutdown(); single.shutdown(); }
    }

    private static void assertSameAnswer(SearchResult expected, SearchResult actual) {
        assertEquals("visited", expected.getVisitedCount(), actual.getVisitedCount());
        assertEquals("expanded", expected.getExpandedCount(), actual.getExpandedCount());
        assertEquals("expanded base layer", expected.getExpandedCountBaseLayer(), actual.getExpandedCountBaseLayer());
        assertEquals("reranked", expected.getRerankedCount(), actual.getRerankedCount());
        assertSameScore("worst approximate score", expected.getWorstApproximateScoreInTopK(), actual.getWorstApproximateScoreInTopK());
        var expectedNodes = expected.getNodes();
        var actualNodes = actual.getNodes();
        assertEquals("result count", expectedNodes.length, actualNodes.length);
        for (int i = 0; i < expectedNodes.length; i++) {
            assertEquals("node at rank " + i, expectedNodes[i].node, actualNodes[i].node);
            assertSameScore("score at rank " + i, expectedNodes[i].score, actualNodes[i].score);
        }
    }

    private static void assertSameScore(String message, float expected, float actual) {
        // SIMD ADD reductions can change addition order between JIT tiers.
        if (Float.isFinite(expected)) {
            assertEquals(message, expected, actual, Math.max(1e-6f, Math.abs(expected) * 2e-6f));
        } else {
            assertEquals(message, Float.floatToIntBits(expected), Float.floatToIntBits(actual));
        }
    }
}
