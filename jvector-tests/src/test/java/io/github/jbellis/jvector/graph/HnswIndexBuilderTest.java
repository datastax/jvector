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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.api.Index;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests {@link HnswIndexBuilder} and its two subclasses, {@link RavvHnswBuilder} and
 * {@link ScoreProviderHnswBuilder}: batch construction with {@link HnswIndexBuilder#populateGraph},
 * incremental and concurrent insertion with {@link HnswIndexBuilder#addGraphNode}, searching during
 * construction, deletes, rescoring, continuing on an existing graph, compression, and recipes.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class HnswIndexBuilderTest extends RandomizedTest {
    private static final int DIMENSION = 16;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    private static ListRandomAccessVectorValues randomRavv(int n, int dimension) {
        return new ListRandomAccessVectorValues(createRandomVectors(n, dimension), dimension);
    }

    private static HnswIndexBuilder configured(HnswIndexBuilder builder) {
        return builder.withMaxDegree(16)
                .withBeamWidth(50)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(true);
    }

    private static HnswIndexBuilder configured(RandomAccessVectorValues ravv) {
        return configured(Indexes.hnswBuilder(ravv, VSF));
    }

    private static HnswIndexBuilder configured(BuildScoreProvider bsp) {
        return configured(Indexes.hnswBuilder(bsp, DIMENSION));
    }

    /** Fraction of nodes whose own vector finds them as the top search result. */
    private static double selfRecall(GraphIndex graph, RandomAccessVectorValues ravv, Iterable<Integer> ordinals) {
        int found = 0;
        int total = 0;
        try (GraphSearcher searcher = graph.searcher()) {
            for (int ord : ordinals) {
                var ssp = DefaultSearchScoreProvider.exact(ravv.getVector(ord), VSF, ravv);
                SearchResult result = searcher.search(ssp, 1, 10, 0.0f, 0.0f, Bits.ALL);
                if (result.getNodes().length > 0 && result.getNodes()[0].node == ord) {
                    found++;
                }
                total++;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return found / (double) total;
    }

    private static Iterable<Integer> range(int from, int to) {
        return () -> IntStream.range(from, to).iterator();
    }

    @Test
    public void indexesPicksTheSubclassFromHowScoringIsSupplied() {
        var ravv = randomRavv(4, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        assertTrue(Indexes.hnswBuilder(ravv, VSF) instanceof RavvHnswBuilder);
        assertTrue(Indexes.hnswBuilder(bsp, DIMENSION) instanceof ScoreProviderHnswBuilder);
    }

    @Test
    public void buildReturnsAnEmptyGraphWithTheConfiguredShape() {
        var ravv = randomRavv(10, DIMENSION);
        PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, VSF)
                .withMaxDegrees(List.of(24, 12))
                .build();
        assertEquals(0, graph.size(0));
        assertEquals(DIMENSION, graph.getDimension());
        assertEquals(List.of(24, 12), graph.maxDegrees());
        assertTrue(graph.isHierarchical());
    }

    @Test
    public void defaultsMatchGraphIndexBuilder() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
        PersistableGraphIndex graph = builder.build();
        assertEquals(List.of(32), graph.maxDegrees());
        assertTrue(graph.isHierarchical());
        assertEquals(CompressionType.NONE, builder.compressionType());
        assertEquals(100, (int) builder.beamWidth);
        assertEquals(1.2f, builder.neighborOverflow, 0.0f);
        assertEquals(1.2f, builder.alpha, 0.0f);
        assertTrue(builder.graphBuilder.isRefineFinalGraph());
    }

    @Test
    public void scoreProviderBuilderBuildsWithoutVectorValues() {
        var ravv = randomRavv(10, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        PersistableGraphIndex graph = Indexes.hnswBuilder(bsp, DIMENSION).build();
        assertEquals(0, graph.size(0));
        assertEquals(DIMENSION, graph.getDimension());
    }

    @Test
    public void builderSettingsAreValidatedByGraphIndexBuilder() {
        var ravv = randomRavv(4, DIMENSION);
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withBeamWidth(0).build(), "beamWidth");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withNeighborOverflow(0.5f).build(), "neighborOverflow");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withAlpha(0).build(), "alpha");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withMaxDegrees(List.of(8, 0)).build(), "degrees");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF)
                .withMaxDegrees(List.of(16, 8))
                .withAddHierarchy(false)
                .build(), "addHierarchy");
    }

    private static void expectIllegalArgument(Runnable r, String expected) {
        try {
            r.run();
            fail("expected IllegalArgumentException mentioning " + expected);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(expected));
        }
    }

    @Test
    public void populateGraphProducesAFinishedGraph() {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        HnswIndexBuilder builder = configured(ravv);
        PersistableGraphIndex graph = builder.populateGraph(ravv);
        assertEquals(n, graph.size(0));
        assertSame(graph, builder.getGraph());
        assertTrue(((OnHeapGraphIndex) graph).allMutationsCompleted());
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
    }

    @Test
    public void buildIsIdempotent() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
        PersistableGraphIndex first = builder.build();
        GraphIndexBuilder gib = builder.graphBuilder;
        builder.addGraphNode(0, ravv.getVector(0));

        // settings changed after the first build() do not take effect
        builder.withMaxDegree(8);
        assertSame(first, builder.build());
        assertSame(gib, builder.graphBuilder);
        assertEquals(List.of(32), first.maxDegrees());
        assertEquals(1, first.size(0));
    }

    @Test
    public void concurrentFirstBuildsCreateOneGraph() throws Exception {
        var ravv = randomRavv(10, DIMENSION);
        for (int attempt = 0; attempt < 20; attempt++) {
            HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
            ExecutorService pool = Executors.newFixedThreadPool(8);
            try {
                var start = new java.util.concurrent.CountDownLatch(1);
                List<Future<PersistableGraphIndex>> results = IntStream.range(0, 8).mapToObj(t -> pool.submit(() -> {
                    start.await();
                    return t % 2 == 0 ? builder.build() : builder.getGraph();
                })).collect(Collectors.toList());
                start.countDown();
                PersistableGraphIndex expected = results.get(0).get();
                for (Future<PersistableGraphIndex> f : results) {
                    assertSame(expected, f.get());
                }
            } finally {
                pool.shutdownNow();
            }
        }
    }

    @Test
    public void rescoreCarriesOverTheSourceSettings() {
        var ravv = randomRavv(200, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var pool = new java.util.concurrent.ForkJoinPool(2);
        try {
            HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF)
                    .withMaxDegrees(List.of(24, 12))
                    .withBeamWidth(40)
                    .withNeighborOverflow(1.5f)
                    .withAlpha(1.3f)
                    .withRefineFinalGraph(false)
                    .withSimdExecutor(pool)
                    .withParallelExecutor(pool);
            builder.populateGraph(ravv);

            HnswIndexBuilder rescored = HnswIndexBuilder.rescore(builder, bsp);
            assertTrue(rescored instanceof ScoreProviderHnswBuilder);
            assertEquals(List.of(24, 12), rescored.maxDegrees);
            assertTrue(rescored.addHierarchy);
            assertEquals(40, (int) rescored.beamWidth);
            assertEquals(1.5f, rescored.neighborOverflow, 0.0f);
            assertEquals(1.3f, rescored.alpha, 0.0f);
            assertFalse(rescored.refineFinalGraph);
            assertSame(pool, rescored.simdExecutor);
            assertSame(pool, rescored.parallelExecutor);
            assertEquals(List.of(24, 12), rescored.getGraph().maxDegrees());
            assertEquals(DIMENSION, rescored.getGraph().getDimension());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void rescoreOfABuilderOnAnExistingGraphUsesTheGraphsShape() {
        var ravv = randomRavv(200, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder first = Indexes.hnswBuilder(bsp, DIMENSION).withMaxDegree(12).withAddHierarchy(false);
        first.populateGraph(ravv);

        // the continuing builder keeps its default maxDegrees/addHierarchy, which the existing graph overrides
        HnswIndexBuilder continuing = Indexes.hnswBuilder(bsp, DIMENSION)
                .withExistingGraph((OnHeapGraphIndex) first.getGraph());
        HnswIndexBuilder rescored = HnswIndexBuilder.rescore(continuing, bsp);
        assertEquals(List.of(12), rescored.maxDegrees);
        assertFalse(rescored.addHierarchy);
        assertEquals(200, rescored.getGraph().size(0));
    }

    @Test
    public void queriesDoNotBuildTheGraph() {
        var ravv = randomRavv(10, DIMENSION);
        // with PQ, building would train a codebook; asking about memory or inserts must not do that
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.PQ);
        assertEquals(0, builder.insertsInProgress());
        assertEquals(0, builder.ramBytesUsed());
        assertTrue(builder.graphBuilder == null);

        builder.withCompressionType(CompressionType.NONE).build();
        assertTrue(builder.ramBytesUsed() > 0);
    }

    @Test
    public void closingAnUnbuiltBuilderDoesNotBuildIt() throws Exception {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
        builder.close();
        assertTrue(builder.graphBuilder == null);
    }

    @Test
    public void tryWithResourcesKeepsTheGraphUsable() throws Exception {
        int n = 500;
        var ravv = randomRavv(n, DIMENSION);
        PersistableGraphIndex graph;
        try (HnswIndexBuilder builder = configured(ravv)) {
            graph = builder.populateGraph(ravv);
        }
        // closing the builder releases its scratch space, not the graph
        assertEquals(n, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
    }

    @Test
    public void usableAfterClose() throws Exception {
        // close() only releases per-thread scratch space; later calls recreate it on demand.
        int n = 500;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = configured(bsp);
        PersistableGraphIndex graph = builder.build();
        IntStream.range(0, n / 2).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        builder.close();
        IntStream.range(n / 2, n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        builder.cleanup();
        assertSame(graph, builder.getGraph());
        assertEquals(n, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
        builder.close();
    }

    @Test
    public void buildAndPopulateBuildsACompleteGraphInOneCall() throws Exception {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF)) {
            PersistableGraphIndex graph = builder.buildAndPopulate();
            assertEquals(n, graph.size(0));
            assertSame(graph, builder.getGraph());
            assertTrue(((OnHeapGraphIndex) graph).allMutationsCompleted());
            assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
        }
    }

    @Test
    public void buildAndPopulateNeedsVectors() {
        var ravv = randomRavv(10, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        try {
            Indexes.hnswBuilder(bsp, DIMENSION).buildAndPopulate();
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("populateGraph"));
        }
    }

    @Test
    public void populateGraphOnlyPopulatesAnEmptyGraph() {
        var ravv = randomRavv(200, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        // a second populate
        HnswIndexBuilder populated = configured(bsp);
        populated.populateGraph(ravv);
        expectIllegalState(() -> populated.populateGraph(ravv), List.of("already has nodes", "addGraphNode()"));

        // after incremental inserts
        HnswIndexBuilder incremental = configured(bsp);
        incremental.addGraphNode(0, ravv.getVector(0));
        expectIllegalState(() -> incremental.populateGraph(ravv), List.of("already has nodes"));

        // on an existing graph, with either populate method
        var existing = (OnHeapGraphIndex) populated.getGraph();
        expectIllegalState(() -> Indexes.hnswBuilder(bsp, DIMENSION).withExistingGraph(existing).populateGraph(ravv),
                List.of("already has nodes"));
        expectIllegalState(() -> Indexes.hnswBuilder(ravv, VSF).withExistingGraph(existing).buildAndPopulate(),
                List.of("already has nodes"));
        assertEquals(200, existing.size(0));
    }

    @Test
    public void settingsChangedAfterBuildAreIgnored() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF).withBeamWidth(40);
        builder.build();

        // each logs a warning and leaves the builder as it was
        builder.withBeamWidth(60)
                .withMaxDegree(8)
                .withNeighborOverflow(1.5f)
                .withAlpha(1.4f)
                .withAddHierarchy(false)
                .withRefineFinalGraph(false)
                .withCompressionType(CompressionType.PQ)
                .applyRecipe(HnswRecipe.DEFAULT);
        assertEquals(40, builder.beamWidth);
        assertEquals(List.of(32), builder.maxDegrees);
        assertEquals(1.2f, builder.neighborOverflow, 0.0f);
        assertEquals(1.2f, builder.alpha, 0.0f);
        assertTrue(builder.addHierarchy);
        assertTrue(builder.refineFinalGraph);
        assertEquals(CompressionType.NONE, builder.compressionType());
    }

    @Test
    public void inputsAreCheckedWhenGiven() {
        var ravv = randomRavv(10, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        expectNullPointer(() -> Indexes.hnswBuilder(null, VSF), "vectorValues");
        expectNullPointer(() -> Indexes.hnswBuilder(ravv, null), "similarityFunction");
        expectNullPointer(() -> Indexes.hnswBuilder(null, DIMENSION), "scoreProvider");
        expectNullPointer(() -> Indexes.hnswBuilder(ravv, VSF).withMaxDegrees(null), "maxDegrees");
        expectIllegalArgument(() -> Indexes.hnswBuilder(bsp, 0), "dimension must be positive");
        expectIllegalArgument(() -> Indexes.hnswBuilder(bsp, -3), "dimension must be positive");
    }

    private static void expectNullPointer(Runnable r, String expected) {
        try {
            r.run();
            fail("expected NullPointerException mentioning " + expected);
        } catch (NullPointerException e) {
            assertEquals(expected, e.getMessage());
        }
    }

    @Test
    public void withMaxDegreesCopiesTheList() {
        var ravv = randomRavv(10, DIMENSION);
        var degrees = new ArrayList<>(List.of(24, 12));
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF).withMaxDegrees(degrees);
        degrees.set(0, 4);
        assertEquals(List.of(24, 12), builder.build().maxDegrees());
    }

    @Test
    public void existingGraphMustMatchTheBuildersDimension() {
        var ravv = randomRavv(100, DIMENSION);
        var existing = (OnHeapGraphIndex) Indexes.hnswBuilder(ravv, VSF).buildAndPopulate();

        var other = randomRavv(100, DIMENSION * 2);
        expectIllegalState(() -> Indexes.hnswBuilder(other, VSF).withExistingGraph(existing).build(),
                List.of("dimension " + DIMENSION, "dimension " + (DIMENSION * 2)));
        // reported together with a shape conflict
        expectIllegalState(() -> Indexes.hnswBuilder(other, VSF).withExistingGraph(existing).withMaxDegree(8).build(),
                List.of("dimension", "withMaxDegree()/withMaxDegrees()"));
    }

    @Test
    public void concurrentPopulatesPopulateOnce() throws Exception {
        int n = 2_000;
        var ravv = randomRavv(n, DIMENSION);
        for (int attempt = 0; attempt < 5; attempt++) {
            HnswIndexBuilder builder = configured(ravv);
            ExecutorService pool = Executors.newFixedThreadPool(2);
            try {
                var start = new java.util.concurrent.CountDownLatch(1);
                List<Future<Boolean>> results = IntStream.range(0, 2).mapToObj(t -> pool.submit(() -> {
                    start.await();
                    try {
                        builder.populateGraph(ravv);
                        return true;
                    } catch (IllegalStateException e) {
                        assertTrue(e.getMessage(), e.getMessage().contains("already has nodes"));
                        return false;
                    }
                })).collect(Collectors.toList());
                start.countDown();
                int succeeded = 0;
                for (Future<Boolean> f : results) {
                    succeeded += f.get() ? 1 : 0;
                }
                assertEquals(1, succeeded);
                assertEquals(n, builder.getGraph().size(0));
                assertEquals(n, builder.getGraph().getIdUpperBound());
            } finally {
                pool.shutdownNow();
            }
        }
    }

    @Test
    public void populateGraphChecksTheDimension() {
        var ravv = randomRavv(100, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var wrong = randomRavv(100, DIMENSION * 2);
        expectIllegalArgument(() -> Indexes.hnswBuilder(bsp, DIMENSION).populateGraph(wrong), "dimension");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).populateGraph(wrong), "dimension");
    }

    @Test
    public void existingGraphMustBeContinuedWithTheSameKindOfScoring() {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        var exactBsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var exactGraph = (OnHeapGraphIndex) configured(exactBsp).populateGraph(ravv);

        HnswIndexBuilder pqBuilder = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.PQ);
        pqBuilder.build();
        var pqBsp = BuildScoreProvider.pqBuildScoreProvider(VSF, (PQVectors) pqBuilder.getCompressedVectors());
        var pqGraph = (OnHeapGraphIndex) configured(pqBsp).populateGraph(ravv);

        // withCompressionType would train a new quantizer the existing graph wasn't built with
        expectIllegalState(() -> Indexes.hnswBuilder(ravv, VSF).withExistingGraph(exactGraph)
                .withCompressionType(CompressionType.PQ).build(), List.of("trains a new quantizer"));
        expectIllegalState(() -> Indexes.hnswBuilder(ravv, VSF).withExistingGraph(pqGraph)
                .withCompressionType(CompressionType.PQ).build(), List.of("trains a new quantizer"));
        // exact inserts into a PQ graph, and PQ inserts into an exact graph
        expectIllegalState(() -> Indexes.hnswBuilder(ravv, VSF).withExistingGraph(pqGraph).build(),
                List.of("scores with compressed vectors", "this builder scores with exact vectors"));
        expectIllegalState(() -> Indexes.hnswBuilder(exactBsp, DIMENSION).withExistingGraph(pqGraph).build(),
                List.of("scores with compressed vectors"));
        expectIllegalState(() -> Indexes.hnswBuilder(pqBsp, DIMENSION).withExistingGraph(exactGraph).build(),
                List.of("scores with exact vectors", "this builder scores with compressed vectors"));

        // the supported ways: the same kind of scoring as the graph was built with
        assertSame(pqGraph, Indexes.hnswBuilder(pqBsp, DIMENSION).withExistingGraph(pqGraph).build());
        assertSame(exactGraph, Indexes.hnswBuilder(ravv, VSF).withExistingGraph(exactGraph).build());
    }

    @Test
    public void defaultRecipeResetsThePqSettings() {
        var ravv = randomRavv(10, 128);
        var builder = (RavvHnswBuilder) Indexes.hnswBuilder(ravv, VSF)
                .withPqSubspaces(16)
                .withPqGlobalCentering(true)
                .withPqAnisotropicThreshold(0.2f);
        builder.applyRecipe(HnswRecipe.DEFAULT);
        assertEquals(RavvHnswBuilder.defaultPqSubspaces(128), builder.pqSubspaces());
        assertFalse(builder.pqGlobalCentering());
        assertEquals(-1.0f, builder.pqAnisotropicThreshold(), 0.0f);
    }

    @Test
    public void vectorBuilderMustBePopulatedWithItsOwnVectors() {
        var ravv = randomRavv(100, DIMENSION);
        var other = randomRavv(50, DIMENSION);
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).populateGraph(other), "same vectors");
        expectNullPointer(() -> Indexes.hnswBuilder(ravv, VSF).withCompressionType(null), "compressionType");
        // its own vectors, or a copy of them, are accepted
        assertEquals(100, Indexes.hnswBuilder(ravv, VSF).populateGraph(ravv.copy()).size(0));
    }

    @Test
    public void getGraphBuildsOnceAndThenReturnsTheSameGraph() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
        PersistableGraphIndex first = builder.getGraph();
        assertSame(first, builder.getGraph());
        builder.addGraphNode(0, ravv.getVector(0));
        assertSame(first, builder.getGraph());
        assertEquals(1, first.size(0));
    }

    @Test
    public void incrementalBuildWithScoreProvider() throws Exception {
        int n = 2_000;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        // CompactionGraph's shape: a score provider and a dimension, vectors supplied one at a time.
        HnswIndexBuilder builder = configured(bsp);
        PersistableGraphIndex graph = builder.build();
        assertEquals(0, graph.size(0));
        IntStream.range(0, n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        builder.cleanup();

        assertEquals(n, graph.size(0));
        assertTrue(graph.isHierarchical());
        assertEquals(DIMENSION, graph.getDimension());
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
        assertTrue(builder.ramBytesUsed() > 0);

        // The graph is an Index, and persistable without a cast.
        Index generic = graph;
        assertTrue(generic.ramBytesUsed() > 0);
        PersistableGraphIndex persistable = graph;
        assertTrue(persistable instanceof OnHeapGraphIndex);
    }

    @Test
    public void incrementalBuildWithVectorValues() {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);

        // CassandraOnHeapGraph's shape: vector values + similarity function, but nothing is inserted
        // until the caller adds it.
        HnswIndexBuilder builder = configured(ravv);
        PersistableGraphIndex graph = builder.build();
        assertEquals(0, graph.size(0));
        for (int i = 0; i < n; i++) {
            assertTrue(builder.addGraphNode(i, ravv.getVector(i)) > 0);
        }
        builder.cleanup();
        assertEquals(n, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
    }

    @Test
    public void addGraphNodeWithASearchScoreProvider() {
        int n = 500;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = configured(bsp);
        IntStream.range(0, n).forEach(i -> builder.addGraphNode(i, bsp.searchProviderFor(i)));
        builder.cleanup();
        assertEquals(n, builder.getGraph().size(0));
        assertTrue(selfRecall(builder.getGraph(), ravv, range(0, n)) > 0.95);
    }

    @Test
    public void searchesRunConcurrentlyWithInserts() throws Exception {
        int n = 5_000;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = configured(bsp);
        PersistableGraphIndex graph = builder.build();
        ExecutorService pool = Executors.newFixedThreadPool(5);
        try {
            AtomicInteger next = new AtomicInteger();
            AtomicBoolean done = new AtomicBoolean();
            List<Future<?>> inserters = IntStream.range(0, 4).mapToObj(t -> pool.submit(() -> {
                int i;
                while ((i = next.getAndIncrement()) < n) {
                    builder.addGraphNode(i, ravv.getVector(i));
                }
            })).collect(Collectors.toList());
            Future<?> searcher = pool.submit(() -> {
                while (!done.get()) {
                    try (GraphSearcher s = graph.searcher()) {
                        var q = ravv.getVector(getRandom().nextInt(n));
                        s.search(DefaultSearchScoreProvider.exact(q, VSF, ravv), 10, Bits.ALL);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            });

            for (Future<?> f : inserters) {
                f.get();
            }
            done.set(true);
            searcher.get();

            assertEquals(0, builder.insertsInProgress());
            builder.cleanup();
            assertEquals(n, graph.size(0));
            assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void deletedNodesAreHiddenAndThenRemoved() throws Exception {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        HnswIndexBuilder builder = configured(ravv);
        PersistableGraphIndex graph = builder.build();
        IntStream.range(0, n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        IntStream.range(0, n).filter(i -> i % 10 == 0).parallel().forEach(builder::markNodeDeleted);

        // hidden from search immediately
        try (GraphSearcher s = graph.searcher()) {
            for (int i = 0; i < n; i += 10) {
                var result = s.search(DefaultSearchScoreProvider.exact(ravv.getVector(i), VSF, ravv), 5, Bits.ALL);
                for (var ns : result.getNodes()) {
                    assertTrue("deleted node " + ns.node + " returned", ns.node % 10 != 0);
                }
            }
        }

        // and physically removed
        assertTrue(builder.removeDeletedNodes() > 0);
        builder.cleanup();
        assertEquals(n - n / 10, graph.size(0));
        for (int i = 0; i < n; i += 10) {
            assertFalse(graph.containsNode(i));
        }
        assertTrue(selfRecall(graph, ravv, () -> IntStream.range(0, n).filter(i -> i % 10 != 0).iterator()) > 0.95);
    }

    @Test
    public void concurrentInsertsAndDeletes() throws Exception {
        int n = 4_000;
        var ravv = randomRavv(n, DIMENSION);
        HnswIndexBuilder builder = configured(ravv);
        PersistableGraphIndex graph = builder.build();
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            // Each inserter deletes every 10th node right after inserting it, so deletes run concurrently
            // with other threads' inserts.
            AtomicInteger next = new AtomicInteger();
            List<Future<?>> workers = IntStream.range(0, 4).mapToObj(t -> pool.submit(() -> {
                int i;
                while ((i = next.getAndIncrement()) < n) {
                    builder.addGraphNode(i, ravv.getVector(i));
                    if (i % 10 == 0) {
                        builder.markNodeDeleted(i);
                    }
                }
            })).collect(Collectors.toList());
            for (Future<?> f : workers) {
                f.get();
            }
        } finally {
            pool.shutdownNow();
        }

        // Before cleanup, deleted nodes are hidden from searches.
        try (GraphSearcher s = graph.searcher()) {
            for (int i = 0; i < n; i += 10) {
                var result = s.search(DefaultSearchScoreProvider.exact(ravv.getVector(i), VSF, ravv), 5, Bits.ALL);
                for (var ns : result.getNodes()) {
                    assertTrue("deleted node " + ns.node + " returned", ns.node % 10 != 0);
                }
            }
        }

        // cleanup() removes them, and no remaining node keeps an edge to one.
        builder.cleanup();
        assertEquals(n - n / 10, graph.size(0));
        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                for (var it = graph.getNodes(level); it.hasNext(); ) {
                    int node = it.nextInt();
                    assertTrue(node % 10 != 0);
                    for (var neighbors = view.getNeighborsIterator(level, node); neighbors.hasNext(); ) {
                        int neighbor = neighbors.nextInt();
                        assertTrue("edge " + node + " -> deleted " + neighbor, neighbor % 10 != 0);
                    }
                }
            }
        }
        assertTrue(selfRecall(graph, ravv, () -> IntStream.range(0, n).filter(i -> i % 10 != 0).iterator()) > 0.95);
    }

    @Test
    public void rescoreCopiesTheGraphAndKeepsDeletes() throws Exception {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = configured(bsp);
        PersistableGraphIndex before = builder.build();
        IntStream.range(0, n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        IntStream.range(0, n).filter(i -> i % 50 == 0).forEach(builder::markNodeDeleted);

        // the way CompactionGraph swaps in a refined PQ codebook
        HnswIndexBuilder rescored = HnswIndexBuilder.rescore(builder,
                BuildScoreProvider.randomAccessScoreProvider(ravv, VSF));
        PersistableGraphIndex after = rescored.getGraph();
        assertNotSame(before, after);
        assertSame(before, builder.getGraph());
        // already built: build() returns the copy rather than replacing it with an empty graph
        assertSame(after, rescored.build());
        assertEquals(n, after.size(0));
        assertEquals(before.maxDegrees(), after.maxDegrees());

        try (GraphSearcher s = after.searcher()) {
            for (int i = 0; i < n; i += 50) {
                var result = s.search(DefaultSearchScoreProvider.exact(ravv.getVector(i), VSF, ravv), 5, Bits.ALL);
                for (var ns : result.getNodes()) {
                    assertTrue("deleted node " + ns.node + " returned after rescore", ns.node % 50 != 0);
                }
            }
        }

        // the rescored builder keeps building on the copy
        rescored.cleanup();
        assertSame(after, rescored.getGraph());
        assertEquals(n - n / 50, after.size(0));
        for (int i = 0; i < n; i += 50) {
            assertFalse(after.containsNode(i));
        }
        assertTrue(selfRecall(after, ravv, () -> IntStream.range(0, n).filter(i -> i % 50 != 0).iterator()) > 0.95);
    }

    @Test
    public void continueBuildingOnAnExistingGraph() throws Exception {
        int n = 2_000;
        int base = 1_500;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        HnswIndexBuilder first = configured(bsp);
        IntStream.range(0, base).parallel().forEach(i -> first.addGraphNode(i, ravv.getVector(i)));
        first.cleanup();
        OnHeapGraphIndex existing = (OnHeapGraphIndex) first.getGraph();

        // OpenSearch leading-segment merge: continue on the existing graph, add new nodes, delete an
        // old one. Shape settings are taken from the existing graph, not the builder.
        HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION)
                .withExistingGraph(existing)
                .withBeamWidth(50);
        PersistableGraphIndex graph = builder.build();
        assertSame(existing, graph);
        assertEquals(base, graph.size(0));
        assertEquals(List.of(16), graph.maxDegrees());
        assertTrue(graph.isHierarchical());

        IntStream.range(existing.getIdUpperBound(), n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        builder.markNodeDeleted(0);
        builder.cleanup();
        assertEquals(n - 1, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(1, n)) > 0.95);
    }

    @Test
    public void existingGraphRejectsExplicitShapeSettings() {
        var ravv = randomRavv(100, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var existing = (OnHeapGraphIndex) configured(bsp).populateGraph(ravv);

        expectIllegalState(() -> Indexes.hnswBuilder(bsp, DIMENSION).withExistingGraph(existing).withMaxDegree(8).build(),
                List.of("don't also set withMaxDegree()/withMaxDegrees()"));
        // order doesn't matter: the check runs when the graph is built
        expectIllegalState(() -> Indexes.hnswBuilder(bsp, DIMENSION).withMaxDegrees(List.of(8)).withExistingGraph(existing).build(),
                List.of("don't also set withMaxDegree()/withMaxDegrees()"));
        expectIllegalState(() -> Indexes.hnswBuilder(bsp, DIMENSION).withExistingGraph(existing).withAddHierarchy(true).build(),
                List.of("don't also set withAddHierarchy()"));
        // both conflicts are reported together, and the methods that build first check too
        HnswIndexBuilder both = Indexes.hnswBuilder(ravv, VSF)
                .withExistingGraph(existing)
                .withMaxDegree(16)
                .withAddHierarchy(true);
        expectIllegalState(() -> both.addGraphNode(0, ravv.getVector(0)),
                List.of("Cannot build HNSW index", "withMaxDegree()/withMaxDegrees()", "withAddHierarchy()"));
        assertTrue(both.graphBuilder == null);
    }

    @Test
    public void recipeShapeValuesDoNotConflictWithAnExistingGraph() {
        var ravv = randomRavv(100, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var existing = (OnHeapGraphIndex) configured(bsp).populateGraph(ravv);
        PersistableGraphIndex graph = Indexes.hnswBuilder(bsp, DIMENSION)
                .applyRecipe(HnswRecipe.DEFAULT)
                .withExistingGraph(existing)
                .build();
        assertSame(existing, graph);
        assertEquals(List.of(16), graph.maxDegrees());
    }

    private static void expectIllegalState(Runnable r, List<String> expected) {
        try {
            r.run();
            fail("expected IllegalStateException mentioning " + expected);
        } catch (IllegalStateException e) {
            for (String part : expected) {
                assertTrue(e.getMessage(), e.getMessage().contains(part));
            }
        }
    }

    @Test
    public void addingToACleanedUpGraphDoesNotCreateSelfEdges() throws Exception {
        // cleanup() freezes the graph (FrozenView, which does not hide incomplete nodes). Inserting
        // afterwards must unfreeze it, or an insert can find its own half-added node as a neighbor.
        int n = 3_000;
        int base = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var baseRavv = new ListRandomAccessVectorValues(
                IntStream.range(0, base).mapToObj(ravv::getVector).collect(Collectors.toList()), DIMENSION);
        // The graph keeps the diversity provider it was built with, so build it with a score provider
        // that already covers the ordinals that will be appended later.
        var frozen = (OnHeapGraphIndex) configured(bsp).populateGraph(baseRavv);
        assertTrue(frozen.allMutationsCompleted());

        HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION)
                .withBeamWidth(50)
                .withExistingGraph(frozen);
        PersistableGraphIndex graph = builder.build();
        IntStream.range(base, n).parallel().forEach(i -> builder.addGraphNode(i, ravv.getVector(i)));
        assertFalse(frozen.allMutationsCompleted());
        builder.cleanup();

        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                for (var it = graph.getNodes(level); it.hasNext(); ) {
                    int node = it.nextInt();
                    for (var neighbors = view.getNeighborsIterator(level, node); neighbors.hasNext(); ) {
                        assertTrue("self edge at node " + node, neighbors.nextInt() != node);
                    }
                }
            }
        }
        assertEquals(n, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
    }

    @Test
    public void pqCompressionBuildsAWorkingGraph() {
        int n = 2_000;
        int dimension = 32;
        var ravv = randomRavv(n, dimension);
        HnswIndexBuilder builder = configured(Indexes.hnswBuilder(ravv, VSF))
                .withCompressionType(CompressionType.PQ);
        PersistableGraphIndex graph = builder.populateGraph(ravv);
        assertEquals(n, graph.size(0));
        assertEquals(dimension, graph.getDimension());
        // built with approximate scores, searched with exact ones
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.8);
    }

    @Test
    public void bqCompressionBuildsAWorkingGraph() {
        int n = 1_000;
        var ravv = randomRavv(n, DIMENSION);
        HnswIndexBuilder builder = configured(Indexes.hnswBuilder(ravv, VSF))
                .withCompressionType(CompressionType.BQ);
        PersistableGraphIndex graph = builder.populateGraph(ravv);
        assertEquals(n, graph.size(0));
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.5);
    }

    @Test
    public void vectorValuesBuilderRecordsCompressionType() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF);
        assertEquals(CompressionType.NONE, builder.compressionType());
        assertSame(builder, builder.withCompressionType(CompressionType.BQ));
        assertEquals(CompressionType.BQ, builder.compressionType());
    }

    @Test
    public void getCompressedVectorsReturnsWhatTheBuilderTrained() {
        int n = 2_000;
        int dimension = 32;
        var ravv = randomRavv(n, dimension);

        HnswIndexBuilder pqBuilder = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.PQ);
        try {
            pqBuilder.getCompressedVectors();
            fail("expected IllegalStateException before the build");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("build"));
        }
        pqBuilder.buildAndPopulate();
        CompressedVectors pq = pqBuilder.getCompressedVectors();
        assertTrue(pq instanceof PQVectors);
        assertEquals(n, pq.count());
        assertTrue(((PQVectors) pq).getCompressor() instanceof ProductQuantization);
        assertSame(pq, pqBuilder.getCompressedVectors());

        HnswIndexBuilder bqBuilder = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.BQ);
        bqBuilder.build();
        assertTrue(bqBuilder.getCompressedVectors() instanceof BQVectors);

        // no compression: null, before and after the build
        HnswIndexBuilder plain = Indexes.hnswBuilder(ravv, VSF);
        assertNull(plain.getCompressedVectors());
        plain.build();
        assertNull(plain.getCompressedVectors());

        // a score-provider builder's compression belongs to the caller
        var bsp = BuildScoreProvider.pqBuildScoreProvider(VSF, (PQVectors) pq);
        HnswIndexBuilder scoreProviderBuilder = Indexes.hnswBuilder(bsp, dimension);
        scoreProviderBuilder.build();
        assertNull(scoreProviderBuilder.getCompressedVectors());
    }

    @Test
    public void pqSubspacesDefaultToCassandrasRule() {
        // Cassandra's VectorSourceModel.defaultPQBytesFor, at and around each boundary
        int[][] expected = {
                {1, 1}, {3, 3}, {32, 32},
                {33, 32}, {64, 32},
                {65, 32}, {128, 64}, {200, 100},
                {201, 100}, {400, 100},
                {401, 100}, {768, 192},
                {769, 192}, {1536, 192},
                {1537, 192}, {3072, 384},
        };
        for (int[] e : expected) {
            assertEquals("dimension " + e[0], e[1], RavvHnswBuilder.defaultPqSubspaces(e[0]));
        }
        assertEquals(64, ((RavvHnswBuilder) Indexes.hnswBuilder(randomRavv(10, 128), VSF)).pqSubspaces());
    }

    @Test
    public void pqSubspacesCanBeSet() {
        int n = 1_000;
        int dimension = 32;
        var ravv = randomRavv(n, dimension);

        HnswIndexBuilder byDefault = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.PQ);
        byDefault.build();
        ProductQuantization defaultPq = ((PQVectors) byDefault.getCompressedVectors()).getCompressor();
        assertEquals(RavvHnswBuilder.defaultPqSubspaces(dimension), defaultPq.getSubspaceCount());
        assertEquals(RavvHnswBuilder.defaultPqSubspaces(dimension), defaultPq.compressedVectorSize());

        HnswIndexBuilder custom = Indexes.hnswBuilder(ravv, VSF)
                .withPqSubspaces(dimension / 4)
                .withCompressionType(CompressionType.PQ);
        custom.build();
        assertEquals(dimension / 4, ((PQVectors) custom.getCompressedVectors()).getCompressor().getSubspaceCount());

        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withPqSubspaces(0), "between 1 and the vector dimension");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withPqSubspaces(dimension + 1), "between 1 and the vector dimension");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withPqAnisotropicThreshold(1.0f), "pqAnisotropicThreshold");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withPqAnisotropicThreshold(-1.5f), "pqAnisotropicThreshold");
        expectIllegalArgument(() -> Indexes.hnswBuilder(ravv, VSF).withPqAnisotropicThreshold(Float.NaN), "pqAnisotropicThreshold");

        // ignored, with a warning, by a score-provider builder
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder scoreProviderBuilder = Indexes.hnswBuilder(bsp, dimension);
        assertSame(scoreProviderBuilder, scoreProviderBuilder.withPqSubspaces(4));
        assertSame(scoreProviderBuilder, scoreProviderBuilder.withPqGlobalCentering(true));
        assertSame(scoreProviderBuilder, scoreProviderBuilder.withPqAnisotropicThreshold(0.2f));
    }

    @Test
    public void scoreProviderBuilderIgnoresCompressionType() {
        var ravv = randomRavv(10, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION);
        assertSame(builder, builder.withCompressionType(CompressionType.PQ));
        assertEquals(CompressionType.NONE, builder.compressionType());
        assertEquals(0, builder.build().size(0));
    }

    @Test
    public void defaultRecipeRestatesTheDefaults() {
        var ravv = randomRavv(10, DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF)
                .withMaxDegree(8)
                .withBeamWidth(20)
                .withNeighborOverflow(1.5f)
                .withAlpha(1.4f)
                .withAddHierarchy(false)
                .withRefineFinalGraph(false)
                .withCompressionType(CompressionType.PQ)
                .applyRecipe(HnswRecipe.DEFAULT);
        assertEquals(CompressionType.NONE, builder.compressionType());
        assertEquals(List.of(32), builder.maxDegrees);
        assertEquals(100, (int) builder.beamWidth);
        assertEquals(1.2f, builder.neighborOverflow, 0.0f);
        assertEquals(1.2f, builder.alpha, 0.0f);
        assertTrue(builder.addHierarchy);
        assertTrue(builder.refineFinalGraph);

        // settings made after the recipe override it
        builder.withBeamWidth(60);
        assertEquals(60, (int) builder.beamWidth);
    }

    @Test
    public void defaultRecipeOnAScoreProviderBuilderDoesNotTouchCompression() {
        var ravv = randomRavv(10, DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION).applyRecipe(HnswRecipe.DEFAULT);
        assertEquals(CompressionType.NONE, builder.compressionType());
        assertEquals(List.of(32), builder.maxDegrees);
    }

    @Test
    public void undefinedRecipesAreRefused() {
        var ravv = randomRavv(4, DIMENSION);
        for (HnswRecipe recipe : List.of(HnswRecipe.HIGH_RECALL, HnswRecipe.HIGH_PERFORMANCE)) {
            assertFalse(recipe.isDefined());
            try {
                Indexes.hnswBuilder(ravv, VSF).applyRecipe(recipe);
                fail("expected UnsupportedOperationException for " + recipe);
            } catch (UnsupportedOperationException e) {
                assertTrue(e.getMessage().contains(recipe.name()));
            }
        }
    }
}
