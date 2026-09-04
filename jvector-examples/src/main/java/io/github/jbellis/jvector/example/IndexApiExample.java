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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.index.Index;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Exercises the generic Index/IndexBuilder hierarchy introduced in docs/index_hierarchy_plan.md,
 * end to end, using the graph/HNSW backing (the only one implemented so far &mdash; IVF is still
 * a seam with no algorithm behind it). This is not a performance or recall benchmark; it is a
 * "how to use the new API correctly" walkthrough:
 * <p>
 * <ol>
 *     <li>build a graph index through {@link Indexes}, the type-first builder entry point</li>
 *     <li>show the two ways a caller ends up holding the result: the concrete {@link GraphIndex}
 *     handle straight from the builder (no cast needed anywhere), and the backing-agnostic
 *     {@link Index} handle narrowed back with {@code instanceof}</li>
 *     <li>search the in-memory graph</li>
 *     <li>persist it to disk with the existing writer pipeline (unchanged by this work)</li>
 *     <li>load it back and confirm the reloaded graph searches the same way</li>
 * </ol>
 * Along the way it also pokes at two pieces of the API that are deliberately incomplete right
 * now &mdash; aggregate builder validation and recipe scaffolding &mdash; so the example doubles
 * as a demonstration of what those look like today.
 */
public class IndexApiExample {
    private static final int VECTOR_COUNT = 100_000;
    private static final int DIMENSION = 128;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;

    public static void main(String[] args) throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

        System.out.printf("Generating %,d random %d-dimensional vectors...%n", VECTOR_COUNT, DIMENSION);
        List<VectorFloat<?>> vectors = randomVectors(vts, VECTOR_COUNT, DIMENSION, new Random(42));
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);

        demonstrateAggregateValidation();
        demonstrateRecipeScaffolding();

        // --- Build, through the new type-first entry point ---
        // Indexes.hnswBuilder() replaces the old Index.hnswBuilder() (see docs/index_hierarchy_plan.md
        // §4): Index itself lives in jvector-api and can no longer reference the concrete builder, so
        // the factory moved to this small facade in jvector-base, alongside the builder it constructs.
        // .nlist()/.nprobe() simply don't exist on this builder -- picking "hnsw" first makes an IVF
        // parameter a compile error, not a runtime one.
        System.out.println("Building the graph, this will take a little while for " + VECTOR_COUNT + " vectors...");
        long startNanos = System.nanoTime();
        GraphIndex graph = Indexes.hnswBuilder()
                .withVectorValues(ravv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .withMaxDegree(16)
                .withBeamWidth(100)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(true)
                .build();
        double buildSeconds = (System.nanoTime() - startNanos) / 1e9;
        System.out.printf("Built a graph of %d nodes in %.1fs%n", graph.size(), buildSeconds);

        // --- The generic handle, and recovering the concrete type from it ---
        // A caller that only needs Index (shared/generic infrastructure that doesn't care which
        // backing it's holding) can treat the result as one. A caller that needs graph-specific
        // behavior narrows back with instanceof -- the Java-11-safe substitute for the sealed-Reader
        // pattern the design doc's source material uses (jvector-api targets Java 11, no sealed
        // interfaces or pattern-matching switch available).
        Index genericHandle = graph;
        if (genericHandle instanceof GraphIndex) {
            GraphIndex narrowed = (GraphIndex) genericHandle;
            System.out.println("Narrowed the generic Index handle back to GraphIndex (dimension="
                    + narrowed.getDimension() + ")");
        }

        // --- Search the in-memory graph ---
        // graph.searcher() is declared to return GraphSearcher, not IndexSearcher -- a covariant
        // override (§5.3 of the plan) so this doesn't need a cast, unlike code that only holds
        // genericHandle above, which would only see IndexSearcher from Index.searcher().
        VectorFloat<?> queryVector = vectors.get(0);
        System.out.println("Searching the in-memory graph for neighbors of vector 0...");
        try (GraphSearcher searcher = graph.searcher()) {
            SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, SIMILARITY_FUNCTION, ravv);
            SearchResult result = searcher.search(ssp, 10, Bits.ALL);
            printResults("in-memory", result);
        }

        // --- Persist to disk ---
        // Everything below is the existing, unchanged persistence pipeline -- it accepts a plain
        // GraphIndex, which is exactly what the new builder hands back.
        Path graphPath = Files.createTempFile("jvector-index-api-example", ".jvector");
        try {
            System.out.println("Writing the graph to " + graphPath + " ...");
            try (GraphIndexWriter writer = GraphIndexWriter
                    .getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, graph, graphPath)
                    .with(new InlineVectors(DIMENSION))
                    .build()) {
                writer.write(Map.of(
                        FeatureId.INLINE_VECTORS,
                        nodeId -> new InlineVectors.State(ravv.getVector(nodeId))));
            }

            // --- Load it back ---
            System.out.println("Loading the graph back from disk...");
            try (ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath)) {
                GraphIndex reloaded = OnDiskGraphIndex.load(readerSupplier);

                // Round-tripped through disk, it's still usable as the generic handle too.
                Index reloadedGenericHandle = reloaded;
                System.out.println("Reloaded graph implements Index: " + (reloadedGenericHandle instanceof Index));

                try (GraphSearcher reloadedSearcher = reloaded.searcher()) {
                    // Views of an on-disk graph with inline vectors double as a RandomAccessVectorValues.
                    var reloadedRavv = (RandomAccessVectorValues) reloadedSearcher.getView();
                    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, SIMILARITY_FUNCTION, reloadedRavv);
                    SearchResult result = reloadedSearcher.search(ssp, 10, Bits.ALL);
                    printResults("reloaded from disk", result);
                }
            }
        } finally {
            Files.deleteIfExists(graphPath);
        }
    }

    /**
     * Every index builder's {@code build()} uses a shared {@code IndexBuilderValidation} helper
     * that reports every missing required value in one exception instead of failing on the first
     * one it finds.
     */
    private static void demonstrateAggregateValidation() {
        System.out.println("Aggregate builder validation: calling build() with nothing set...");
        try {
            Indexes.hnswBuilder().build();
            throw new AssertionError("expected build() to fail");
        } catch (IllegalStateException e) {
            System.out.println("  -> " + e.getMessage());
        }
    }

    /**
     * Recipes ({@link HnswRecipe}, and {@code IvfRecipe} once IVF exists) are wired end to end, but
     * no recipe has real fixed-value formulas defined yet -- see docs/index_hierarchy_plan.md §5.4.
     */
    private static void demonstrateRecipeScaffolding() {
        System.out.println("Recipe scaffolding: applying HnswRecipe.HIGH_RECALL...");
        try {
            Indexes.hnswBuilder().applyRecipe(HnswRecipe.HIGH_RECALL);
            throw new AssertionError("expected applyRecipe() to fail");
        } catch (UnsupportedOperationException e) {
            System.out.println("  -> " + e.getMessage());
        }
    }

    private static void printResults(String label, SearchResult result) {
        System.out.println("Top results (" + label + "):");
        for (SearchResult.NodeScore ns : result.getNodes()) {
            System.out.printf("  node=%d score=%.4f%n", ns.node, ns.score);
        }
    }

    private static List<VectorFloat<?>> randomVectors(VectorTypeSupport vts, int count, int dimension, Random random) {
        List<VectorFloat<?>> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            VectorFloat<?> v = vts.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) {
                v.set(d, random.nextFloat() * 2 - 1);
            }
            VectorUtil.l2normalize(v);
            vectors.add(v);
        }
        return vectors;
    }
}
