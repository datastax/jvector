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

package io.github.jbellis.jvector.index;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.ivf.IvfIndexBuilder;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Exercises the module split and the {@code Indexes} entry point introduced for the generic
 * Index/IndexBuilder hierarchy (see docs/index_hierarchy_plan.md): the type-first builders still
 * produce a working index that is usable through the generic {@link Index} and
 * {@link IndexSearcher} handles, and the IVF seam validates its inputs and then refuses cleanly
 * rather than silently returning null.
 */
public class IndexHierarchyTest {

    private static ListRandomAccessVectorValues randomVectors(int count, int dimension) {
        Random random = new Random(0);
        List<VectorFloat<?>> vectors = IntStream.range(0, count)
                .mapToObj(i -> TestUtil.randomVector(random, dimension))
                .collect(Collectors.toList());
        return new ListRandomAccessVectorValues(vectors, dimension);
    }

    @Test
    public void hnswBuilderProducesAWorkingGraphIndex() throws Exception {
        var vectors = randomVectors(64, 8);
        try (GraphIndex index = Indexes.hnswBuilder(vectors, VectorSimilarityFunction.EUCLIDEAN)
                .withMaxDegree(8)
                .withBeamWidth(20)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(false)
                .populateGraph(vectors)) {
            assertEquals(64, index.size());

            // GraphIndex.searcher() is covariantly typed to GraphSearcher (§5.3) -- no cast needed.
            GraphSearcher searcher = index.searcher();
            assertTrue(searcher != null);

            // and Index itself still works as the generic, backing-agnostic handle.
            Index generic = index;
            assertTrue(generic instanceof GraphIndex);
        }
    }

    @Test
    public void builtGraphIsPersistableAndGenericSearchersAreCloseable() throws Exception {
        var vectors = randomVectors(64, 8);
        // populateGraph() returns a PersistableGraphIndex, so the writer accessors need no cast.
        PersistableGraphIndex graph = Indexes.hnswBuilder(vectors, VectorSimilarityFunction.EUCLIDEAN)
                .withMaxDegree(8)
                .withBeamWidth(20)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(false)
                .populateGraph(vectors);

        Path path = Files.createTempFile("index-hierarchy-test", ".graph");
        try {
            try (GraphIndexWriter writer = graph.getWriterBuilder(path)
                    .with(new InlineVectors(vectors.dimension()))
                    .build()) {
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(vectors.getVector(node))));
            }

            try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
                 OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs)) {
                assertEquals(64, onDisk.size(0));

                // Code holding only Index can close the searcher (here holding an on-disk view)
                // without narrowing to GraphSearcher.
                Index generic = onDisk;
                IndexSearcher searcher;
                try (IndexSearcher s = generic.searcher()) {
                    searcher = s;
                    assertTrue(s instanceof GraphSearcher);
                }
                assertTrue(searcher instanceof java.io.Closeable);
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void pathWriterBuildersOpenTheFileOnlyWhenBuilt() throws Exception {
        var vectors = randomVectors(64, 8);
        PersistableGraphIndex graph = Indexes.hnswBuilder(vectors, VectorSimilarityFunction.EUCLIDEAN)
                .withMaxDegree(8)
                .populateGraph(vectors);

        Path dir = Files.createTempDirectory("index-hierarchy-test");
        Path path = dir.resolve("deferred.graph");
        try {
            // Getting a builder, or a build() that fails validation, opens nothing.
            for (var builder : List.of(graph.getWriterBuilder(path), graph.getParallelWriterBuilder(path))) {
                assertFalse(Files.exists(path));
                try {
                    builder.build(); // no vector feature
                    fail("expected IllegalArgumentException");
                } catch (IllegalArgumentException e) {
                    assertFalse(Files.exists(path));
                }
            }

            // build() opens (and creates) the file; the writer closes it.
            try (GraphIndexWriter writer = graph.getParallelWriterBuilder(path)
                    .with(new InlineVectors(vectors.dimension()))
                    .build()) {
                assertTrue(Files.exists(path));
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(vectors.getVector(node))));
            }
            try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
                 OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs)) {
                assertEquals(64, onDisk.size(0));
            }
        } finally {
            Files.deleteIfExists(path);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    public void ivfBuilderRejectsConflictingScoringOptions() {
        var vectors = randomVectors(4, 4);
        try {
            Indexes.ivfBuilder()
                    .withVectorValues(vectors)
                    .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                    .withScoreProvider(BuildScoreProvider.randomAccessScoreProvider(vectors, VectorSimilarityFunction.EUCLIDEAN))
                    .build();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().startsWith("Cannot build IvfIndex: "));
            assertTrue(e.getMessage(), e.getMessage().contains("not both"));
        }
    }

    @Test
    public void ivfBuilderReportsMissingAndConflictingValuesTogether() {
        var vectors = randomVectors(4, 4);
        try {
            Indexes.ivfBuilder()
                    // vectorValues missing
                    .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                    .withScoreProvider(BuildScoreProvider.randomAccessScoreProvider(vectors, VectorSimilarityFunction.EUCLIDEAN))
                    .build();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertEquals("Cannot build IvfIndex, missing required value(s): vectorValues; "
                         + "Set either withScoreProvider() or withSimilarityFunction(), not both", e.getMessage());
        }
    }

    @Test
    public void ivfBuilderValidatesCommonInputsThenRefusesCleanly() {
        // no vectorValues/similarityFunction supplied: the common-input validation should fire
        // before the "not yet implemented" refusal.
        try {
            Indexes.ivfBuilder().build();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("vectorValues"));
            assertTrue(e.getMessage().contains("similarityFunction"));
        }

        // with common inputs supplied, IVF itself isn't implemented yet -- it should say so
        // explicitly rather than return null (the old stub behavior).
        var vectors = randomVectors(4, 4);
        try {
            new IvfIndexBuilder()
                    .withVectorValues(vectors)
                    .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                    .build();
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // expected: IVF's construction parameters and backing implementation don't exist yet
        }
    }

    @Test
    public void onlyTheDefaultRecipeIsDefinedSoFar() {
        for (HnswRecipe recipe : HnswRecipe.values()) {
            assertEquals(recipe == HnswRecipe.DEFAULT, recipe.isDefined());
        }

        try {
            Indexes.hnswBuilder(randomVectors(4, 4), VectorSimilarityFunction.EUCLIDEAN)
                    .applyRecipe(HnswRecipe.HIGH_RECALL);
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // expected: HIGH_RECALL has no values defined yet
        }

        try {
            Indexes.ivfBuilder().applyRecipe(IvfRecipe.HIGH_RECALL);
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }
}
