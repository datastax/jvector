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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Exercises the module split and the {@code Indexes}/{@code IndexBuilderValidation} plumbing
 * introduced for the generic Index/IndexBuilder hierarchy (see
 * docs/index_hierarchy_plan.md): the type-first builders still produce a working index, aggregate
 * validation still reports every missing field at once, and the IVF seam refuses cleanly rather
 * than silently returning null.
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
        try (GraphIndex index = Indexes.hnswBuilder()
                .withVectorValues(vectors)
                .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                .withMaxDegree(8)
                .withBeamWidth(20)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(false)
                .build()) {
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
        // build() returns PersistableGraphIndex, so the writer accessors need no cast.
        PersistableGraphIndex graph = Indexes.hnswBuilder()
                .withVectorValues(vectors)
                .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                .withMaxDegree(8)
                .withBeamWidth(20)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(false)
                .build();

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
    public void hnswBuilderAggregatesEveryMissingField() {
        try {
            Indexes.hnswBuilder().build();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            // every required field should be named in one message, not just the first one found
            assertTrue(e.getMessage().contains("vectorValues"));
            assertTrue(e.getMessage().contains("similarityFunction"));
            assertTrue(e.getMessage().contains("beamWidth"));
            assertTrue(e.getMessage().contains("neighborOverflow"));
            assertTrue(e.getMessage().contains("alpha"));
        }
    }

    @Test
    public void hnswBuilderRejectsConflictingScoringOptions() {
        var vectors = randomVectors(4, 4);
        try {
            Indexes.hnswBuilder()
                    .withVectorValues(vectors)
                    .withSimilarityFunction(VectorSimilarityFunction.EUCLIDEAN)
                    .withScoreProvider(io.github.jbellis.jvector.graph.similarity.BuildScoreProvider
                            .randomAccessScoreProvider(vectors, VectorSimilarityFunction.EUCLIDEAN))
                    .build();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not both"));
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
    public void recipesAreScaffoldedButNotYetDefined() {
        try {
            Indexes.hnswBuilder().applyRecipe(HnswRecipe.HIGH_RECALL);
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // expected: no recipe has real fixed-value formulas yet
        }

        try {
            Indexes.ivfBuilder().applyRecipe(IvfRecipe.HIGH_RECALL);
            fail("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }
}
