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
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.List;
import java.util.Optional;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

/**
 * Tests that an in-memory index reports the similarity function it was built with, through every way of
 * building one, and reports none when it can't know. {@code TestRecordedSimilarity} covers on-disk graphs.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestIndexSimilarityFunction extends RandomizedTest {
    private static final int DIMENSION = 32;

    private static ListRandomAccessVectorValues unitRavv(int n, int dimension) {
        List<VectorFloat<?>> vectors = createRandomVectors(n, dimension);
        vectors.forEach(VectorUtil::l2normalize);
        return new ListRandomAccessVectorValues(vectors, dimension);
    }

    @Test
    public void hnswBuilderGraphsReportTheBuildersFunction() throws Exception {
        var ravv = unitRavv(300, DIMENSION);
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            for (CompressionType compression : CompressionType.values()) {
                try (HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, vsf).withCompressionType(compression)) {
                    PersistableGraphIndex graph = builder.buildAndPopulate();
                    assertEquals(vsf + " " + compression, Optional.of(vsf), graph.similarityFunction());
                }
            }
        }
    }

    @Test
    public void scoreProviderGraphsReportTheProvidersFunction() throws Exception {
        var ravv = unitRavv(100, DIMENSION);
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
            assertEquals(Optional.of(vsf), bsp.searchSimilarityFunction());
            try (HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION)) {
                assertEquals(Optional.of(vsf), builder.populateGraph(ravv).similarityFunction());
            }
            // the legacy builder, unchanged for existing callers, reports it too
            try (var legacy = new GraphIndexBuilder(bsp, DIMENSION, 16, 50, 1.2f, 1.2f, true)) {
                assertEquals(Optional.of(vsf), legacy.build(ravv).similarityFunction());
            }
        }
    }

    @Test
    public void aCustomScoreProviderNamesNoFunction() throws Exception {
        var ravv = unitRavv(100, DIMENSION);
        var inner = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.COSINE);
        // a caller's own provider, written before searchSimilarityFunction existed
        var custom = new BuildScoreProvider() {
            @Override public boolean isExact() { return inner.isExact(); }
            @Override public VectorFloat<?> approximateCentroid() { return inner.approximateCentroid(); }
            @Override public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) { return inner.searchProviderFor(vector); }
            @Override public SearchScoreProvider searchProviderFor(int node1) { return inner.searchProviderFor(node1); }
            @Override public SearchScoreProvider diversityProviderFor(int node1) { return inner.diversityProviderFor(node1); }
        };
        assertEquals(Optional.empty(), custom.searchSimilarityFunction());
        try (var builder = new GraphIndexBuilder(custom, DIMENSION, 16, 50, 1.2f, 1.2f, true)) {
            assertEquals(Optional.empty(), builder.build(ravv).similarityFunction());
        }
    }
}
