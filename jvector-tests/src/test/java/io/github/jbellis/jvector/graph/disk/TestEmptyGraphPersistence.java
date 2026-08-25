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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * A prior attempt at the {@code GraphIndex}/{@code PersistableGraphIndex}/{@code MutableGraphIndex}
 * hierarchy silently regressed empty-graph handling: it dropped the {@code ENTRY_NODE_ABSENT}
 * null-safety guards in the write path and broke {@code getMaxLevel()} to return -1 instead of 0
 * for an empty graph. That went undetected for months. These tests build a real (not hand-mocked)
 * empty graph via {@link GraphIndexBuilder}, persist it through the new
 * {@link PersistableGraphIndex#writer} fluent API, and read it back, to guard against a repeat.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestEmptyGraphPersistence extends LuceneTestCase {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testEmptyOnHeapGraphMaxLevel() throws IOException {
        int dimension = 8;
        var ravv = MockVectorValues.empty(dimension);

        try (var builder = new GraphIndexBuilder(ravv, SIMILARITY, 8, 20, 1.2f, 1.2f, true)) {
            var graph = builder.build(ravv);

            assertEquals(0, graph.size(0));
            assertEquals(0, graph.getIdUpperBound());
            // The regression under test: an empty graph must report level 0, not -1.
            assertEquals(0, graph.getMaxLevel());
            try (var view = graph.getView()) {
                assertNull(view.entryNode());
            }
        }
    }

    @Test
    public void testEmptyGraphWriteAndLoad() throws IOException {
        int dimension = 8;
        var ravv = MockVectorValues.empty(dimension);

        try (var builder = new GraphIndexBuilder(ravv, SIMILARITY, 8, 20, 1.2f, 1.2f, true)) {
            builder.build(ravv);
            PersistableGraphIndex graph = builder.getGraph();

            var graphPath = testDirectory.resolve("empty_graph");
            try (PersistableGraphIndex.WriteBuilder writer = graph.writer(graphPath).with(new InlineVectors(dimension))) {
                // No ordinals exist in an empty graph, so this supplier must never actually be invoked.
                writer.write(Map.of(FeatureId.INLINE_VECTORS,
                        ord -> { throw new AssertionError("supplier invoked for empty graph, ordinal " + ord); }));
            }

            try (var readerSupplier = new SimpleMappedReader.Supplier(graphPath);
                 var onDiskGraph = OnDiskGraphIndex.load(readerSupplier))
            {
                assertEquals(0, onDiskGraph.size(0));
                assertEquals(0, onDiskGraph.getIdUpperBound());
                assertEquals(0, onDiskGraph.getMaxLevel());
                try (var view = onDiskGraph.getView()) {
                    assertNull(view.entryNode());
                }

                var query = TestUtil.randomVector(getRandom(), dimension);
                try (var searcher = new GraphSearcher(onDiskGraph)) {
                    var ssp = DefaultSearchScoreProvider.exact(query, SIMILARITY, ravv);
                    var result = searcher.search(ssp, 5, Bits.ALL);
                    assertTrue("empty graph must return zero results, not throw", result.getNodes().length == 0);
                }
            }
        }
    }
}
