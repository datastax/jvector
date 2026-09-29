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

package io.github.jbellis.jvector.compat;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Code written against JVector 4.0.x, which called {@link GraphIndex} {@code ImmutableGraphIndex},
 * must keep compiling for one more release. This test compiles the ways that name is used by
 * Cassandra (CassandraDiskAnn, CloseableReranker, BruteForceRowIdIteratorTest) and by the tutorials,
 * from a package outside JVector's so no package-private access helps it, and checks the results
 * behave. Delete it together with {@code ImmutableGraphIndex}.
 */
@SuppressWarnings({"removal", "deprecation"}) // exercising the deprecated name is the point
public class ImmutableGraphIndexCompatibilityTest {
    private static final int DIMENSION = 8;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    @Test
    public void legacySpellingsStillCompileAndWork() throws Exception {
        Random random = new Random(0);
        List<VectorFloat<?>> vectors = IntStream.range(0, 200)
                .mapToObj(i -> TestUtil.randomVector(random, DIMENSION))
                .collect(Collectors.toList());
        var ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);

        // Tutorials: the builder's results assigned to ImmutableGraphIndex.
        ImmutableGraphIndex graph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(ravv, VSF, 8, 20, 1.2f, 1.2f, true)) {
            graph = builder.build(ravv);
            ImmutableGraphIndex sameGraph = builder.getGraph();
            assertSame(graph, sameGraph);

            // CassandraOnHeapGraph: a View parameter typed with the old name.
            try (ImmutableGraphIndex.View view = builder.getGraph().getView()) {
                assertTrue(view.size() > 0);
            }
        }
        assertEquals(200, graph.size(0));
        assertTrue(graph.getView().entryNode().node != ImmutableGraphIndex.ENTRY_NODE_ABSENT);
        assertTrue(ImmutableGraphIndex.prettyPrint(graph).contains("# Level 0"));

        // An ImmutableGraphIndex is a GraphIndex, so it goes anywhere the new API takes one.
        GraphIndex asNewType = graph;
        assertSame(graph, asNewType);

        Path path = Files.createTempFile("immutable-graph-index-compat", ".graph");
        try {
            OnDiskGraphIndex.write(graph, ravv, path);

            // CassandraDiskAnn: a field of the old type holding an on-disk graph, and its views cast
            // to the old ScoringView name.
            try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
                 OnDiskGraphIndex rawGraph = OnDiskGraphIndex.load(rs)) {
                ImmutableGraphIndex onDisk = rawGraph;
                try (GraphSearcher searcher = new GraphSearcher(onDisk)) {
                    var view = (ImmutableGraphIndex.ScoringView) searcher.getView();
                    VectorFloat<?> q = vectors.get(0);
                    var ssp = new DefaultSearchScoreProvider(view.rerankerFor(q, VSF));
                    SearchResult result = searcher.search(ssp, 1, Bits.ALL);
                    assertEquals(0, result.getNodes()[0].node);
                }
            }
        } finally {
            Files.deleteIfExists(path);
        }

        // BruteForceRowIdIteratorTest: implementing the view interfaces through the old name.
        ImmutableGraphIndex.ScoringView legacyView = new LegacyScoringView();
        assertEquals(new ImmutableGraphIndex.NodeAtLevel(0, 0), legacyView.entryNode());
    }

    /** Implements the old nested types exactly as Cassandra's test does. */
    private static class LegacyScoringView implements ImmutableGraphIndex.ScoringView {
        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            throw new UnsupportedOperationException();
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            throw new UnsupportedOperationException();
        }

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void processNeighbors(int level, int node, ScoreFunction scoreFunction,
                                     ImmutableGraphIndex.IntMarker visited,
                                     ImmutableGraphIndex.NeighborProcessor neighborProcessor) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public ImmutableGraphIndex.NodeAtLevel entryNode() {
            return new ImmutableGraphIndex.NodeAtLevel(0, 0);
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public boolean contains(int level, int node) {
            return false;
        }

        @Override
        public void close() {
        }
    }
}
