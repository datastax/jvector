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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The resident adjacency equals the on-disk base layer, upper levels and the entry node are the
 * source's own, and a search over the resident view with the same score function as a search
 * over the on-disk view visits the same graph: identical results, node for node, score for score.
 */
public class TestResidentGraph extends com.carrotsearch.randomizedtesting.RandomizedTest {
    private Path dir;

    @Before
    public void setup() throws IOException {
        dir = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(dir);
    }

    @Test
    public void testAdjacencyAndSearchParity() throws Exception {
        int size = 600, dim = 16;
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vecs = new ArrayList<>(TestUtil.createRandomVectors(size, dim));
        var ravv = new ListRandomAccessVectorValues(vecs, dim);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 8, 40, 1.2f, 1.2f, true);
        var graph = TestUtil.buildSequentially(builder, ravv);
        Path path = dir.resolve("graph");
        TestUtil.writeGraph(graph, ravv, path);

        try (var rs = new SimpleMappedReader.Supplier(path); var onDisk = OnDiskGraphIndex.load(rs)) {
            var resident = ResidentGraph.fromStream(onDisk);
            assertEquals(onDisk.getIdUpperBound(), resident.nodeCount);
            assertEquals(onDisk.getDegree(0), resident.degree);
            assertEquals(ResidentGraph.bytesFor(size, onDisk.getDegree(0)), resident.bytes());
            try (var diskView = onDisk.getView(); var residentView = resident.view((OnDiskGraphIndex.View) onDisk.getView())) {
                for (int level = 0; level <= onDisk.getMaxLevel(); level++) {
                    for (var it = onDisk.getNodes(level); it.hasNext(); ) {
                        int n = it.nextInt();
                        assertArrayEquals("level " + level + " node " + n, edges(diskView, level, n), edges(residentView, level, n));
                        assertTrue(residentView.contains(level, n));
                    }
                }
                assertEquals(diskView.entryNode(), residentView.entryNode());
                assertEquals(diskView.size(), residentView.size());

                // same score function on both sides: exact similarity read through the on-disk view
                var tmp = vts.createFloatVector(dim);
                var rnd = new Random(2);
                try (var diskSearcher = new GraphSearcher.Builder(diskView).build();
                     var residentSearcher = new GraphSearcher.Builder(residentView).build()) {
                    diskSearcher.usePruning(false);
                    residentSearcher.usePruning(false);
                    for (int q = 0; q < 20; q++) {
                        VectorFloat<?> query = TestUtil.randomVector(rnd, dim);
                        ScoreFunction.ExactScoreFunction sf = node -> {
                            diskView.getVectorInto(node, tmp, 0);
                            return VectorSimilarityFunction.COSINE.compare(query, tmp);
                        };
                        var ssp = new DefaultSearchScoreProvider(sf);
                        SearchResult a = diskSearcher.search(ssp, 10, 10, 0f, 0f, Bits.ALL);
                        SearchResult b = residentSearcher.search(ssp, 10, 10, 0f, 0f, Bits.ALL);
                        assertEquals(a.getNodes().length, b.getNodes().length);
                        for (int i = 0; i < a.getNodes().length; i++) {
                            assertEquals("query " + q + " rank " + i, a.getNodes()[i].node, b.getNodes()[i].node);
                            assertEquals(a.getNodes()[i].score, b.getNodes()[i].score, 0f);
                        }
                        assertEquals(a.getVisitedCount(), b.getVisitedCount());
                    }
                }
            }
        }
    }

    private static int[] edges(io.github.jbellis.jvector.graph.ImmutableGraphIndex.View view, int level, int node) {
        var it = view.getNeighborsIterator(level, node);
        int[] out = new int[it.size()];
        for (int k = 0; k < out.length; k++) out[k] = it.nextInt();
        return out;
    }
}
