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

package io.github.jbellis.jvector.graph.similarity;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;

/**
 * Bulk exact scoring must give the same scores as one-at-a-time scoring for every kind of source: one that holds its
 * vectors, a remapped one, the on-disk view, and custom copying and shared sources on the default path.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestBulkExactScoring extends RandomizedTest {
    private Path dir;

    @Before
    public void setUp() throws Exception {
        dir = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(dir);
    }

    @Test
    public void testBulkMatchesSingle() throws Exception {
        int size = 300;
        for (int dim : new int[] {3, 16, 37, 128}) {
            for (var vsf : VectorSimilarityFunction.values()) {
                var ravv = new ListRandomAccessVectorValues(TestUtil.createRandomVectors(size, dim), dim);
                var builder = new GraphIndexBuilder(ravv, vsf, 8, 30, 1.2f, 1.2f, false);
                var graph = TestUtil.buildSequentially(builder, ravv);
                Path path = dir.resolve("g-" + dim + "-" + vsf);
                TestUtil.writeGraph(graph, ravv, path);
                try (var readers = new SimpleMappedReader.Supplier(path.toAbsolutePath());
                     var onDisk = OnDiskGraphIndex.load(readers);
                     var view = onDisk.getView()) {
                    var q = TestUtil.randomVector(getRandom(), dim);
                    check(DefaultSearchScoreProvider.exact(q, vsf, ravv).scoreFunction(), size, vsf + " held");
                    check(DefaultSearchScoreProvider.exact(q, vsf, (RandomAccessVectorValues) view).scoreFunction(), size, vsf + " on-disk");
                    check(DefaultSearchScoreProvider.exact(q, vsf, copying(ravv, null)).scoreFunction(), size, vsf + " copying");
                    check(DefaultSearchScoreProvider.exact(q, vsf, shared(ravv)).scoreFunction(), size, vsf + " shared");
                    int[] reversed = new int[size];
                    for (int i = 0; i < size; i++) reversed[i] = size - 1 - i;
                    check(DefaultSearchScoreProvider.exact(q, reversed, vsf, ravv).scoreFunction(), size, vsf + " remapped");
                }
            }
        }
    }

    /** The default bulk method reads each node once, as a loop over getVector would. */
    @Test
    public void testDefaultReadsEachNodeOnce() {
        int dim = 16, size = 50;
        var ravv = new ListRandomAccessVectorValues(TestUtil.createRandomVectors(size, dim), dim);
        int[] reads = new int[1];
        var source = copying(ravv, reads);
        int[] nodes = {3, 1, 4, 1, 5, 9, 2, 6};
        float[] results = new float[nodes.length];
        source.similarityTo(TestUtil.randomVector(getRandom(), dim), VectorSimilarityFunction.DOT_PRODUCT, nodes, nodes.length, results);
        assertEquals(nodes.length, reads[0]);
    }

    private static void check(ScoreFunction sf, int size, String where) {
        int count = 23;
        int[] nodes = new int[count];
        for (int i = 0; i < count; i++) nodes[i] = (i * 37) % size;
        float[] bulk = new float[count];
        sf.similarityTo(nodes, count, bulk);
        for (int i = 0; i < count; i++) {
            float single = sf.similarityTo(nodes[i]);
            assertEquals(where + " i=" + i, single, bulk[i], 1e-5f * Math.max(1f, Math.abs(single)));
        }
    }

    /** A source that returns one reused vector for every read and does not override the bulk method. */
    private static RandomAccessVectorValues shared(RandomAccessVectorValues inner) {
        var reused = inner.getVector(0).copy();
        return new RandomAccessVectorValues() {
            @Override public int size() { return inner.size(); }
            @Override public int dimension() { return inner.dimension(); }
            @Override public VectorFloat<?> getVector(int node) { reused.copyFrom(inner.getVector(node), 0, 0, inner.dimension()); return reused; }
            @Override public boolean isValueShared() { return true; }
            @Override public RandomAccessVectorValues copy() { return shared(inner); }
        };
    }

    /** A source that returns a new copy on every read and does not override the bulk method. */
    private static RandomAccessVectorValues copying(RandomAccessVectorValues inner, int[] reads) {
        return new RandomAccessVectorValues() {
            @Override public int size() { return inner.size(); }
            @Override public int dimension() { return inner.dimension(); }
            @Override public VectorFloat<?> getVector(int node) { if (reads != null) reads[0]++; return inner.getVector(node).copy(); }
            @Override public boolean isValueShared() { return false; }
            @Override public RandomAccessVectorValues copy() { return this; }
        };
    }
}
