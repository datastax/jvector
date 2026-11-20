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
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Batched exact scoring must give the same scores, results and search work as scoring one node at a time. */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestBatchedExactScoring extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static VectorFloat<?> randomVector(Random random, int dim) {
        var v = vts.createFloatVector(dim);
        for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian());
        return v;
    }

    @Test
    public void testSearchMatchesUnbatchedSearch() throws IOException {
        Random random = getRandom();
        int dim = 33, size = 600, topK = 10;
        for (var vsf : VectorSimilarityFunction.values()) {
            var vectors = new ArrayList<VectorFloat<?>>();
            for (int i = 0; i < size; i++) vectors.add(randomVector(random, dim));
            var ravv = new ListRandomAccessVectorValues(vectors, dim);

            ImmutableGraphIndex heapGraph;
            try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, vsf), dim, 16, 60, 1.2f, 1.2f, false)) {
                heapGraph = builder.build(ravv);
            }
            Path path = Files.createTempFile("batched-exact", ".graph");
            try {
                OnDiskGraphIndex.write(heapGraph, ravv, path);
                try (var readerSupplier = ReaderSupplierFactory.open(path);
                     var graph = OnDiskGraphIndex.load(readerSupplier);
                     var batchedSearcher = new GraphSearcher(graph);
                     var plainSearcher = new GraphSearcher(graph)) {
                    var batchedRavv = (RandomAccessVectorValues) batchedSearcher.getView();
                    var plainRavv = (RandomAccessVectorValues) plainSearcher.getView();
                    for (int q = 0; q < 25; q++) {
                        var query = randomVector(random, dim);

                        var batchedSsp = DefaultSearchScoreProvider.exact(query, vsf, batchedRavv);
                        assertTrue(batchedSsp.scoreFunction().supportsSimilarityToBatch());
                        // identical scoring, but without the batch capability
                        ScoreFunction.ExactScoreFunction plain = node -> vsf.compare(query, plainRavv.getVector(node));
                        var plainSsp = new DefaultSearchScoreProvider(plain);

                        var batched = batchedSearcher.search(batchedSsp, topK, 50, 0.0f, 0.0f, Bits.ALL);
                        var unbatched = plainSearcher.search(plainSsp, topK, 50, 0.0f, 0.0f, Bits.ALL);

                        assertEquals(vsf + " visited", unbatched.getVisitedCount(), batched.getVisitedCount());
                        assertEquals(vsf + " expanded", unbatched.getExpandedCount(), batched.getExpandedCount());
                        var a = batched.getNodes();
                        var b = unbatched.getNodes();
                        assertEquals(b.length, a.length);
                        for (int i = 0; i < a.length; i++) {
                            assertEquals(vsf + " node " + i, b[i].node, a[i].node);
                            assertEquals(vsf + " score " + i, b[i].score, a[i].score, 0f);
                        }
                    }
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }
}
