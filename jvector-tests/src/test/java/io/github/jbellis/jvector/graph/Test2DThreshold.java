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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class Test2DThreshold extends LuceneTestCase {
    @Test
    public void testThreshold10k() throws IOException {
        for (int i = 0; i < 10; i++) {
            testThreshold(10_000, 16, 0.85f, 0.9f, false);
            testThreshold(10_000, 16, 0.85f, 0.9f, true);
        }
    }

    @Test
    public void testThreshold20k() throws IOException {
        for (int i = 0; i < 10; i++) {
            testThreshold(20_000, 24, 0.80f, 0.9f, false);
            testThreshold(20_000, 24, 0.80f, 0.9f, true);
        }
    }

    public void testThreshold(int graphSize, int maxDegree, float visitedRatioThreshold, float recallThreshold, boolean addHierarchy) throws IOException {
        var R = getRandom();

        // build index
        VectorFloat<?>[] vectors = TestVectorGraph.createRandomFloatVectors(graphSize, 2, R);
        var ravv = new ListRandomAccessVectorValues(List.of(vectors), 2);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.EUCLIDEAN, maxDegree, 2 * maxDegree, 1.2f, 1.4f, addHierarchy);
        var onHeapGraph = builder.build(ravv);

        // test raw vectors
        var searcher = new GraphSearcher(onHeapGraph);

        int nQueries = 100;
        float meanVisitedRatio = 0;
        float meanRecall = 0;

        for (int i = 0; i < nQueries; i++) {
            TestParams tp = createTestParams(vectors);

            var sf = ravv.rerankerFor(tp.q, VectorSimilarityFunction.EUCLIDEAN);
            var result = searcher.search(new SearchScoreProvider(sf), vectors.length, tp.th, Bits.ALL);

            meanVisitedRatio += ((float) result.getVisitedCount()) / (vectors.length * nQueries);
            meanRecall += ((float) result.getNodes().length) / (tp.exactCount * nQueries);
        }

        assert meanVisitedRatio < visitedRatioThreshold : "visited " + meanVisitedRatio * 100 + "% of the vectors, which is more than " + visitedRatioThreshold * 100 + "%";
        assert meanRecall > recallThreshold : "the recall is too low: " + meanRecall + " < " + recallThreshold;

        // test compressed
        // FIXME see https://github.com/jbellis/jvector/issues/254
//        Path outputPath = Files.createTempFile("graph", ".jvector");
//        TestUtil.writeGraph(onHeapGraph, ravv, outputPath);
//        var pq = ProductQuantization.compute(ravv, ravv.dimension(), 256, false);
//        var cv = new PQVectors(pq, pq.encodeAll(List.of(vectors)));
//
//        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
//             var onDiskGraph = new OnDiskGraphIndex(marr::duplicate, 0);
//             var view = onDiskGraph.getView())
//        {
//            for (int i = 0; i < 10; i++) {
//                TestParams tp = createTestParams(vectors);
//                searcher = new GraphSearcher(onDiskGraph.getView());
//                var reranker = Reranker.from(tp.q, VectorSimilarityFunction.EUCLIDEAN, view);
//                var asf = cv.precomputedScoreFunctionFor(tp.q, VectorSimilarityFunction.EUCLIDEAN);
//                var ssp = new SearchScoreProvider(asf, reranker);
//                var result = searcher.search(ssp, vectors.length, tp.th, Bits.ALL);
//
//                // System.out.printf("visited %d to find %d/%d results for threshold %s%n", result.getVisitedCount(), result.getNodes().length, tp.exactCount, tp.th);
//                assert result.getVisitedCount() < vectors.length : "visited all vectors for threshold " + tp.th;
//                assert result.getNodes().length >= 0.9 * tp.exactCount : "returned " + result.getNodes().length + " nodes for threshold " + tp.th + " but should have returned at least " + tp.exactCount;
//            }
//        }
    }

    /**
     * Create "interesting" test parameters -- shouldn't match too many (we want to validate
     * that threshold code doesn't just crawl the entire graph) or too few (we might not find them)
     */
    private TestParams createTestParams(VectorFloat<?>[] vectors) {
        var R = getRandom();

        // Generate a random query vector and threshold
        VectorFloat<?> q = TestUtil.randomVector(R, 2);
        float th = (float) (0.3 + 0.45 * R.nextDouble());

        // Count the number of vectors that have a similarity score greater than or equal to the threshold
        long exactCount = Arrays.stream(vectors).filter(v -> VectorSimilarityFunction.EUCLIDEAN.compare(q, v) >= th).count();

        return new TestParams(exactCount, q, th);
    }

    /**
     * Encapsulates a search vector q and a threshold th with the exact number of matches in the graph
     */
    private static class TestParams {
        public final long exactCount;
        public final VectorFloat<?> q;
        public final float th;

        public TestParams(long exactCount, VectorFloat<?> q, float th) {
            this.exactCount = exactCount;
            this.q = q;
            this.th = th;
        }
    }
}
