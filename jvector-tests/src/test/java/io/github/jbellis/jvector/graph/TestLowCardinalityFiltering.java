package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestLowCardinalityFiltering extends LuceneTestCase {
    @Test
    public void testLowCardinalityFiltering() throws IOException {
        testLowCardinalityFiltering(32, 0.01f, 0.87f, false);
        testLowCardinalityFiltering(32, 0.01f, 0.87f, true);
    }
    public void testLowCardinalityFiltering(int maxDegree, float visitedRatioThreshold, float recallThreshold, boolean addHierarchy) throws IOException {
        var R = getRandom();

        int nVectors = 100_000;
        int nQueries = 100;
        int dimensions = 16;
        int topK = 10;

        VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;

        // build index
        VectorFloat<?>[] vectors = TestVectorGraph.createRandomFloatVectors(nVectors, dimensions, R);
        var ravv = new ListRandomAccessVectorValues(List.of(vectors), dimensions);
        var builder = new GraphIndexBuilder(ravv, similarityFunction, maxDegree, 2 * maxDegree, 1.2f, 1.2f, addHierarchy);
        var onHeapGraph = builder.build(ravv);

        // Build the set of accepted ordinals. There are two classes evenly split.
        Map<Boolean, BitSet> bitSets = new HashMap<>();
        bitSets.put(true, new FixedBitSet(nVectors));
        bitSets.put(false, new FixedBitSet(nVectors));
        for (int j = 0; j < nVectors; j++) {
            bitSets.get(R.nextBoolean()).set(j);
        }

        // test raw vectors
        var searcher = new GraphSearcher(onHeapGraph);

        float meanVisitedRatio = 0;
        float meanRecall = 0;

        for (int i = 0; i < nQueries; i++) {
            VectorFloat<?> query = TestUtil.randomVector(R, dimensions);
            boolean queryClass = R.nextBoolean();

            var sf = ravv.rerankerFor(query, similarityFunction);
            var result = searcher.search(new DefaultSearchScoreProvider(sf), topK, 0, bitSets.get(queryClass));

            float recall = getRecall(ravv, bitSets, similarityFunction, query, queryClass, topK, result);

            meanVisitedRatio += ((float) result.getVisitedCount()) / (vectors.length * nQueries);
            meanRecall += recall / (nQueries * topK);
        }

        System.out.println("meanVisitedRatio " +  meanVisitedRatio);
        System.out.println("meanRecall " +  meanRecall);

        assert meanVisitedRatio < visitedRatioThreshold : "visited " + meanVisitedRatio * 100 + "% of the vectors, which is more than " + visitedRatioThreshold * 100 + "%";
        assert meanRecall > recallThreshold : "the recall is too low: " + meanRecall + " < " + recallThreshold;
    }

    /**
     * Create "interesting" test parameters -- shouldn't match too many (we want to validate
     * that threshold code doesn't just crawl the entire graph) or too few (we might not find them)
     */
    private float getRecall(RandomAccessVectorValues ravv, Map<Boolean, BitSet> bitSets, VectorSimilarityFunction similarityFunction, VectorFloat<?> query, boolean queryClass, int topK, SearchResult result) {
        var resultNodes = result.getNodes();
        assertEquals(topK, resultNodes.length);

        NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
        for (int j = 0; j < ravv.size(); j++) {
            if (bitSets.get(queryClass).get(j)) {
                expected.push(j, similarityFunction.compare(query, ravv.getVector(j)));
            }
        }
        var actualNodeIds = Arrays.stream(resultNodes, 0, topK).mapToInt(nodeScore -> nodeScore.node).toArray();

        return computeOverlap(actualNodeIds, expected.nodesCopy());
    }

    private int computeOverlap(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        int overlap = 0;
        for (int i = 0, j = 0; i < a.length && j < b.length; ) {
            if (a[i] == b[j]) {
                ++overlap;
                ++i;
                ++j;
            } else if (a[i] > b[j]) {
                ++j;
            } else {
                ++i;
            }
        }
        return overlap;
    }
}
