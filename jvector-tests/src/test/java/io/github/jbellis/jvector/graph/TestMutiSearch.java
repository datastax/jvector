package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestMutiSearch  extends RandomizedTest {
    @Test
    public void testMultiSearch() {
        for (int nIndices = 1; nIndices < 11; nIndices++) {
            System.out.println("nIndices: " + nIndices);
//            testMultiSearch(nIndices, false);
            testMultiSearch(nIndices, true);
    }
    }

    public void testMultiSearch(int nIndices, boolean addHierarchy) {
        int nTrials = 10;

        int dimension = 20;
        int nVectors = 1_000;
        int nQueries = 100;

        int topK = 10;
        int rerankK = 30;

        double intersectionAVG = 0;
        double intersectionAVGMulti = 0;
        double expandedCountAVG = 0;
        double expandedAVGMulti = 0;

        for (int trial = 0; trial < nTrials; trial++) {
            var vectors = createRandomVectors(nVectors, dimension);
            var queries = createRandomVectors(nQueries, dimension);
            var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;

            var ravv = MockVectorValues.fromValues(vectors.toArray(sz -> new VectorFloat<?>[sz]));

            GraphIndexBuilder builder =
                    new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, addHierarchy);
            var singleGraph = builder.build(ravv);

            var ravvs = new ArrayList<RandomAccessVectorValues>();
            int batchSize = vectors.size() / nIndices;
            for (int ii = 0; ii < nIndices; ii++) {
                int maxSize = Math.min((ii + 1) * batchSize, vectors.size());
                ravvs.add(MockVectorValues.fromValues(vectors.subList(ii * batchSize, maxSize).toArray(sz -> new VectorFloat<?>[sz])));
            }

            var graphs = new ArrayList<GraphIndex>();
            for (RandomAccessVectorValues innerRavv : ravvs) {
                GraphIndexBuilder innerBuilder =
                        new GraphIndexBuilder(innerRavv, similarityFunction, 32, 100, 1.0f, 1.2f, addHierarchy);
                var graph = innerBuilder.build(innerRavv);
                graphs.add(graph);
            }

            for (var query : queries) {
                int[] groundTruth = computeGroundTruth(similarityFunction, vectors, query, topK);

                SearchResult sr = GraphSearcher.search(query,
                        topK,
                        rerankK,
                        ravv,
                        similarityFunction,
                        singleGraph,
                        Bits.ALL
                );
                SearchResult.NodeScore[] nn = sr.getNodes();
                int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
                assertEquals(String.format("Number of found results is not equal to [%d].", topK), topK, nodes.length);

                float overfactorRerankK;
                if (nIndices == 1) {
                    overfactorRerankK = 1.f;
                } else {
                    overfactorRerankK = 1.f + 0.15f * nIndices;
                }
                int overprovisionRerankK = (int) (rerankK * overfactorRerankK);

                MultiSearchResult srMulti = MultiGraphSearcher.search(query,
                        topK,
                        overprovisionRerankK,
                        ravvs,
                        similarityFunction,
                        graphs,
                        listBitAll(nIndices)
                );
                MultiSearchResult.NodeScore[] nnMulti = srMulti.getNodes();
                int[] nodesMulti = Arrays.stream(nnMulti).mapToInt(nodeScore -> nodeScore.node + nodeScore.index * batchSize).toArray();
                assertEquals(String.format("Number of found results is not equal to [%d].", topK), topK, nodesMulti.length);

                expandedCountAVG += sr.getExpandedCount();
                expandedAVGMulti += srMulti.getExpandedCount();

                intersectionAVG += intersectionSize(groundTruth, nodes);
                intersectionAVGMulti += intersectionSize(groundTruth, nodesMulti);
            }
        }

        intersectionAVG /= topK * nQueries * nTrials;
        intersectionAVGMulti /= topK * nQueries * nTrials;
        System.out.println("Recall: " + intersectionAVG + " single-index");
        System.out.println("Recall: " + intersectionAVGMulti + " multi-index");

        expandedCountAVG /= nQueries * nTrials;
        expandedAVGMulti /= nQueries * nTrials;
        System.out.println("Number of expansions: " + expandedCountAVG + " single-index");
        System.out.println("Number of expansions: " + expandedAVGMulti + " multi-index");

        System.out.println("--");

        assertTrue(
                String.format("The matches between both arrays only reach %f", intersectionAVG),
                        intersectionAVG >= 0.98);
        assertTrue(
                String.format("The matches between both arrays only reach %f", intersectionAVGMulti),
                intersectionAVGMulti >= 0.98);
    }

    public static List<Bits> listBitAll(int size) {
        var list = new ArrayList<Bits>();
        for (int i = 0; i < size; i++) {
            list.add(Bits.ALL);
        }
        return list;
    }

    public static int intersectionSize(int[] a, int[] b) {
        var bTemp = Arrays.stream(b).boxed().collect(Collectors.toList());
        List<Integer> intersection = Arrays.stream(a)
                .filter(bTemp::contains)
                .boxed()
                .collect(Collectors.toList());
        return intersection.size();
    }

    public static int[] computeGroundTruth(VectorSimilarityFunction vsf, List<VectorFloat<?>> vectors, VectorFloat<?> query, int topK) {
        double[] groundTruthDistances = vectors.stream().mapToDouble(vector -> vsf.compare(query, vector)).toArray();
        int[] idx = argsort(groundTruthDistances, false);
        return Arrays.copyOfRange(idx, 0, topK);
    }

    public static int[] argsort(final double[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Double.compare(a[i1], a[i2]);
            }
        });
        return Arrays.stream(indexes)
                .mapToInt(Integer::intValue)
                .toArray();
    }
}
