package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestLimit {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static VectorFloat<?> createOneVector(float[] array) {
        var vec = vectorTypeSupport.createFloatVector(array.length);
        for (int i = 0; i < array.length; i++) {
            vec.set(i, array[i]);
        }
        return vec;
    }

    public static VectorFloat<?>[] createVectors() {
        float[] vec1 = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
        float[] vec2 = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        float[] vec3 = {0.7f, 0.7f, 0.7f, 0.7f, 0.7f};
        return List.of(createOneVector(vec1), createOneVector(vec2), createOneVector(vec3)).toArray(sz -> new VectorFloat<?>[sz]);
    }

    private void runSearch(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction, GraphIndex graph, String prefix) {
        var query = createOneVector(new float[]{0.8f, 0.8f, 0.8f, 0.8f, 0.8f});

        SearchResult.NodeScore[] nn =
                GraphSearcher.search(
                        query,
                        10,
                        20,
                        ravv.copy(),
                        similarityFunction,
                        graph,
                        Bits.ALL
                ).getNodes();

        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
//        System.out.println(Arrays.toString(nodes));
        assertEquals(prefix + ": number of found results is not equal to [2].", 2, nodes.length);

    }

    @Test
    public void testLimit() {
        var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        var ravv = MockVectorValues.fromValues(createVectors());

        for (int trial = 0; trial < 1000; trial++) {
            GraphIndexBuilder builder = new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, true);
            builder.addGraphNode(0, ravv.getVector(0));
            builder.addGraphNode(1, ravv.getVector(1));

            runSearch(ravv, similarityFunction, builder.graph, "Before update");

            builder.markNodeDeleted(0);
            builder.addGraphNode(2, ravv.getVector(2));

            runSearch(ravv, similarityFunction, builder.graph, "After update, no cleanup");
        }

        for (int trial = 0; trial < 1000; trial++) {
            GraphIndexBuilder builder = new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, true);
            builder.addGraphNode(0, ravv.getVector(0));
            builder.addGraphNode(1, ravv.getVector(1));

            runSearch(ravv, similarityFunction, builder.graph, "Before update");

            builder.markNodeDeleted(0);
            builder.cleanup();
            builder.addGraphNode(0, ravv.getVector(2));

            runSearch(ravv, similarityFunction, builder.graph, "After update, pre cleanup");
        }

        for (int trial = 0; trial < 1000; trial++) {
            GraphIndexBuilder builder = new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, true);
            builder.addGraphNode(0, ravv.getVector(0));
            builder.addGraphNode(1, ravv.getVector(1));

            runSearch(ravv, similarityFunction, builder.graph, "Before update");

            builder.markNodeDeleted(0);
            builder.addGraphNode(2, ravv.getVector(2));
            builder.cleanup();

            runSearch(ravv, similarityFunction, builder.graph, "After update, post cleanup");
        }

        for (int trial = 0; trial < 1000; trial++) {
            GraphIndexBuilder builder = new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, true);
            builder.addGraphNode(0, ravv.getVector(0));
            builder.addGraphNode(1, ravv.getVector(1));

            runSearch(ravv, similarityFunction, builder.graph, "Before update");

            builder.markNodeDeleted(0);
            builder.cleanup();
            builder.addGraphNode(0, ravv.getVector(2));
            builder.cleanup();

            runSearch(ravv, similarityFunction, builder.graph, "After update, pre/post cleanup");
        }
    }
}
