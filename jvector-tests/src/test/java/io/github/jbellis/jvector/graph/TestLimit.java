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
        return List.of(createOneVector(vec1), createOneVector(vec2)).toArray(sz -> new VectorFloat<?>[sz]);
    }

    @Test
    public void testLimit() {
        int nDoc = 1000;
        var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        var ravv = MockVectorValues.fromValues(createVectors());

        GraphIndexBuilder builder = new GraphIndexBuilder(ravv, similarityFunction, 32, 100, 1.0f, 1.2f, true);
        var graph = builder.build(ravv);

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
        assertEquals("Number of found results is not equal to [2].", 2, nodes.length);

        var newNode = createOneVector(new float[]{0.7f, 0.7f, 0.7f, 0.7f, 0.7f});

        graph.removeNode(0);
        builder.addGraphNode(0, newNode);

        SearchResult.NodeScore[] nn2 =
                GraphSearcher.search(
                        query,
                        10,
                        20,
                        ravv.copy(),
                        similarityFunction,
                        graph,
                        Bits.ALL
                ).getNodes();

        int[] nodes2 = Arrays.stream(nn2).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [2].", 2, nodes2.length);

    }
}
