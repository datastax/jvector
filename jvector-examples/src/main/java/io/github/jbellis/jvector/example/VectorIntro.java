package io.github.jbellis.jvector.example;

import java.io.IOException;
import java.util.List;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class VectorIntro {
    public static void main(String[] args) throws IOException {
        // `VectorizationProvider` is automatically picked based on the system, language version and runtime flags
        // and determines the actual type of the vector data, and provides implementations for common operations
        // like the inner product.
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

        int dimension = 3;

        // Create a `VectorFloat` from a `float[]`.
        // The types that can be converted to a VectorFloat are technically dependent on which VectorizationProvider is picked,
        // but `float[]` is generally a safe bet.
        float[] vector0Array = new float[]{0.1f, 0.2f, 0.3f};
        VectorFloat<?> vector0 = vts.createFloatVector(vector0Array);

        // This toy example uses only three vectors, in practical cases you might have millions or more.
        List<VectorFloat<?>> baseVectors = List.of(
            vector0,
            vts.createFloatVector(new float[]{0.01f, 0.15f, -0.3f}),
            vts.createFloatVector(new float[]{-0.2f, 0.1f, 0.35f})
        );

        // RAVV or `ravv` is convenient shorthand for a RandomAccessVectorValues instance
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, dimension /* 3 */);

        // The type of similarity score to use. JVector supports EUCLIDEAN (L2 distance), DOT_PRODUCT and COSINE.
        VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        // A simple score provider which can compute exact similarity scores by holding a reference to all the base vectors.
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);

        // Graph construction parameters
        int M = 32;  // maximum degree of each node
        int efConstruction = 100;  // search depth during construction
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;  // note: not the best setting for 3D vectors, but good in the general case
        boolean addHierarchy = true;  // use an HNSW-style hierarchy
        boolean refineFinalGraph = true;

        // Build the graph index using a Builder
        ImmutableGraphIndex graph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                dimension,
                M,
                efConstruction,
                neighborOverflow,
                alpha,
                addHierarchy,
                refineFinalGraph)) {
            graph = builder.build(ravv);
        }

        VectorFloat<?> queryVector = vts.createFloatVector(new float[]{0.2f, 0.3f, 0.4f});  // for example
        // The in-memory graph index doesn't own the actual vectors used to construct it.
        // To compute exact scores at search time, you need to pass in the base RAVV again,
        // in addition to the actual query vector
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, ravv);

        int topK = 10;  // number of approximate nearest neighbors to fetch

        // Search the graph via a GraphSearcher
        SearchResult result;
        try (GraphSearcher searcher = new GraphSearcher(graph)) {
            // You can provide a filter to the query as a bit mask.
            // In this case we want the actual topK neighbors without filtering,
            // so we pass in a virtual bit mask representing all ones.
            result = searcher.search(ssp, topK, Bits.ALL);
        }

        for (NodeScore ns : result.getNodes()) {
            int id = ns.node;  // you can look up this ID in the RAVV
            float score = ns.score;  // the similarity score between this vector and the query vector (higher -> more similar)
            System.out.println("ID: " + id + ", Score: " + score + ", Vector: " + ravv.getVector(id));
        }
    }
}
