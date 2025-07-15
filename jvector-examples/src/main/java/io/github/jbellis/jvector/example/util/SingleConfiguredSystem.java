package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

public class SingleConfiguredSystem implements QueryExecutor, AutoCloseable {
    private final DataSet ds;
    private final GraphIndex index;
    private final CompressedVectors cv;
    private final ExplicitThreadLocal<GraphSearcher> searchers;

    public SingleConfiguredSystem(DataSet ds, GraphIndex index, CompressedVectors cv) {
        this.ds = ds;
        this.index = index;
        this.cv = cv;

        this.searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));
    }

    private SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view) {
        // if we're not compressing then just use the exact score function
        if (cv == null) {
            return DefaultSearchScoreProvider.exact(queryVector, ds.similarityFunction, ds.getBaseRavv());
        }

        var scoringView = (GraphIndex.ScoringView) view;
        ScoreFunction.ApproximateScoreFunction asf = cv.precomputedScoreFunctionFor(queryVector, ds.similarityFunction);
        var rr = scoringView.rerankerFor(queryVector, ds.similarityFunction);
        return new DefaultSearchScoreProvider(asf, rr);
    }

    @Override
    public void close() throws Exception {
        searchers.close();
    }

    @Override
    public int size() {
        return ds.queryVectors.size();
    }

    @Override
    public SearchResult executeQuery(int topK, int rerankK, boolean usePruning, int i) {
        var queryVector = ds.queryVectors.get(i);
        var searcher = searchers.get();
        searcher.usePruning(usePruning);
        var sf = scoreProviderFor(queryVector, searcher.getView());
        return searcher.search(sf, topK, rerankK, 0.0f, Bits.ALL);
    }

    @Override
    public List<? extends List<Integer>> getGroundTruth() {
        return ds.groundTruth;
    }

    @Override
    public String indexName() { return index.toString(); }
}
