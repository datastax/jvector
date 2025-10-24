package io.github.jbellis.jvector.example.util;

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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
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

public class SingleConfiguredSystem implements ConfiguredSystem, AutoCloseable {
    private final DataSet ds;
    private final ImmutableGraphIndex index;
    private final CompressedVectors cv;
    private final ExplicitThreadLocal<GraphSearcher> searchers;

    public SingleConfiguredSystem(DataSet ds, ImmutableGraphIndex index, CompressedVectors cv) {
        this.ds = ds;
        this.index = index;
        this.cv = cv;

        this.searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));
    }

    private SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, ImmutableGraphIndex.View view) {
        // if we're not compressing then just use the exact score function
        if (cv == null) {
            return DefaultSearchScoreProvider.exact(queryVector, ds.similarityFunction, ds.getBaseRavv());
        }

        var scoringView = (ImmutableGraphIndex.ScoringView) view;
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