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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.MultiGraphSearcher;
import io.github.jbellis.jvector.graph.MultiSearchResult;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class MultiConfiguredSystem implements ConfiguredSystem, AutoCloseable {
    private final DataSet ds;
    private final MultiSharder sharder;
    private final List<ImmutableGraphIndex> indices;
    private final List<CompressedVectors> cvs;
    private final Function<MultiSearchResult.NodeScore, SearchResult.NodeScore> converter;
    private final ExplicitThreadLocal<MultiGraphSearcher> searchers;
    private final ExplicitThreadLocal<List<SearchScoreProvider>> searchScoreProviders;

    private final List<Bits> acceptedBits;

    public MultiConfiguredSystem(DataSet ds,
                                 MultiSharder sharder,
                                 List<ImmutableGraphIndex> indices,
                                 List<CompressedVectors> cvs,
                                 Function<MultiSearchResult.NodeScore, SearchResult.NodeScore> converter) {
        this.ds = ds;
        this.sharder = sharder;
        this.indices = indices;
        this.cvs = cvs;
        this.converter = converter;

        this.searchers = ExplicitThreadLocal.withInitial(() -> new MultiGraphSearcher(indices));
        this.searchScoreProviders = ExplicitThreadLocal.withInitial(() -> new ArrayList<>(indices.size()));

        acceptedBits = new ArrayList<>(indices.size());
        for (var i = 0; i < indices.size(); i++) {
            acceptedBits.add(Bits.ALL);
        }
    }

    private SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, ImmutableGraphIndex.View view, int index) {
        var cv = cvs.get(index);

        // if we're not compressing then just use the exact score function
        if (cv == null) {
            return DefaultSearchScoreProvider.exact(queryVector, ds.similarityFunction, sharder.getShard(index).threadLocalSupplier().get());
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

        var views = searcher.getViews();
        var ssps = searchScoreProviders.get();
        if (ssps.isEmpty()) { // resize the list if it's empty
            for (var iView = 0; iView < views.size(); iView++) {
                ssps.add(null);
            }
        }
        for (var iView = 0; iView < views.size(); iView++) {
            var ssp = scoreProviderFor(queryVector, views.get(iView), iView);
            ssps.set(iView, ssp);
        }
        var multiSR = searcher.search(ssps, topK, rerankK, 0.f, acceptedBits);
        // This is not the most efficient way to convert the results, and it is not needed in practice.
        // In practice, we want the MultiSearchResult returned directly. Here we perform the conversion to compute the
        // accuracy.
        SearchResult.NodeScore[] converted = Arrays.stream(multiSR.getNodes()).map(converter).toArray(SearchResult.NodeScore[]::new);
        return new SearchResult(converted, multiSR.getVisitedCount(), multiSR.getExpandedCount(),
                multiSR.getExpandedCountBaseLayer(), multiSR.getRerankedCount(),
                multiSR.getWorstApproximateScoreInTopK());
    }

    @Override
    public List<? extends List<Integer>> getGroundTruth() {
        return ds.groundTruth;
    }

    @Override
    public String indexName() {
        StringBuilder sb = new StringBuilder("[");
        for (var index : indices) {
            sb.append(index.toString()).append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}