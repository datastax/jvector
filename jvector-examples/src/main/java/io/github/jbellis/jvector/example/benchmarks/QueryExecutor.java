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

package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.util.Bits;

public class QueryExecutor {
    /**
     * Executes the query at index i using the given parameters.
     *
     * @param cs         Configured system that contains the query vectors.
     * @param topK       Number of top results.
     * @param rerankK    Number of candidates for reranking.
     * @param usePruning Whether to use pruning.
     * @param i          The query vector index.
     * @return the SearchResult for query i.
     */
    public static SearchResult executeQuery(ConfiguredSystem cs, int topK, int rerankK, boolean usePruning, int i) {
        var queryVector = cs.getDataSet().queryVectors.get(i);
        var searcher = cs.getSearcher();
        searcher.usePruning(usePruning);
        var sf = cs.scoreProviderFor(queryVector, searcher.getView());
        return searcher.search(sf, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
    }
}

