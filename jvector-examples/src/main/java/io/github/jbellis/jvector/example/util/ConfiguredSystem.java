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

import io.github.jbellis.jvector.graph.SearchResult;

import java.util.List;

public interface ConfiguredSystem {
    /**
     * Returns the number of query vectors.
     *
     * @return The number of query vectors.
     */
    int size();

    /**
     * Executes the query at index i using the given parameters.
     *
     * @param topK       Number of top results.
     * @param rerankK    Number of candidates for reranking.
     * @param usePruning Whether to use pruning.
     * @param i          The query vector index.
     * @return the SearchResult for query i.
     */
    SearchResult executeQuery(int topK, int rerankK, boolean usePruning, int i);

    List<? extends List<Integer>> getGroundTruth();

    /**
     * Returns the name of the index
     * @return the name of the index
     */
    String indexName();
}