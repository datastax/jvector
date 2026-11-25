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
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class QueryExecutor {
    /**
     * Outside measurement, checks repeated searches on each caller's owned
     * searcher using one prepared provider. Concurrent callers may form native
     * projection batches; their floating-point reduction can differ slightly
     * from a separate singleton preparation. Kernel-oracle and recall checks
     * validate that numerical work independently.
     */
    public static void verifyConcurrentQuery(ConfiguredSystem cs, int topK, int rerankK,
            boolean usePruning, int ordinal) {
        verifyConcurrentQuery(cs, cs.queries(), topK, rerankK, usePruning, ordinal);
    }

    private static <Q> void verifyConcurrentQuery(ConfiguredSystem cs, QuerySet<Q> queries, int topK, int rerankK,
            boolean usePruning, int ordinal) {
        var searcher = cs.getSearcher();
        searcher.usePruning(usePruning);
        var provider = queries.scoreProviderFor(queries.get(ordinal), searcher.getView());
        var expected = searcher.search(provider, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
        var actual = searcher.search(provider, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
        if (!java.util.Arrays.equals(expected.getNodes(), actual.getNodes())
                || expected.getVisitedCount() != actual.getVisitedCount())
            throw new IllegalStateException("Concurrent prepared query mismatch at " + ordinal);
    }

    /** Executes only traversal/reranking after a caller has prepared the query. */
    public static SearchResult executePreparedQuery(ConfiguredSystem cs, int topK, int rerankK,
            boolean usePruning, io.github.jbellis.jvector.graph.similarity.PreparedQuery prepared) {
        var searcher = cs.getSearcher();
        searcher.usePruning(usePruning);
        return searcher.search(cs.scoreProviderFor(prepared, searcher.getView()), topK, rerankK, 0.0f, 0.0f, Bits.ALL);
    }

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
        return executeQueryAt(cs, cs.queries(), topK, rerankK, usePruning, i);
    }

    private static <Q> SearchResult executeQueryAt(ConfiguredSystem cs, QuerySet<Q> queries, int topK, int rerankK,
                                                   boolean usePruning, int i) {
        return executeQuery(cs, queries, topK, rerankK, usePruning, queries.get(i));
    }

    // Overload to allow single query injection (e.g., for warm-up with random vectors)
    public static SearchResult executeQuery(ConfiguredSystem cs, int topK, int rerankK, boolean usePruning, VectorFloat<?> queryVector
    ) {
        var searcher = cs.getSearcher();
        searcher.usePruning(usePruning);
        var sf = cs.scoreProviderFor(queryVector, searcher.getView());
        return searcher.search(sf, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
    }

    /** Executes {@code query}, of the system's vector type, scored by {@code queries}. */
    public static <Q> SearchResult executeQuery(ConfiguredSystem cs, QuerySet<Q> queries, int topK, int rerankK,
                                                boolean usePruning, Q query) {
        var searcher = cs.getSearcher();
        searcher.usePruning(usePruning);
        var sf = queries.scoreProviderFor(query, searcher.getView());
        return searcher.search(sf, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
    }
}

