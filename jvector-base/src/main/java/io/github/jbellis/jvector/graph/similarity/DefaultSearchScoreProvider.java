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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public final class DefaultSearchScoreProvider implements SearchScoreProvider {
    private final ScoreFunction scoreFunction;
    private final ScoreFunction.ExactScoreFunction reranker;

    /**
     * @param scoreFunction the primary, fast scoring function
     * <p>
     * No reranking is performed.
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction) {
        this(scoreFunction, null);
    }

    /**
     * @param scoreFunction the primary, fast scoring function
     * @param reranker optional reranking function
     * Generally, reranker will be null iff scoreFunction is an ExactScoreFunction.  However,
     * it is allowed, and sometimes useful, to only perform approximate scoring without reranking.
     * <p>
     * Most often it will be convenient to get the reranker either using `RandomAccessVectorValues.rerankerFor`
     * or `ScoringView.rerankerFor`.
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker) {
        assert scoreFunction != null;
        this.scoreFunction = scoreFunction;
        this.reranker = reranker;
    }

    public ScoreFunction scoreFunction() {
        return scoreFunction;
    }

    public ScoreFunction.ExactScoreFunction reranker() {
        return reranker;
    }

    public ScoreFunction.ExactScoreFunction exactScoreFunction() {
        return scoreFunction.isExact()
                ? (ScoreFunction.ExactScoreFunction) scoreFunction
                : reranker;
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static DefaultSearchScoreProvider exact(VectorFloat<?> v, VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        // don't use ESF.reranker, we need thread safety here
        return new DefaultSearchScoreProvider(new BatchedExactScoreFunction(v, vsf, ravv, null));
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static DefaultSearchScoreProvider exact(VectorFloat<?> v, int[] graphToRavvOrdMap ,VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        // don't use ESF.reranker, we need thread safety here
        return new DefaultSearchScoreProvider(new BatchedExactScoreFunction(v, vsf, ravv, graphToRavvOrdMap));
    }

    /**
     * Exact scoring against a RandomAccessVectorValues. {@link #similarityTo} scores one node at a time, with
     * no shared state. {@link #similarityToBatch} gathers the vectors of several nodes into one contiguous
     * scratch vector and scores them with a single batched kernel call, which amortizes the per-call
     * overhead of the vector kernels (e.g. the native downcall) and lets the loads of independent vectors
     * overlap. Scores are identical either way.
     * <p>
     * The gather buffer is per thread and reused across queries, not allocated per instance: a score function is
     * created for every query, and a per-query buffer of count * dimension floats (about 100 KB at 384 dimensions)
     * costs more in allocation and GC than the batching saves at high QPS. The buffer is only used within a single
     * {@code similarityToBatch} call, so it is safe to share between score functions on the same thread, and
     * between threads since each has its own.
     */
    private static final class BatchedExactScoreFunction implements ScoreFunction.ExactScoreFunction {
        // One buffer per thread, grown on demand and never shrunk. Any dimension can use it, since callers only
        // address the first count * dimension floats.
        private static final ThreadLocal<VectorFloat<?>> GATHER_SCRATCH = new ThreadLocal<>();

        private final VectorFloat<?> query;
        private final VectorSimilarityFunction vsf;
        private final RandomAccessVectorValues ravv;
        private final int[] graphToRavvOrdMap; // null means identity

        BatchedExactScoreFunction(VectorFloat<?> query, VectorSimilarityFunction vsf,
                                  RandomAccessVectorValues ravv, int[] graphToRavvOrdMap) {
            this.query = query;
            this.vsf = vsf;
            this.ravv = ravv;
            this.graphToRavvOrdMap = graphToRavvOrdMap;
        }

        private int ordinal(int node) {
            return graphToRavvOrdMap == null ? node : graphToRavvOrdMap[node];
        }

        @Override
        public float similarityTo(int node2) {
            return vsf.compare(query, ravv.getVector(ordinal(node2)));
        }

        @Override
        public boolean supportsSimilarityToBatch() {
            return true;
        }

        @Override
        public void similarityToBatch(int[] nodes, int count, float[] out) {
            final int dim = query.length();
            if (ravv.dimension() != dim) {
                throw new IllegalArgumentException("vector dimensions differ: " + dim + "!=" + ravv.dimension());
            }
            VectorFloat<?> gathered = GATHER_SCRATCH.get();
            if (gathered == null || gathered.length() < count * dim) {
                // size for a typical node degree so that growth is rare; this happens once per thread
                gathered = vts.createFloatVector(Math.max(count, 64) * dim);
                GATHER_SCRATCH.set(gathered);
            }
            for (int i = 0; i < count; i++) {
                ravv.getVectorInto(ordinal(nodes[i]), gathered, i * dim);
            }
            vsf.compareBatch(query, gathered, count, out);
        }
    }
}
