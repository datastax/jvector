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

import io.github.jbellis.jvector.vector.VectorRepresentation;
import io.github.jbellis.jvector.vector.VectorSimilarityType;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public final class DefaultSearchScoreBundle<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> implements SearchScoreBundle {
    private final AsymmetricSimilarityFunction<Primary> asymmetricSimilarityFunction;
    private final AsymmetricSimilarityFunction<Secondary> reranker;

    /**
     * @param asymmetricSimilarityFunction the primary, fast scoring function
     * <p>
     * No reranking is performed.
     */
    public DefaultSearchScoreBundle(AsymmetricSimilarityFunction<Primary> asymmetricSimilarityFunction) {
        this(asymmetricSimilarityFunction, null);
    }

    /**
     * @param asymmetricSimilarityFunction the primary, fast scoring function
     * @param reranker optional reranking function
     * Generally, reranker will be null iff scoreFunction is an ExactScoreFunction.  However,
     * it is allowed, and sometimes useful, to only perform approximate scoring without reranking.
     * <p>
     * Most often it will be convenient to get the reranker either using `RandomAccessVectorValues.rerankerFor`
     * or `ScoringView.rerankerFor`.
     */
    public DefaultSearchScoreBundle(AsymmetricSimilarityFunction<Primary> asymmetricSimilarityFunction, AsymmetricSimilarityFunction<Secondary> reranker) {
        assert asymmetricSimilarityFunction != null;
        if (!asymmetricSimilarityFunction.compatible(reranker)) {
            throw new IllegalArgumentException("reranker is not compatible with scoreFunction");
        }
        this.asymmetricSimilarityFunction = asymmetricSimilarityFunction;
        this.reranker = reranker;
    }

    @Override
    public AsymmetricSimilarityFunction<Primary> primaryScoreFunction() {
        return asymmetricSimilarityFunction;
    }

    @Override
    public AsymmetricSimilarityFunction<Secondary> secondaryScoreFunction() {
        return reranker;
    }

    @Override
    public boolean isPrimaryExact() {
        return asymmetricSimilarityFunction.isExact();
    }

    @Override
    public boolean isSecondaryExact() {
        if (reranker == null) {
            return false;
        }
        return reranker.isExact();
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static DefaultSearchScoreBundle<VectorFloat<?>, VectorFloat<?>> exact(VectorSimilarityType vsf) {
        return new DefaultSearchScoreBundle<VectorFloat<?>, VectorFloat<?>>(new DefaultScoreFunctionAsymmetric(vsf));
    }

}