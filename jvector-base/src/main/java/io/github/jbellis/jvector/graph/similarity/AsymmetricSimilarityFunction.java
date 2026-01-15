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

/**
 * Provides an API for encapsulating similarity to another node or vector.  Used both for
 * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
 * with a reference to the query vector).
 * <p>
 * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
 * can be defined as a simple lambda.
 */
public interface AsymmetricSimilarityFunction<Vec extends VectorRepresentation> {
    /**
     * @return true if the ScoreFunction returns exact, full-resolution scores. That is, if the underlying VectorRepresentation is exact.
     */
    boolean isExact();


    void fixQuery(VectorFloat<?> query);

    /**
     * @return the similarity to another vector representation. This method is stateful and requires calling fixFirstArgument.
     */
    float similarityTo(Vec other);

    /**
     * @return the similarity between vec1 and vec2. This method is stateless, it does not use the fixQuery path.
     */
    float similarity(Vec vec1, Vec vec2);

    /**
     * @return the VectorSimilarityFunction used by this score function
     */
    VectorSimilarityType getSimilarityFunction();

    /**
     * @return a copy of this score function
     */
    AsymmetricSimilarityFunction<Vec> copy();
}
