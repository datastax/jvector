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

public interface SearchScoreBundle<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> {

    SimilarityFunction<Primary> primaryScoreFunction();

    SimilarityFunction<Secondary> secondaryScoreFunction();

    /**
     * Convenience function to avoid instantiating a ScoreFunction<Primary>
     * @return true if the primary representations are exact
     */
    boolean isPrimaryExact();

    /**
     * Convenience function to avoid instantiating a ScoreFunction<Secondary>
     * @return true if the secondary representations are exact
     */
    boolean isSecondaryExact();
}