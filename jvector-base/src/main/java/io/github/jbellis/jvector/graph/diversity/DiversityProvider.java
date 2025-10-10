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

package io.github.jbellis.jvector.graph.diversity;

import io.github.jbellis.jvector.graph.NodeArray;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.DocIdSetIterator;

import static java.lang.Math.min;

/**
 * Provides diversity selection functionality for graph neighbors.
 * Implementations determine which neighbors to retain to maintain graph quality.
 */
public interface DiversityProvider {
    /**
     * Updates {@code selected} with the diverse members of {@code neighbors}. The {@code neighbors} array is not modified.
     *
     * @param neighbors the candidate neighbors to select from
     * @param maxDegree the maximum number of neighbors to retain
     * @param diverseBefore the index before which neighbors are already diverse and don't need re-checking
     * @param selected a BitSet to update with the indices of selected diverse neighbors
     * @return the fraction of short edges (neighbors within alpha=1.0)
     */
    double retainDiverse(NodeArray neighbors, int maxDegree, int diverseBefore, BitSet selected);
}
