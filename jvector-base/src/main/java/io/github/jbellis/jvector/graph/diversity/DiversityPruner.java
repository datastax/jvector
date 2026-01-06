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
import io.github.jbellis.jvector.util.BitSet;

import static java.lang.Math.min;

public interface DiversityPruner {
    /**
     * Updates `selected` with the diverse members of `neighbors`.  `neighbors` is not modified.
     * The parameter `diverseHigher` can be used to speed up the pruning, by assuming that all scores higher than it
     * have already been pruned and only the lower ones need to go through pruning. This assumes that pruning operates
     * in decreasing order of the scores.
     * @param neighbors the list of candidates to prune
     * @param maxDegree the max numbers of elements to be included in the pruned array
     * @param diverseHigher all scores higher than this value have already been pruned
     * @param selected the elements that are not pruned out from the `neighbors` array.
     */
    void retainDiverse(NodeArray neighbors, int maxDegree, float diverseHigher, BitSet selected);
}
