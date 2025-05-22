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

public class VamanaDiversityProvider implements DiversityProvider {
    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    public final float alpha;

    /** used to compute diversity */
    public final BuildScoreProvider scoreProvider;

    /** Create a new diversity provider */
    public VamanaDiversityProvider(BuildScoreProvider scoreProvider, float alpha) {
        this.scoreProvider = scoreProvider;
        this.alpha = alpha;
    }

    /**
     * Update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
     * It assumes that the i-th neighbor with 0 {@literal <=} i {@literal <} diverseBefore is already diverse.
     * @return the fraction of short edges (neighbors within alpha=1.0)
     */
    public double retainDiverse(NodeArray neighbors, int maxDegree, int diverseBefore, BitSet selected) {
        for (int i = 0; i < min(diverseBefore, maxDegree); i++) {
            selected.set(i);
        }

        int nSelected = diverseBefore;
        double shortEdges = Double.NaN;
        // add diverse candidates, gradually increasing alpha to the threshold
        // (so that the nearest candidates are prioritized)
        float currentAlpha = 1.0f;
        while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
            for (int i = diverseBefore; i < neighbors.size() && nSelected < maxDegree; i++) {
                if (selected.get(i)) {
                    continue;
                }

                int cNode = neighbors.getNode(i);
                float cScore = neighbors.getScore(i);
                var sf = scoreProvider.diversityProviderFor(cNode).scoreFunction();
                if (isDiverse(cNode, cScore, neighbors, sf, selected, currentAlpha)) {
                    selected.set(i);
                    nSelected++;
                }
            }

            if (currentAlpha == 1.0f) {
                // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                // when we don't have to worry about concurrent changes
                shortEdges = nSelected / (float) maxDegree;
            }

            currentAlpha += 0.2f;
        }
        return shortEdges;
    }

    // is the candidate node with the given score closer to the base node than it is to any of the
    // already-selected neighbors
    private static boolean isDiverse(int node, float score, NodeArray others, ScoreFunction sf, BitSet selected, float alpha) {
        assert others.size() > 0;

        for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
            int otherNode = others.getNode(i);
            if (node == otherNode) {
                break;
            }
            if (sf.similarityTo(otherNode) > score * alpha) {
                return false;
            }
        }
        return true;
    }
}
