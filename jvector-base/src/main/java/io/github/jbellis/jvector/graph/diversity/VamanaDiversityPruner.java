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
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.representations.RandomAccessVectorRepresentations;
import io.github.jbellis.jvector.graph.similarity.SimilarityFunction;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.vector.VectorRepresentation;

import static java.lang.Math.min;

public class VamanaDiversityPruner<Vec extends VectorRepresentation> implements DiversityPruner {
    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    public final float alpha;

    /** used to compute diversity */
    public final SimilarityFunction<Vec> similarityFunction;
    public final RandomAccessVectorRepresentations<Vec> representations;

    /** Create a new diversity provider */
    public VamanaDiversityPruner(SimilarityFunction<Vec> similarityFunction, RandomAccessVectorRepresentations<Vec> representations, float alpha) {
        this.similarityFunction = similarityFunction;
        this.representations = representations;
        this.alpha = alpha;
    }

    public void retainDiverse(NodeArray neighbors, int maxDegree, float diverseHigher, BitSet selected) {
        int nSelected = 0;

        // add diverse candidates, gradually increasing alpha to the threshold
        // (so that the nearest candidates are prioritized)
        float currentAlpha = 1.0f;
        while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
            NodesIterator it =  neighbors.getIteratorSortedByScores();
            while (it.hasNext() && nSelected < maxDegree) {
                int i = it.nextInt();

                if (selected.get(i)) {
                    continue;
                }

                int cNode = neighbors.getNode(i);
                float cScore = neighbors.getScore(i);

                if (cScore > diverseHigher) {
                    selected.set(i);
                    nSelected++;
                }

                Vec vector = representations.getVector(cNode);
                if (isDiverse(cNode, cScore, neighbors, similarityFunction, vector, selected, currentAlpha)) {
                    selected.set(i);
                    nSelected++;
                }
            }

            currentAlpha += 0.2f;
        }
    }

    // is the candidate node with the given score closer to the base node than it is to any of the
    // already-selected neighbors
    private boolean isDiverse(int node, float score, NodeArray others, SimilarityFunction<Vec> sf, Vec vector, BitSet selected, float alpha) {
        assert others.size() > 0;

        for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
            int otherNode = others.getNode(i);
            if (node == otherNode) {
                break;
            }
            var other = representations.getVector(otherNode);
            if (sf.similarity(vector, other) > score * alpha) {
                return false;
            }
        }
        return true;
    }
}
