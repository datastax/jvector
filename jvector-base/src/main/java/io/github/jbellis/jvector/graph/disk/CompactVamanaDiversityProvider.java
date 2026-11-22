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

package io.github.jbellis.jvector.graph.disk;

import java.util.*;
import java.util.concurrent.*;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import static java.lang.Math.*;

/**
 * Provides Vamana-style diversity filtering for neighbor selection during compaction.
 */
final class CompactVamanaDiversityProvider {
    /** Per-pass increase of the diversity threshold, from 1.0 up to the provider's alpha. */
    private static final float DIVERSITY_ALPHA_STEP = 0.2f;
    /**
     * the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more
     */
    public final float alpha;

    /**
     * used to compute diversity
     */
    public final VectorSimilarityFunction vsf;

    /**
     * Create a new diversity provider
     */
    public CompactVamanaDiversityProvider(VectorSimilarityFunction vsf, float alpha) {
        this.vsf = vsf;
        this.alpha = alpha;
    }

    // candidates whose vector the gather step already holds (decoded wide codes)
    private VectorFloat<?>[] candVecs;
    private boolean[] candHasVec;
    // When candVecs are decoded from a compact code, the pruning test's two sides must carry the
    // same bias: comparing a code-to-code similarity against an exact query score under-prunes,
    // because symmetric quantization reads similarities low. Recomputing the threshold from the
    // decoded candidate puts both sides on one footing. Ordering still uses the exact scores.
    private VectorFloat<?> query;
    private boolean codedCandidateVectors;

    CompactVamanaDiversityProvider withCandidateVectors(VectorFloat<?>[] vecs, boolean[] has) {
        this.candVecs = vecs;
        this.candHasVec = has;
        return this;
    }

    CompactVamanaDiversityProvider withCodedCandidateVectors(VectorFloat<?> query) {
        this.query = query;
        this.codedCandidateVectors = true;
        return this;
    }

    /**
     * Selects diverse neighbors from the candidates listed in {@code order[0..orderSize)}, using a
     * gradually increasing alpha threshold so that the nearest candidates are prioritized. Fills
     * {@code selectedCache}; the candidate arrays are not modified.
     */
    public void retainDiverse(int[] candSrc, int[] candNode, float[] candScore, int[] order, int orderSize,
                              int maxDegree, SelectedVecCache selectedCache, VectorFloat<?> tmp, GraphSearcher[] gs) {
        selectedCache.reset();
        if (orderSize == 0) return;
        int nSelected = 0;

        // add diverse candidates, gradually increasing alpha to the threshold
        // (so that the nearest candidates are prioritized)
        float currentAlpha = 1.0f;
        while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
            for (int i = 0; i < orderSize && nSelected < maxDegree; i++) {
                int ci = order[i];
                int cSrc = candSrc[ci];
                int cNode = candNode[ci];
                float cScore = candScore[ci];

                OnDiskGraphIndex.View cView = (OnDiskGraphIndex.View) gs[cSrc].getView();
                VectorFloat<?> cVec = tmp;
                if (candVecs != null && candHasVec[ci]) {
                    cVec = candVecs[ci];
                } else {
                    cView.getVectorInto(cNode, tmp, 0);
                }
                // see withCodedCandidateVectors: keep both sides of the pruning inequality on
                // the same scale when the candidate vector came from a code
                float cThreshold = (codedCandidateVectors && cVec != tmp && candHasVec[ci])
                        ? vsf.compare(cVec, query) : cScore;
                if (isDiverse(cView, cNode, cVec, cThreshold, currentAlpha, selectedCache)) {
                    selectedCache.add(cSrc, cView, cNode, cScore, cVec);
                    nSelected++;
                }
            }

            currentAlpha += DIVERSITY_ALPHA_STEP;
        }
    }

    /**
     * Checks if a candidate is diverse enough by ensuring it's closer to the base node
     * than to any already-selected neighbor (scaled by alpha threshold).
     */
    private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache) {
        for (int j = 0; j < selectedCache.size; j++) {
            if (selectedCache.views[j] == cView && selectedCache.nodes[j] == cNode) {
                return false; // already selected; don't add a duplicate
            }
            float sim = vsf.compare(cVec, selectedCache.vecs[j]);
            if (sim > cScore * alpha) {
                return false;
            }
        }
        return true;
    }

}
