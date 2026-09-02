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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Asymmetric distance computation against codes the caller supplies: the exact query on one
 * side, a product-quantization code on the other. {@link #setQuery} builds the per-query
 * partial-sum table over every (subspace, centroid) once; {@link #similarityTo} then scores a
 * code with {@code M} table lookups and no vector read.
 *
 * <p>Same formulas as the {@link PQDecoder}s behind {@link PQVectors#precomputedScoreFunctionFor},
 * but over a code handed in rather than one looked up in a {@link PQVectors}, so a caller that
 * keeps codes elsewhere — a merge's pre-encode cache keyed by output ordinal — can score
 * candidates without reading their records. One instance per thread: the tables are mutable.
 */
@Experimental
public final class AdcScorer {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;
    private final VectorSimilarityFunction vsf;
    private final int subspaces;
    private final int clusters;
    private final VectorFloat<?> partialSums;
    private final VectorFloat<?> partialMagnitudes; // cosine only: per-centroid squared magnitudes
    private float queryMagnitude;
    private boolean querySet;

    public AdcScorer(ProductQuantization pq, VectorSimilarityFunction vsf) {
        this.pq = pq;
        this.vsf = vsf;
        this.subspaces = pq.getSubspaceCount();
        this.clusters = pq.getClusterCount();
        this.partialSums = vts.createFloatVector(subspaces * clusters);
        if (vsf == VectorSimilarityFunction.COSINE) {
            partialMagnitudes = vts.createFloatVector(subspaces * clusters);
            for (int m = 0; m < subspaces; m++) {
                VectorUtil.calculatePartialSelfMagnitudes(pq.codebooks[m], m, pq.subvectorSizesAndOffsets[m][0], clusters, partialMagnitudes);
            }
        } else {
            partialMagnitudes = null;
        }
    }

    /** Builds the partial-sum table for {@code query}; every {@link #similarityTo} until the next call scores against it. */
    public void setQuery(VectorFloat<?> query) {
        VectorFloat<?> centered = pq.globalCentroid == null ? query : VectorUtil.sub(query, pq.globalCentroid);
        VectorSimilarityFunction tableFn = vsf == VectorSimilarityFunction.COSINE ? VectorSimilarityFunction.DOT_PRODUCT : vsf;
        for (int m = 0; m < subspaces; m++) {
            int offset = pq.subvectorSizesAndOffsets[m][1];
            int size = pq.subvectorSizesAndOffsets[m][0];
            VectorUtil.calculatePartialSums(pq.codebooks[m], m, size, clusters, centered, offset, tableFn, partialSums);
        }
        if (vsf == VectorSimilarityFunction.COSINE) {
            queryMagnitude = VectorUtil.dotProduct(centered, centered);
        }
        querySet = true;
    }

    /** jvector-scaled similarity of the current query to {@code code} (length {@code subspaces}). */
    public float similarityTo(ByteSequence<?> code) {
        if (!querySet) {
            throw new IllegalStateException("setQuery first");
        }
        switch (vsf) {
            case DOT_PRODUCT:
                return (1 + VectorUtil.assembleAndSum(partialSums, clusters, code, 0, subspaces)) / 2;
            case EUCLIDEAN:
                return 1 / (1 + VectorUtil.assembleAndSum(partialSums, clusters, code, 0, subspaces));
            case COSINE:
                return (1 + VectorUtil.pqDecodedCosineSimilarity(code, 0, subspaces, clusters, partialSums, partialMagnitudes, queryMagnitude)) / 2;
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + vsf);
        }
    }

    /** The approximate vector a code stands for (centroids joined, global centroid restored). */
    public void decode(ByteSequence<?> code, VectorFloat<?> target) {
        pq.decode(code, target);
    }

    public int codeSize() {
        return subspaces;
    }
}
