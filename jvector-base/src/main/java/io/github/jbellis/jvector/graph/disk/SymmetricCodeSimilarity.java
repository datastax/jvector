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

import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Similarity between two product-quantized codes, computed as the similarity of their decoded
 * vectors without decoding: per subspace, the centroid-pair partial sums come from
 * {@link ProductQuantization#createCodebookPartialSums} (the table the builder's PQ diversity
 * path uses) and are assembled with {@link VectorUtil#assembleAndSumPQ}. A global centroid, when
 * the quantizer has one, contributes closed-form correction terms. Scores use the same [0, 1]
 * conventions as {@link VectorSimilarityFunction#compare}.
 * <p>
 * Used by the compactor to run pairwise diversity checks on reverse-offer candidates from their
 * codes, so that folding offers never has to read an offerer's vector.
 */
final class SymmetricCodeSimilarity {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final ProductQuantization pq;
    private final VectorSimilarityFunction vsf;
    private final int subspaceCount;
    private final int clusterCount;
    // packed upper-triangle centroid-pair sums per subspace: dot products, or squared L2 for EUCLIDEAN
    private final VectorFloat<?> partialSums;
    // [m * clusterCount + c] = dot(global centroid slice m, centroid c of subspace m); null when the
    // PQ is not center-adjusted or the centroid cancels (EUCLIDEAN compares a difference)
    private final float[] centroidTerm;
    private final float centroidNorm2;
    private final ThreadLocal<ByteSequence<?>[]> buffers;

    static boolean supports(VectorSimilarityFunction vsf) {
        return vsf == VectorSimilarityFunction.DOT_PRODUCT
                || vsf == VectorSimilarityFunction.EUCLIDEAN
                || vsf == VectorSimilarityFunction.COSINE;
    }

    SymmetricCodeSimilarity(ProductQuantization pq, VectorSimilarityFunction vsf) {
        if (!supports(vsf)) {
            throw new IllegalArgumentException("unsupported similarity " + vsf);
        }
        this.pq = pq;
        this.vsf = vsf;
        this.subspaceCount = pq.getSubspaceCount();
        this.clusterCount = pq.getClusterCount();
        this.partialSums = pq.createCodebookPartialSums(vsf);
        VectorFloat<?> center = pq.getGlobalCentroid();
        if (center != null && vsf != VectorSimilarityFunction.EUCLIDEAN) {
            float[] terms = new float[subspaceCount * clusterCount];
            int offset = 0;
            for (int m = 0; m < subspaceCount; m++) {
                int size = pq.getSubvectorSize(m);
                VectorFloat<?> codebook = pq.getCodebookVector(m);
                for (int c = 0; c < clusterCount; c++) {
                    terms[m * clusterCount + c] = VectorUtil.dotProduct(codebook, c * size, center, offset, size);
                }
                offset += size;
            }
            this.centroidTerm = terms;
            this.centroidNorm2 = VectorUtil.dotProduct(center, center);
        } else {
            this.centroidTerm = null;
            this.centroidNorm2 = 0f;
        }
        this.buffers = ThreadLocal.withInitial(() -> new ByteSequence<?>[]{
                vts.createByteSequence(subspaceCount), vts.createByteSequence(subspaceCount)});
    }

    /** Similarity of the vectors the two codes decode to, on the score scale of the similarity function. */
    float similarity(byte[] a, byte[] b) {
        ByteSequence<?>[] seqs = buffers.get();
        ByteSequence<?> seqA = seqs[0];
        ByteSequence<?> seqB = seqs[1];
        for (int m = 0; m < subspaceCount; m++) {
            seqA.set(m, a[m]);
            seqB.set(m, b[m]);
        }
        float sum = VectorUtil.assembleAndSumPQ(partialSums, subspaceCount, seqA, 0, seqB, 0, clusterCount);
        switch (vsf) {
            case EUCLIDEAN:
                return 1 / (1 + sum);
            case DOT_PRODUCT:
                return (1 + sum + centroidTerms(a) + centroidTerms(b) + centroidNorm2) / 2;
            default: // COSINE
                float dot = sum + centroidTerms(a) + centroidTerms(b) + centroidNorm2;
                float normA = (float) Math.sqrt(Math.max(1e-12f, norm2(a, seqA)));
                float normB = (float) Math.sqrt(Math.max(1e-12f, norm2(b, seqB)));
                return (1 + dot / (normA * normB)) / 2;
        }
    }

    /** dot(global centroid, decoded centered vector) */
    private float centroidTerms(byte[] code) {
        if (centroidTerm == null) {
            return 0f;
        }
        float s = 0;
        for (int m = 0; m < subspaceCount; m++) {
            s += centroidTerm[m * clusterCount + (code[m] & 0xFF)];
        }
        return s;
    }

    /** squared norm of the decoded (un-centered) vector: diagonal partial sums plus centroid terms */
    private float norm2(byte[] code, ByteSequence<?> seq) {
        return VectorUtil.assembleAndSumPQ(partialSums, subspaceCount, seq, 0, seq, 0, clusterCount)
                + 2 * centroidTerms(code) + centroidNorm2;
    }
}
