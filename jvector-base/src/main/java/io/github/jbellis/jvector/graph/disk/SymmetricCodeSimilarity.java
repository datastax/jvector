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
    private final int M, K;
    private final VectorFloat<?> partialSums;   // packed upper-triangle centroid-pair sums (dot, or squared L2 for EUCLIDEAN)
    private final float[] centroidTerm;         // [m*K + c] = dot(globalCentroid slice m, codebook centroid c); null when not centered
    private final float centroidNorm2;
    private final ThreadLocal<ByteSequence<?>[]> bufs;

    static boolean supports(VectorSimilarityFunction vsf) {
        return vsf == VectorSimilarityFunction.DOT_PRODUCT || vsf == VectorSimilarityFunction.EUCLIDEAN || vsf == VectorSimilarityFunction.COSINE;
    }

    SymmetricCodeSimilarity(ProductQuantization pq, VectorSimilarityFunction vsf) {
        if (!supports(vsf)) throw new IllegalArgumentException("unsupported similarity " + vsf);
        this.pq = pq;
        this.vsf = vsf;
        this.M = pq.getSubspaceCount();
        this.K = pq.getClusterCount();
        this.partialSums = pq.createCodebookPartialSums(vsf);
        VectorFloat<?> center = pq.getGlobalCentroid();
        if (center != null && vsf != VectorSimilarityFunction.EUCLIDEAN) {   // the centroid cancels in a difference
            float[] ct = new float[M * K]; int off = 0; float n2 = 0;
            for (int m = 0; m < M; m++) {
                int sz = pq.getSubvectorSize(m); var cb = pq.getCodebookVector(m);
                for (int c = 0; c < K; c++) { float acc = 0; for (int i = 0; i < sz; i++) acc += center.get(off + i) * cb.get(c * sz + i); ct[m * K + c] = acc; }
                off += sz;
            }
            for (int i = 0; i < center.length(); i++) n2 += center.get(i) * center.get(i);
            this.centroidTerm = ct; this.centroidNorm2 = n2;
        } else {
            this.centroidTerm = null; this.centroidNorm2 = 0f;
        }
        this.bufs = ThreadLocal.withInitial(() -> new ByteSequence<?>[]{vts.createByteSequence(M), vts.createByteSequence(M)});
    }

    /** Similarity of the vectors the two codes decode to, on the score scale of the similarity function. */
    float similarity(byte[] a, byte[] b) {
        ByteSequence<?>[] bs = bufs.get();
        for (int m = 0; m < M; m++) { bs[0].set(m, a[m]); bs[1].set(m, b[m]); }
        float sum = VectorUtil.assembleAndSumPQ(partialSums, M, bs[0], 0, bs[1], 0, K);
        switch (vsf) {
            case EUCLIDEAN:
                return 1 / (1 + sum);
            case DOT_PRODUCT:
                return (1 + sum + centroidTerms(a) + centroidTerms(b) + centroidNorm2) / 2;
            default: { // COSINE
                float dot = sum + centroidTerms(a) + centroidTerms(b) + centroidNorm2;
                float na = (float) Math.sqrt(Math.max(1e-12f, norm2(a, bs[0]))), nb = (float) Math.sqrt(Math.max(1e-12f, norm2(b, bs[1])));
                return (1 + dot / (na * nb)) / 2;
            }
        }
    }

    private float centroidTerms(byte[] code) {
        if (centroidTerm == null) return 0f;
        float s = 0; for (int m = 0; m < M; m++) s += centroidTerm[m * K + (code[m] & 0xFF)];
        return s;
    }

    /** squared norm of the decoded vector: diagonal partial sums plus centroid terms */
    private float norm2(byte[] code, ByteSequence<?> seq) {
        return VectorUtil.assembleAndSumPQ(partialSums, M, seq, 0, seq, 0, K) + 2 * centroidTerms(code) + centroidNorm2;
    }
}
