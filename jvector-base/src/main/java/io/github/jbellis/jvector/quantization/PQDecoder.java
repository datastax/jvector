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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Performs similarity comparisons with compressed vectors without decoding them
 */
abstract class PQDecoder implements ScoreFunction.ApproximateScoreFunction {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    protected final PQVectors cv;

    protected PQDecoder(PQVectors cv) {
        this.cv = cv;
    }

    protected static abstract class CachingDecoder extends PQDecoder {
        protected final VectorFloat<?> partialSums;

        protected CachingDecoder(PQVectors cv, VectorFloat<?> query, VectorSimilarityFunction vsf) {
            super(cv);
            var pq = this.cv.pq;
            partialSums = cv.reusablePartialSums();

            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, vsf, partialSums);
            }
        }

        protected float decodedSimilarity(ByteSequence<?> encoded) {
            return VectorUtil.assembleAndSum(partialSums, cv.pq.getClusterCount(), encoded);
        }
    }

    static class DotProductDecoder extends CachingDecoder {
        public DotProductDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv, query, VectorSimilarityFunction.DOT_PRODUCT);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(cv.get(node2))) / 2;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv, query, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(cv.get(node2)));
        }
    }

    static class CosineDecoder extends PQDecoder {
        protected final VectorFloat<?> partialSums;
        protected final VectorFloat<?> aMagnitude;
        protected final float bMagnitude;

        public CosineDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv);
            var pq = this.cv.pq;

            // this part is not query-dependent, so we can cache it
            aMagnitude = cv.partialSquaredMagnitudes().updateAndGet(current -> {
                if (current != null) {
                    return current;
                }

                var partialMagnitudes = vts.createFloatVector(pq.getSubspaceCount() * pq.getClusterCount());
                for (int m = 0; m < pq.getSubspaceCount(); ++m) {
                    int size = pq.subvectorSizesAndOffsets[m][0];
                    var codebook = pq.codebooks[m];
                    for (int j = 0; j < pq.getClusterCount(); ++j) {
                        partialMagnitudes.set((m * pq.getClusterCount()) + j, VectorUtil.dotProduct(codebook, j * size, codebook, j * size, size));
                    }
                }
                return partialMagnitudes;
            });


            // Compute and cache partial sums and magnitudes for query vector
            partialSums = cv.reusablePartialSums();

            VectorFloat<?> center = pq.globalCentroid;
            VectorFloat<?> centeredQuery = center == null ? query : VectorUtil.sub(query, center);

            for (int m = 0; m < pq.getSubspaceCount(); ++m) {
                int offset = pq.subvectorSizesAndOffsets[m][1];
                int size = pq.subvectorSizesAndOffsets[m][0];
                var codebook = pq.codebooks[m];
                for (int j = 0; j < pq.getClusterCount(); ++j) {
                    partialSums.set((m * pq.getClusterCount()) + j, VectorUtil.dotProduct(codebook, j * size, centeredQuery, offset, size));
                }
            }

            this.bMagnitude = VectorUtil.dotProduct(centeredQuery, centeredQuery);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedCosine(node2)) / 2;
        }

        protected float decodedCosine(int node2) {

            ByteSequence<?> encoded = cv.get(node2);

            return VectorUtil.pqDecodedCosineSimilarity(encoded, cv.pq.getClusterCount(), partialSums, aMagnitude, bMagnitude);
        }
    }
}
