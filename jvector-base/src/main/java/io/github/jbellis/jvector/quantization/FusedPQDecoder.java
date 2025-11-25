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

import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2ObjectHashMap;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use vectors fused into a graph.
 */
public abstract class FusedPQDecoder implements ScoreFunction.ApproximateScoreFunction {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    protected final ProductQuantization pq;
    Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures;
    protected final VectorFloat<?> query;
    protected final ExactScoreFunction esf;
    // connected to the Graph View by caller
    protected final FusedPQ.PackedNeighbors packedNeighbors;
    // caller passes this to us for re-use across calls
    protected final ByteSequence<?> neighborCodes;
    // decoder state
    protected final VectorFloat<?> partialSums;
    protected final VectorSimilarityFunction vsf;
    protected int origin;

    protected FusedPQDecoder(ProductQuantization pq,
                             Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                             VectorFloat<?> query, FusedPQ.PackedNeighbors packedNeighbors,
                             ByteSequence<?> neighborCodes, VectorFloat<?> results, ExactScoreFunction esf,
                             VectorSimilarityFunction vsf) {
        this.pq = pq;
        this.hierarchyCachedFeatures = hierarchyCachedFeatures;
        this.query = query;
        this.esf = esf;
        this.packedNeighbors = packedNeighbors;
        this.neighborCodes = neighborCodes;
        this.vsf = vsf;
        this.origin = -1;

        // compute partialSums
        // cosine similarity is a special case where we need to compute the squared magnitude of the query
        // in the same loop, so we skip this and compute it in the cosine constructor
        partialSums = pq.reusablePartialSums();
        if (vsf != VectorSimilarityFunction.COSINE) {
            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, vsf, partialSums);
            }
        }
    }

    @Override
    public boolean supportsSimilarityToNeighbors() {
        return true;
    }

    @Override
    public void enableSimilarityToNeighbors(int origin) {
        if (this.origin != origin) {
            this.origin = origin;
            packedNeighbors.readInto(origin, neighborCodes);
        }
    }

    @Override
    public float similarityTo(int node2) {
        if (!hierarchyCachedFeatures.containsKey(node2)) {
            throw new IllegalArgumentException("Node " + node2 + " is not in the hierarchy");
        }

        var code2 = (FusedPQ.FusedPQInlineSource) hierarchyCachedFeatures.get(node2);
        float sim = VectorUtil.assembleAndSum(partialSums, pq.getClusterCount(), code2.getCode());
        return distanceToScore(sim);
    }

    @Override
    public float similarityToNeighbor(int origin, int neighborIndex) {
        if (this.origin != origin) {
            throw new IllegalArgumentException("origin must be the same as the origin used to enable similarityToNeighbor");
        }
        int position = neighborIndex * pq.getSubspaceCount();
        float sim = VectorUtil.assembleAndSum(partialSums, pq.getClusterCount(), neighborCodes, position, pq.getSubspaceCount());
        return distanceToScore(sim);
    }

    protected abstract float distanceToScore(float distance);

    static class DotProductDecoder extends FusedPQDecoder {
        public DotProductDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                 Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                 VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                 ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors, neighborCodes, results, esf, VectorSimilarityFunction.DOT_PRODUCT);
        }

        @Override
        protected float distanceToScore(float distance) {
            return (distance + 1) / 2;
        }
    }

    static class EuclideanDecoder extends FusedPQDecoder {
        public EuclideanDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors, neighborCodes, results, esf, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        protected float distanceToScore(float distance) {
            return 1 / (1 + distance);
        }
    }


    // CosineDecoder differs from DotProductDecoder/EuclideanDecoder because there are two different tables of fragments to sum: query to codebook entry dot products,
    // and codebook entry to codebook entry dot products. The latter can be calculated once per ProductQuantization.
    static class CosineDecoder extends FusedPQDecoder {
        private final float queryMagnitudeSquared;
        private final VectorFloat<?> partialSquaredMagnitudes;

        protected CosineDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors, neighborCodes, results, esf, VectorSimilarityFunction.COSINE);

            // this part is not query-dependent, so we can cache it
            partialSquaredMagnitudes = pq.partialSquaredMagnitudes().updateAndGet(current -> {
                if (current != null) {
                    return current;
                }

                var partialSquaredMagnitudes = vts.createFloatVector(pq.getSubspaceCount() * pq.getClusterCount());
                for (int m = 0; m < pq.getSubspaceCount(); ++m) {
                    int size = pq.subvectorSizesAndOffsets[m][0];
                    var codebook = pq.codebooks[m];
                    float minPartialMagnitude = Float.POSITIVE_INFINITY;
                    float maxPartialMagnitude = 0;
                    for (int j = 0; j < pq.getClusterCount(); ++j) {
                        var partialMagnitude = VectorUtil.dotProduct(codebook, j * size, codebook, j * size, size);
                        minPartialMagnitude = Math.min(minPartialMagnitude, partialMagnitude);
                        maxPartialMagnitude = Math.max(maxPartialMagnitude, partialMagnitude);
                        partialSquaredMagnitudes.set((m * pq.getClusterCount()) + j, partialMagnitude);
                    }
                }
                return partialSquaredMagnitudes;
            });

            // compute partialSums
            VectorFloat<?> center = pq.globalCentroid;
            float queryMagSum = 0.0f;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                // cosine numerator is the same partial sums as if we were using DOT_PRODUCT
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, VectorSimilarityFunction.DOT_PRODUCT, partialSums);
                queryMagSum += VectorUtil.dotProduct(centeredQuery, offset, centeredQuery, offset, size);
            }

            this.queryMagnitudeSquared = queryMagSum;
        }

        @Override
        public float similarityTo(int node2) {
            if (!hierarchyCachedFeatures.containsKey(node2)) {
                throw new IllegalArgumentException("Node " + node2 + " is not in the hierarchy");
            }

            var code2 = (FusedPQ.FusedPQInlineSource) hierarchyCachedFeatures.get(node2);
            float cos = VectorUtil.pqDecodedCosineSimilarity(code2.getCode(), 0, pq.getSubspaceCount(), pq.getClusterCount(), partialSums, partialSquaredMagnitudes, queryMagnitudeSquared);
            return distanceToScore(cos);
        }

        @Override
        public float similarityToNeighbor(int origin, int neighborIndex) {
            if (this.origin != origin) {
                throw new IllegalArgumentException("origin must be the same as the origin used to enable similarityToNeighbor");
            }
            int position = neighborIndex * pq.getSubspaceCount();
            float cos = VectorUtil.pqDecodedCosineSimilarity(neighborCodes, position, pq.getSubspaceCount(), pq.getClusterCount(), partialSums, partialSquaredMagnitudes, queryMagnitudeSquared);
            return distanceToScore(cos);
        }


        protected float distanceToScore(float distance) {
            return (1 + distance) / 2;
        };
    }

    public static FusedPQDecoder newDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures, VectorFloat<?> query,
                                            ByteSequence<?> reusableNeighborCodes, VectorFloat<?> results,
                                            VectorSimilarityFunction similarityFunction, ExactScoreFunction esf) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new DotProductDecoder(neighbors, pq, hierarchyCachedFeatures, query, reusableNeighborCodes, results, esf);
            case EUCLIDEAN:
                return new EuclideanDecoder(neighbors, pq, hierarchyCachedFeatures, query, reusableNeighborCodes, results, esf);
            case COSINE:
                return new CosineDecoder(neighbors, pq, hierarchyCachedFeatures, query, reusableNeighborCodes, results, esf);
            default:
                throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
        }
    }
}
