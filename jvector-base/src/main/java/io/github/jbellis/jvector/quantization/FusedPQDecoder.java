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

import java.util.Arrays;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use Quick(er) ADC-style transposed vectors fused into a graph.
 */
public abstract class FusedPQDecoder implements ScoreFunction.ApproximateScoreFunction {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    protected final ProductQuantization pq;
    Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures;
    protected final VectorFloat<?> query;
    protected final ExactScoreFunction esf;
    protected final ByteSequence<?> partialQuantizedSums;
    // connected to the Graph View by caller
    protected final FusedPQ.PackedNeighbors packedNeighbors;
    // caller passes this to us for re-use across calls
    protected final VectorFloat<?> results;
    // caller passes this to us for re-use across calls
    protected final ByteSequence<?> neighborCodes;
    // decoder state
    protected final VectorFloat<?> partialSums;
    protected final VectorFloat<?> partialBestDistances;
    protected final int invocationThreshold;
    protected int invocations = 0;
    protected float bestDistance;
    protected float worstDistance;
    protected float delta;
    protected boolean supportsQuantizedSimilarity = false;
    protected final VectorSimilarityFunction vsf;

    // Implements section 3.4 of "Quicker ADC : Unlocking the Hidden Potential of Product Quantization with SIMD"
    // The main difference is that since our graph structure rapidly converges towards the best results,
    // we don't need to scan K values to have enough confidence that our worstDistance bound is reasonable.
    protected FusedPQDecoder(ProductQuantization pq,
                             Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                             VectorFloat<?> query, int invocationThreshold, FusedPQ.PackedNeighbors packedNeighbors,
                             ByteSequence<?> neighborCodes, VectorFloat<?> results, ExactScoreFunction esf,
                             VectorSimilarityFunction vsf) {
        this.pq = pq;
        this.hierarchyCachedFeatures = hierarchyCachedFeatures;
        this.query = query;
        this.esf = esf;
        this.invocationThreshold = invocationThreshold;
        this.packedNeighbors = packedNeighbors;
        this.neighborCodes = neighborCodes;
        this.results = results;
        this.vsf = vsf;

        // compute partialSums, partialBestDistances, and bestDistance from the codebooks
        // cosine similarity is a special case where we need to compute the squared magnitude of the query
        // in the same loop, so we skip this and compute it in the cosine constructor
        partialSums = pq.reusablePartialSums();
        partialBestDistances = pq.reusablePartialBestDistances();
        if (vsf != VectorSimilarityFunction.COSINE) {
            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, vsf, partialSums, partialBestDistances);
            }
            bestDistance = VectorUtil.sum(partialBestDistances);
        }

        // these will be computed by edgeLoadingSimilarityTo as we search
        partialQuantizedSums = pq.reusablePartialQuantizedSums();
    }

    @Override
    public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
        results.zero();

        packedNeighbors.readInto(origin, neighborCodes);

        if (supportsQuantizedSimilarity) {
            // we have seen enough data to compute `delta`, so take the fast path using the permuted nodes
            VectorUtil.bulkShuffleQuantizedSimilarity(neighborCodes, pq.compressedVectorSize(), partialQuantizedSums, delta, bestDistance, results, vsf);
        } else {
            // we have not yet computed worstDistance or delta, so we need to follow the unquantized path
            VectorUtil.bulkShuffleRawSimilarity(neighborCodes, pq.compressedVectorSize(), partialSums, results);

            // update worstDistance from our new set of results
            var maxDegree = packedNeighbors.maxDegree();
            for (int i = 0; i < maxDegree; i++) {
                var result = results.get(i);
                invocations++;
                updateWorstDistance(result);
                results.set(i, distanceToScore(result));
            }

            // once we have enough data, set up delta, partialQuantizedSums, and partialQuantizedMagnitudes for the fast path
            if (invocations >= invocationThreshold) {
                delta = (worstDistance - bestDistance) / 65535;
                VectorUtil.quantizePartials(delta, partialSums, partialBestDistances, partialQuantizedSums);
                supportsQuantizedSimilarity = true;
            }
        }
        return results;
    }

    @Override
    public boolean supportsEdgeLoadingSimilarity() {
        return true;
    }

    protected float decodedSimilarity(ByteSequence<?> encoded) {
        return VectorUtil.assembleAndSum(partialSums, pq.getClusterCount(), encoded, 0, encoded.length());
    }

    protected abstract float distanceToScore(float distance);

    protected abstract void updateWorstDistance(float distance);

    static class DotProductDecoder extends FusedPQDecoder {
        public DotProductDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                 Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                 VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                 ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors.maxDegree(), neighbors, neighborCodes, results, esf, VectorSimilarityFunction.DOT_PRODUCT);
            worstDistance = Float.MAX_VALUE; // initialize at best value, update as we search
        }

        @Override
        public float similarityTo(int node2) {
            if (!hierarchyCachedFeatures.containsKey(node2)) {
                throw new IllegalArgumentException("Node " + node2 + " is not in the hierarchy");
            }
            var code2 = (FusedPQ.FusedADCInlineSource) hierarchyCachedFeatures.get(node2);
            return distanceToScore(decodedSimilarity(code2.getCode()));
        }

        @Override
        protected float distanceToScore(float distance) {
            return (distance + 1) / 2;
        }

        @Override
        protected void updateWorstDistance(float distance) {
            worstDistance = Math.min(worstDistance, distance);
        }
    }

    static class EuclideanDecoder extends FusedPQDecoder {
        public EuclideanDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors.maxDegree(), neighbors, neighborCodes, results, esf, VectorSimilarityFunction.EUCLIDEAN);
            worstDistance = 0; // initialize at best value, update as we search
        }

        @Override
        public float similarityTo(int node2) {
            if (!hierarchyCachedFeatures.containsKey(node2)) {
                throw new IllegalArgumentException("Node " + node2 + " is not in the hierarchy");
            }
            var code2 = (FusedPQ.FusedADCInlineSource) hierarchyCachedFeatures.get(node2);
            return distanceToScore(decodedSimilarity(code2.getCode()));
        }

        @Override
        protected float distanceToScore(float distance) {
            return 1 / (1 + distance);
        }

        @Override
        protected void updateWorstDistance(float distance) {
            worstDistance = Math.max(worstDistance, distance);
        }
    }


    // CosineDecoder differs from DotProductDecoder/EuclideanDecoder because there are two different tables of quantized fragments to sum: query to codebook entry dot products,
    // and codebook entry to codebook entry dot products. The latter can be calculated once per ProductQuantization, but for lookups to go at the appropriate speed, they must
    // also be quantized. We use a similar quantization to partial sums, but we know exactly the worst/best bounds, so overflow does not matter.
    static class CosineDecoder extends FusedPQDecoder {
        private final float queryMagnitudeSquared;
        private final VectorFloat<?> partialSquaredMagnitudes;
        private final ByteSequence<?> partialQuantizedSquaredMagnitudes;
        // prior to quantization, we need a good place on-heap to aggregate these for worstDistance tracking/result calculation
        private final float[] resultSumAggregates;
        private final float[] resultMagnitudeAggregates;
        // store these to avoid repeated volatile lookups
        private float minSquaredMagnitude;
        private float squaredMagnitudeDelta;

        protected CosineDecoder(FusedPQ.PackedNeighbors neighbors, ProductQuantization pq,
                                Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
                                VectorFloat<?> query, ByteSequence<?> neighborCodes, VectorFloat<?> results,
                                ExactScoreFunction esf) {
            super(pq, hierarchyCachedFeatures, query, neighbors.maxDegree(), neighbors, neighborCodes, results, esf, VectorSimilarityFunction.COSINE);
            worstDistance = Float.MAX_VALUE; // initialize at best value, update as we search

            // this part is not query-dependent, so we can cache it
            partialSquaredMagnitudes = pq.partialSquaredMagnitudes().updateAndGet(current -> {
                if (current != null) {
                    squaredMagnitudeDelta = pq.squaredMagnitudeDelta;
                    minSquaredMagnitude = pq.minSquaredMagnitude;
                    return current;
                }

                // we only need these for quantization, minSquaredMagnitude/squaredMagnitudeDelta are sufficient for dequantization
                float maxMagnitude = 0;
                VectorFloat<?> partialMinMagnitudes = vts.createFloatVector(pq.getSubspaceCount());

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

                    partialMinMagnitudes.set(m, minPartialMagnitude);
                    maxMagnitude += maxPartialMagnitude;
                    minSquaredMagnitude += minPartialMagnitude;
                }
                squaredMagnitudeDelta = (maxMagnitude - minSquaredMagnitude) / 65535;
                var partialQuantizedSquaredMagnitudes = vts.createByteSequence(pq.getSubspaceCount() * pq.getClusterCount() * 2);
                VectorUtil.quantizePartials(squaredMagnitudeDelta, partialSquaredMagnitudes, partialMinMagnitudes, partialQuantizedSquaredMagnitudes);

                // publish for future use in other decoders using this PQ
                pq.squaredMagnitudeDelta = squaredMagnitudeDelta;
                pq.minSquaredMagnitude = minSquaredMagnitude;
                pq.partialQuantizedSquaredMagnitudes().set(partialQuantizedSquaredMagnitudes);

                return partialSquaredMagnitudes;
            });
            partialQuantizedSquaredMagnitudes = pq.partialQuantizedSquaredMagnitudes().get();

            // compute partialSums, partialBestDistances, bestDistance, and queryMagnitudeSquared from the codebooks
            VectorFloat<?> center = pq.globalCentroid;
            float queryMagSum = 0.0f;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                // cosine numerator is the same partial sums as if we were using DOT_PRODUCT
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, VectorSimilarityFunction.DOT_PRODUCT, partialSums, partialBestDistances);
                queryMagSum += VectorUtil.dotProduct(centeredQuery, offset, centeredQuery, offset, size);
            }
            this.queryMagnitudeSquared = queryMagSum;
            bestDistance = VectorUtil.sum(partialBestDistances);

            this.resultSumAggregates = new float[neighbors.maxDegree()];
            this.resultMagnitudeAggregates = new float[neighbors.maxDegree()];
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            results.zero();

            packedNeighbors.readInto(origin, neighborCodes);

            if (supportsQuantizedSimilarity) {
                // we have seen enough data to compute `delta`, so take the fast path using the permuted nodes
                VectorUtil.bulkShuffleQuantizedSimilarityCosine(neighborCodes, pq.compressedVectorSize(), partialQuantizedSums, delta, bestDistance, partialQuantizedSquaredMagnitudes, squaredMagnitudeDelta, minSquaredMagnitude, queryMagnitudeSquared, results);
            } else {
                // we have not yet computed worstDistance or delta, so we need to assemble the results manually
                VectorUtil.bulkShuffleRawSimilarityCosine(neighborCodes, pq.compressedVectorSize(), partialSums, partialSquaredMagnitudes, resultSumAggregates, resultMagnitudeAggregates);

                // update worstDistance from our new set of results
                var maxDegree = packedNeighbors.maxDegree();
                for (int i = 0; i < maxDegree; i++) {
                    updateWorstDistance(resultSumAggregates[i]);
                    var result = resultSumAggregates[i] / (float) Math.sqrt(resultMagnitudeAggregates[i] * queryMagnitudeSquared);
                    invocations++;
                    results.set(i, distanceToScore(result));
                }

                // once we have enough data, set up delta and partialQuantizedSums for the fast path
                if (invocations >= invocationThreshold) {
                    delta = (worstDistance - bestDistance) / 65535;
                    VectorUtil.quantizePartials(delta, partialSums, partialBestDistances, partialQuantizedSums);
                    supportsQuantizedSimilarity = true;
                }
            }

            return results;
        }

        @Override
        public float similarityTo(int node2) {
            if (!hierarchyCachedFeatures.containsKey(node2)) {
                throw new IllegalArgumentException("Node " + node2 + " is not in the hierarchy");
            }
            var code2 = (FusedPQ.FusedADCInlineSource) hierarchyCachedFeatures.get(node2);
            float cos = VectorUtil.pqDecodedCosineSimilarity(code2.getCode(), 0, pq.getSubspaceCount(), pq.getClusterCount(), partialSums, partialSquaredMagnitudes, queryMagnitudeSquared);
            return distanceToScore(cos);
        }

        protected float distanceToScore(float distance) {
            return (1 + distance) / 2;
        };

        protected void updateWorstDistance(float distance) {
            worstDistance = Math.min(worstDistance, distance);
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
