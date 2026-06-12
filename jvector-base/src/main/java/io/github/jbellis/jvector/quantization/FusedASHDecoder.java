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

import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;

import java.util.Objects;

/**
 * Scores FusedASH neighborhood blocks.
 *
 * <p>The fused layout is intentionally block-oriented: {@link #enableSimilarityToNeighbors(int)}
 * reads and scores the whole packed neighborhood for one origin node, and
 * {@link #similarityToNeighbor(int, int)} becomes a cheap array lookup. This preserves the
 * existing fused-feature search API while avoiding one-neighbor-at-a-time decode work in the
 * hot graph-expansion loop.</p>
 *
 * <p>This class is the scalar/reference implementation over the final FusedASH block layout.
 * SIMD kernels should consume the same byte layout and produce the same lane scores.</p>
 *
 * <p>FusedASH has two primary scoring paths:</p>
 *
 * <ul>
 *   <li>L0 neighborhood scoring uses fused packed ASH blocks via
 *       {@link #enableSimilarityToNeighbors(int)} and
 *       {@link #similarityToNeighbor(int, int)}.</li>
 *   <li>Upper-layer/direct node scoring uses canonical per-node ASH source
 *       features via {@link #similarityTo(int)}. Upper layers are not fused.</li>
 * </ul>
 *
 * <p>FusedASH lane headers are expected to contain projection-adjusted offsets:</p>
 *
 * <pre>
 *   offset = &lt;x, μ&gt; - ||μ||² - scale * &lt;Aμ, code&gt;
 * </pre>
 *
 * <p>Given that encode-time adjustment, query-time scoring is:</p>
 *
 * <pre>
 *   score = scale * &lt;Aq, code&gt; + &lt;q, μ&gt; + offset
 * </pre>
 */
public final class FusedASHDecoder implements ScoreFunction.ApproximateScoreFunction {

    /**
     * Source of fused neighborhood bytes. The implementation is provided by the graph feature,
     * typically by reading the FusedASH inline payload for {@code origin} into {@code dest}.
     */
    public interface PackedNeighborhoods {
        void readInto(int origin, byte[] dest);
        int maxDegree();
        int featureSize();
    }

    /**
     * Converts an in-memory upper-layer/source feature into the canonical ASH vector
     * used for primary direct-node ASH scoring via {@link #similarityTo(int)}.
     */
    @FunctionalInterface
    public interface InlineSourceAccessor {
        AsymmetricHashing.QuantizedVector vector(FusedFeature.InlineSource source);
    }

    private final AsymmetricHashing ash;
    private final PackedNeighborhoods packedNeighborhoods;
    private final Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures;
    private final InlineSourceAccessor inlineSourceAccessor;
    private final ASHScorer.ASHScoreFunction sourceScorer;

    private final int maxDegree;
    private final int blockSize;
    private final int blocksPerNode;
    private final int quantizedDim;
    private final int bitsPerDimension;

    private final byte[] neighborhoodScratch;
    private final float[] neighborScores;

    private final float[] queryLut;
    private final float[] dotQMuByLandmark;

    private int origin = -1;

    public FusedASHDecoder(
            PackedNeighborhoods packedNeighborhoods,
            AsymmetricHashing ash,
            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
            InlineSourceAccessor inlineSourceAccessor,
            VectorFloat<?> query,
            byte[] reusableNeighborhoodBytes,
            float[] reusableScores,
            int blockSize,
            VectorSimilarityFunction similarityFunction) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException("FusedASH supports DOT_PRODUCT only");
        }

        this.packedNeighborhoods = Objects.requireNonNull(packedNeighborhoods, "packedNeighborhoods");
        this.ash = Objects.requireNonNull(ash, "ash");
        this.hierarchyCachedFeatures = Objects.requireNonNull(
                hierarchyCachedFeatures,
                "hierarchyCachedFeatures"
        );
        this.inlineSourceAccessor = Objects.requireNonNull(
                inlineSourceAccessor,
                "inlineSourceAccessor"
        );
        this.sourceScorer = new ASHScorer(ash).scoreFunctionFor(query, similarityFunction);

        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        FusedASHLayout.validateBlockSize(blockSize);

        this.maxDegree = packedNeighborhoods.maxDegree();
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }

        this.blockSize = blockSize;
        this.blocksPerNode = FusedASHLayout.blocksPerNode(maxDegree, blockSize);
        this.quantizedDim = ash.quantizedDim;
        this.bitsPerDimension = ash.bitsPerDimension;

        int expectedFeatureSize = FusedASHLayout.featureSize(
                maxDegree,
                quantizedDim,
                bitsPerDimension,
                blockSize
        );
        if (packedNeighborhoods.featureSize() != expectedFeatureSize) {
            throw new IllegalArgumentException(
                    "FusedASH feature size mismatch: expected " + expectedFeatureSize +
                            ", got " + packedNeighborhoods.featureSize());
        }
        if (reusableNeighborhoodBytes.length < expectedFeatureSize) {
            throw new IllegalArgumentException("reusableNeighborhoodBytes is smaller than FusedASH feature size");
        }
        if (reusableScores.length < maxDegree) {
            throw new IllegalArgumentException("reusableScores is smaller than maxDegree");
        }

        this.neighborhoodScratch = reusableNeighborhoodBytes;
        this.neighborScores = reusableScores;

        int groups = FusedASHLayout.codeGroups(quantizedDim, bitsPerDimension);
        this.queryLut = new float[groups * 16];
        this.dotQMuByLandmark = new float[ash.landmarkCount];

        precomputeQuery(query, queryLut, dotQMuByLandmark);
    }

    public static FusedASHDecoder newDecoder(
            PackedNeighborhoods packedNeighborhoods,
            AsymmetricHashing ash,
            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
            InlineSourceAccessor inlineSourceAccessor,
            VectorFloat<?> query,
            byte[] reusableNeighborhoodBytes,
            float[] reusableScores,
            int blockSize,
            VectorSimilarityFunction similarityFunction) {
        return new FusedASHDecoder(
                packedNeighborhoods,
                ash,
                hierarchyCachedFeatures,
                inlineSourceAccessor,
                query,
                reusableNeighborhoodBytes,
                reusableScores,
                blockSize,
                similarityFunction);
    }

    @Override
    public boolean supportsSimilarityToNeighbors() {
        return true;
    }

    @Override
    public void enableSimilarityToNeighbors(int origin) {
        if (this.origin == origin) {
            return;
        }

        this.origin = origin;
        packedNeighborhoods.readInto(origin, neighborhoodScratch);
        scoreNeighborhood();
    }

    /**
     * Scores a directly-addressed node using its canonical ASH source feature.
     *
     * <p>This method is used for upper-layer graph traversal and other direct-node
     * primary scoring paths. It must not call the configured reranker. Missing
     * source features indicate an invalid fused graph/source-feature layout.</p>
     */
    @Override
    public float similarityTo(int node2) {
        FusedFeature.InlineSource source = hierarchyCachedFeatures.get(node2);
        if (source == null) {
            throw new IllegalStateException(
                    "Missing FusedASH source feature for direct ASH scoring of node " + node2 +
                            ". FusedASH requires canonical ASH source features for every upper-layer node " +
                            "that can be scored directly."
            );
        }

        AsymmetricHashing.QuantizedVector vector = inlineSourceAccessor.vector(source);
        if (vector == null) {
            throw new IllegalStateException(
                    "FusedASH source accessor returned null for node " + node2
            );
        }

        return sourceScorer.similarityTo(vector);
    }

    @Override
    public float similarityToNeighbor(int origin, int neighborIndex) {
        if (this.origin != origin) {
            throw new IllegalArgumentException(
                    "origin must be the same as the origin used to enable similarityToNeighbor");
        }
        if (neighborIndex < 0 || neighborIndex >= maxDegree) {
            throw new IllegalArgumentException("neighborIndex out of range: " + neighborIndex);
        }
        return neighborScores[neighborIndex];
    }

    private void scoreNeighborhood() {
        for (int block = 0; block < blocksPerNode; block++) {
            int blockOffset = FusedASHLayout.blockOffset(block, quantizedDim, bitsPerDimension, blockSize);
            int laneBase = block * blockSize;
            int lanes = Math.min(blockSize, maxDegree - laneBase);

            for (int lane = 0; lane < lanes; lane++) {
                neighborScores[laneBase + lane] = FusedASHLayout.scoreLane(
                        neighborhoodScratch,
                        blockOffset,
                        lane,
                        quantizedDim,
                        bitsPerDimension,
                        blockSize,
                        queryLut,
                        dotQMuByLandmark);
            }
        }
    }

    private void precomputeQuery(VectorFloat<?> query, float[] lut, float[] dotQMu) {
        final int D = ash.originalDimension;
        final int d = ash.quantizedDim;
        final float[][] A = ash.stiefelTransform.AFloat;

        final VectorUtilSupport vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

        float[] qArr = new float[D];
        for (int i = 0; i < D; i++) {
            qArr[i] = query.get(i);
        }

        float[] qProj = new float[d];
        for (int j = 0; j < d; j++) {
            qProj[j] = vecUtil.ashDotRow(A[j], qArr);
        }

        FusedASHLayout.buildQueryLut(qProj, d, bitsPerDimension, lut);

        for (int c = 0; c < ash.landmarkCount; c++) {
            dotQMu[c] = VectorUtil.dotProduct(query, ash.landmarks[c]);
        }
    }
}