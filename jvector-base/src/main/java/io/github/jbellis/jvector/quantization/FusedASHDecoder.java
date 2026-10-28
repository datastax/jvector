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
 * <p>Each decoder and its mutable scratch/cache are confined to one graph view/search thread.
 * The underlying prepared neighborhood scorer may be shared only with separate scratch storage.</p>
 *
 * <p>Scalar and SIMD execution share the same packed layout. The property
 * {@code jvector.ash.blockKernel=auto|scalar|simd} selects execution once per decoder;
 * forced SIMD fails at construction if the backend cannot provide it.</p>
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
 * <p>Given that encode-time adjustment, the raw dot-product estimate is:</p>
 *
 * <pre>
 *   rawDotProduct = scale * &lt;Aq, code&gt; + &lt;q, μ&gt; + offset
 *   similarity = max(0, (1 + rawDotProduct) / 2)
 * </pre>
 */
public final class FusedASHDecoder implements ScoreFunction.ApproximateScoreFunction {

    /**
     * Source of fused neighborhood bytes. The implementation is provided by the graph feature,
     * typically by reading the FusedASH inline payload for {@code origin} into {@code dest}.
     */
    public interface PackedNeighborhoods {
        /** Replaces the complete serialized payload for {@code origin} in caller-owned storage. */
        void readInto(int origin, byte[] dest);

        /** Returns the scored capacity, including padding, of each neighborhood. */
        int maxDegree();

        /** Returns the complete serialized byte count for the configured layout. */
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

    private final PackedNeighborhoods packedNeighborhoods;
    private final Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures;
    private final InlineSourceAccessor inlineSourceAccessor;
    private final ASHScorer.ASHScoreFunction sourceScorer;

    private final int maxDegree;

    private final byte[] neighborhoodScratch;
    private final float[] neighborScores;

    private final ASHNeighborhoodScorer neighborhoodScorer;

    private int origin = -1;

    /**
     * Creates a query-local decoder and shares one projection between L0 and upper-layer scoring.
     *
     * @param packedNeighborhoods source of complete fused payloads with a fixed layout
     * @param ash compressor that encoded the graph's ASH features
     * @param hierarchyCachedFeatures canonical features available for direct/upper-layer scoring
     * @param inlineSourceAccessor reads a canonical ASH vector from a cached feature
     * @param query original-dimensional query, normalized by the caller when appropriate
     * @param reusableNeighborhoodBytes decoder-owned scratch, at least the complete feature size
     * @param reusableScores decoder-owned scratch, at least the source's maximum degree
     * @param blockSize serialized block width: 8, 16, or 32
     * @param similarityFunction must be DOT_PRODUCT
     * @throws IllegalArgumentException if dimensions, payload size, or scratch capacity disagree
     * @throws UnsupportedOperationException for another similarity function
     */
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
        Objects.requireNonNull(ash, "ash");
        this.hierarchyCachedFeatures = Objects.requireNonNull(
                hierarchyCachedFeatures,
                "hierarchyCachedFeatures"
        );
        this.inlineSourceAccessor = Objects.requireNonNull(
                inlineSourceAccessor,
                "inlineSourceAccessor"
        );
        Objects.requireNonNull(query, "query");
        if (query.length() != ash.originalDimension) {
            throw new IllegalArgumentException("ASH query dimension must be " + ash.originalDimension);
        }

        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        FusedASHLayout.validateBlockSize(blockSize);

        this.maxDegree = packedNeighborhoods.maxDegree();
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }

        int quantizedDim = ash.quantizedDim;
        int bitsPerDimension = ash.bitsPerDimension;

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

        // Upper-layer source scoring and L0 block scoring use the same Aq and
        // <q, landmark> values. Prepare them once, then build each scoring form.
        var scorer = new ASHScorer(ash);
        var preparedQuery = scorer.precomputeQuery(query);
        this.sourceScorer = scorer.scoreFunctionFor(preparedQuery);

        this.neighborhoodScorer = new ASHNeighborhoodScorer(ash, preparedQuery, maxDegree, blockSize);
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

        // Invalidate before I/O: a failed read must not leave partially replaced scores cached.
        this.origin = -1;
        packedNeighborhoods.readInto(origin, neighborhoodScratch);
        scoreNeighborhood();
        this.origin = origin;
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

        return ASHScorer.toSimilarity(sourceScorer.similarityTo(vector));
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

    /** Uses the same prepared scorer as the isolated benchmark; graph I/O stays in this decoder. */
    private void scoreNeighborhood() {
        neighborhoodScorer.score(neighborhoodScratch, neighborScores);
    }

    @Override
    public String toString() {
        return "FusedASHDecoder[" + neighborhoodScorer + "]";
    }
}
