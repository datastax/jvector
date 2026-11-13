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
 * A decoder borrowing view-owned query scratch is invalidated by the next query on that view.
 * Standalone constructors own independent preparation state.</p>
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
 *   similarity = (1 + rawDotProduct) / 2   (unclamped, the exact DOT_PRODUCT scale)
 * </pre>
 */
public final class FusedASHDecoder implements ScoreFunction.AdaptiveApproximateScoreFunction {

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

    /**
     * Reusable preparation and scoring arrays for one graph view and one ASH model.
     * Starting a new query invalidates that view's previous decoder; use a separate
     * view for overlapping searches. Standalone constructors retain independent state.
     */
    public static final class QueryScratch {
        private final AsymmetricHashing ash;
        private final ASHScorer scorer;
        private final ASHScorer.QueryPrecompute prepared;
        private final float[] input;
        private final float[] lut;
        private final byte[] neighborhoodBytes;
        private final float[] scores;
        private final int blockSize;
        private long generation;

        public QueryScratch(AsymmetricHashing ash, int maxDegree, int blockSize) {
            this.ash = Objects.requireNonNull(ash, "ash");
            int size = FusedASHLayout.featureSize(maxDegree, ash.quantizedDim, ash.bitsPerDimension, blockSize);
            this.blockSize = blockSize;
            this.scorer = new ASHScorer(ash);
            this.prepared = scorer.newQueryPrecompute();
            this.input = new float[ash.originalDimension];
            this.lut = new float[Math.multiplyExact(FusedASHLayout.codeGroups(ash.quantizedDim, ash.bitsPerDimension), 16)];
            this.neighborhoodBytes = new byte[size];
            this.scores = new float[maxDegree];
        }
    }

    private final QueryScratch queryScratch;
    private final long generation;
    private final ASHQueryPreparer.State externalPrepared;
    private final long preparedGeneration;

    private final PackedNeighborhoods packedNeighborhoods;
    private final Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures;
    private final InlineSourceAccessor inlineSourceAccessor;
    private final ASHScorer.ASHScoreFunction sourceScorer;
    private final float dotProductBound;

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

        this(packedNeighborhoods, ash, hierarchyCachedFeatures, inlineSourceAccessor, query,
                reusableNeighborhoodBytes, reusableScores, blockSize, similarityFunction, null, null);
    }

    private FusedASHDecoder(
            PackedNeighborhoods packedNeighborhoods, AsymmetricHashing ash,
            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyCachedFeatures,
            InlineSourceAccessor inlineSourceAccessor, VectorFloat<?> query,
            byte[] reusableNeighborhoodBytes, float[] reusableScores, int blockSize,
            VectorSimilarityFunction similarityFunction, QueryScratch scratch,
            ASHQueryPreparer.State externalPrepared) {

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
        if (scratch != null && (scratch.ash != ash || scratch.blockSize != blockSize
                || scratch.scores.length != maxDegree)) {
            throw new IllegalArgumentException("Query scratch belongs to a different ASH model or graph layout");
        }
        this.queryScratch = scratch;
        this.externalPrepared = externalPrepared;
        this.preparedGeneration = externalPrepared == null ? 0 : externalPrepared.generation;
        var scorer = scratch == null ? new ASHScorer(ash) : scratch.scorer;
        ASHScorer.QueryPrecompute preparedQuery;
        if (externalPrepared != null) {
            externalPrepared.check(preparedGeneration);
            this.generation = scratch == null ? 0 : ++scratch.generation;
            preparedQuery = externalPrepared.prepared;
        } else if (scratch == null) {
            this.generation = 0;
            preparedQuery = scorer.precomputeQuery(query);
        } else {
            // Invalidate before projection: failure must not leave a usable old decoder
            // whose query buffers have been partially overwritten.
            this.generation = ++scratch.generation;
            preparedQuery = scratch.prepared;
            scorer.prepareQuery(query, scratch.input, preparedQuery);
        }
        this.neighborhoodScorer = new ASHNeighborhoodScorer(ash, preparedQuery, maxDegree, blockSize,
                externalPrepared == null ? (scratch == null ? null : scratch.lut) : externalPrepared.lut,
                externalPrepared != null);
        this.sourceScorer = scorer.scoreFunctionFor(preparedQuery, neighborhoodScorer.queryLut());
        this.dotProductBound = scorer.dotProductBound(preparedQuery);
    }

    /** The query's length times the largest encoded vector length, taken when this decoder was prepared. */
    @Override
    public float dotProductBound() {
        return dotProductBound;
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

    /** Creates a query decoder using scratch owned by its graph view. */
    public static FusedASHDecoder newDecoder(
            PackedNeighborhoods neighborhoods, AsymmetricHashing ash,
            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyFeatures,
            InlineSourceAccessor sourceAccessor, VectorFloat<?> query, QueryScratch scratch,
            int blockSize, VectorSimilarityFunction similarityFunction) {
        Objects.requireNonNull(scratch, "scratch");
        return new FusedASHDecoder(neighborhoods, ash, hierarchyFeatures, sourceAccessor, query,
                scratch.neighborhoodBytes, scratch.scores, blockSize, similarityFunction, scratch, null);
    }

    /** Binds caller-owned preparation; no projection or LUT construction is repeated. */
    public static FusedASHDecoder newDecoder(
            PackedNeighborhoods neighborhoods, AsymmetricHashing ash,
            Int2ObjectHashMap<FusedFeature.InlineSource> hierarchyFeatures,
            InlineSourceAccessor sourceAccessor,
            io.github.jbellis.jvector.graph.similarity.PreparedQuery preparedQuery,
            QueryScratch scratch, int blockSize) {
        Objects.requireNonNull(scratch, "scratch");
        ASHQueryPreparer.State prepared = ASHQueryPreparer.state(ash, preparedQuery, true);
        return new FusedASHDecoder(neighborhoods, ash, hierarchyFeatures, sourceAccessor,
                prepared.queryVector(), scratch.neighborhoodBytes, scratch.scores, blockSize,
                VectorSimilarityFunction.DOT_PRODUCT, scratch, prepared);
    }

    private void checkQueryLifetime() {
        if (externalPrepared != null) externalPrepared.check(preparedGeneration);
        if (queryScratch != null && generation != queryScratch.generation) {
            throw new IllegalStateException("FusedASH decoder was invalidated by a new query on the same graph view");
        }
    }

    @Override
    public boolean supportsSimilarityToNeighbors() {
        return true;
    }

    @Override
    public void enableSimilarityToNeighbors(int origin) {
        checkQueryLifetime();
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
        checkQueryLifetime();
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
        checkQueryLifetime();
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
