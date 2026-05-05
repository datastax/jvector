package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;
import java.util.Arrays;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction.ApproximateScoreFunction;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.MutableCompressedVectors;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import lombok.AllArgsConstructor;

public class AshVectors implements MutableCompressedVectors<VectorFloat<?>> {

    private static final AshDriverFactory ashDriverFactory = VectorizationProvider.getInstance().getAshDriverFactory();

    private final AsymmetricHashing ash;
    private final AshDriver driver;
    private final AshDriver.PackedVectors packed;
    private final int bitDepth;
    private final int nVecs;

    private final float[] scales;
    private final float[] offsets;
    private final byte[] landmarks;

    /** Creates an AshVectors instance with space allocated for `numVectors` vectors */
    public AshVectors(AsymmetricHashing ash, int bitDepth, int numVectors) {
        this.ash = ash;
        this.driver = ashDriverFactory.createDriver(bitDepth, ash.quantizedDim);
        this.bitDepth = bitDepth;
        this.nVecs = numVectors;

        // vectors not initialized at this stage
        this.packed = driver.create(numVectors);
        this.scales = new float[numVectors];
        this.offsets = new float[numVectors];
        this.landmarks = new byte[numVectors];
    }

    // ----------------------------------------
    // Methods from CompressedVectors
    // ----------------------------------------

    @Override
    public long ramBytesUsed() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'ramBytesUsed'");
    }

    @Override
    public void write(IndexWriter out, int version) throws IOException {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'write'");
    }

    @Override
    public int getOriginalSize() {
        return ash.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getCompressedSize'");
    }

    @Override
    public VectorCompressor<?> getCompressor() {
        return ash;
    }

    @Override
    public ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q,
            VectorSimilarityFunction similarityFunction) {
        return this.scoreFunctionFor(q, similarityFunction);
    }

    /**
     * Diversity-aware scoring is not supported for ASH at this stage.
     *
     * <p>
     * TODO: Define how diversity should be measured.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(
            int node1,
            VectorSimilarityFunction similarityFunction) {
        throw new UnsupportedOperationException("ASH diversity scoring not implemented");
    }

    @Override
    public ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new IllegalArgumentException("ASH quantizer only supports dot product");
        }
        var qp = AshQueryPrecompute.precomputeQuery(ash, q);

        return node -> {

            final int c = landmarks[node] & 0xFF; // unsigned [0,C)
            assert c < qp.C : "Invalid landmark id " + c + " for landmarkCount=" + qp.C;

            // scale = ||x − μ_c|| / √d  (precomputed at encode-time)
            final float scale = scales[node];

            // offset = <x, μ_c> − ||μ_c||²  (precomputed at encode-time)
            final float offset = offsets[node];

            final float sumTildeQ = qp.sumTildeQ[c];
            final float dotQMu = qp.dotQMu[c];

            final float scaleDifferential = 2f / ((float) (1 << bitDepth) - 1f);

            final var d = qp.d;
            final int tildeBase = c * d;

            final var innerDot = driver.asymmetricScorePackedInts(packed, node, qp.tildeQPool, tildeBase);

            // ASH dot-product approximation (multi-landmark, {0,1} encoding):
            //
            //   scale * (2⟨q̃_c, b⟩ − ⟨q̃_c, 1⟩) + ⟨q, μ_c⟩ + (⟨x, μ_c⟩ − ||μ_c||²)
            //
            // where q̃_c = A(q − μ_c) is precomputed once per query and per landmark c.
            return scale * (scaleDifferential * innerDot - sumTildeQ) + dotQMu + offset;
        };
    }

    @Override
    public int count() {
        return nVecs;
    }

    // ---------------------------------
    // Methods added by MutableCompressedVectors
    // ---------------------------------

    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        // NOTE:
        // Single-landmark baseline configuration (Table 2, C = 1).
        // All vectors are encoded relative to landmark μ_c (C=1 => μ_0 is the dataset mean).
        // The landmark id is always 0 in this mode.

        // Landmark assignment: nearest centroid by L2 distance
        byte landmark = 0;
        float bestDist = Float.POSITIVE_INFINITY;

        for (int c = 0; c < ash.landmarkCount; c++) {
            float dist = VectorUtil.squareL2Distance(vector, ash.landmarks[c]);
            if (dist < bestDist) {
                bestDist = dist;
                landmark = (byte) c;
            }
        }

        final VectorFloat<?> mu = ash.landmarks[landmark & 0xFF];

        // dest.encodeInto(vector, mu, landmark, landmarkNormSq[landmark], stiefelTransform);
        var landmarkNormSq = ash.landmarkNormSq[landmark];
        var stiefel = ash.stiefelTransform;

        // Compute <x, μ>
        final float dotXMu = VectorUtil.dotProduct(vector, mu);

        // Compute ||x_i − μ_i*||_2
        // We store the true L2 norm (not squared), matching the paper’s use in Eq. 6 normalization.
        final float sqDist = VectorUtil.squareL2Distance(vector, mu);
        final float residualNorm = (float) Math.sqrt(sqDist);
        final int quantizedDim = stiefel.quantizedDim();

        // Sanity: quantizedDim already accounts for physical header bits
        // Binary body uses exactly quantizedDim bits.
        assert quantizedDim > 0;

        // Store:
        //   scale  = ||x − μ|| / sqrt(d)
        //   offset = <x, μ> − ||μ||^2
        // NOTE: we still compute residualNorm (||x − μ||) even though we don't store it alone,
        // because the binarizer needs it for Eq. 6 normalization.
        final float invSqrtD = (float) (1.0 / Math.sqrt(quantizedDim));
        final float baseScale = residualNorm * invSqrtD;
        final float offset = dotXMu - landmarkNormSq;

        // Binary body: sign(A · (x − μ)) with Eq. 6 normalization inside quantizer
        // this.quantizeVectorInto(vector, residualNorm, stiefel, mu);

        var out = new float[ash.quantizedDim];
        NbAshProjector.projectTo(vector, residualNorm, ash.stiefelTransform, mu, out);

        int[] quant = NbAshProjector.quantizeProjection(bitDepth, out);

        var thinNorm = NbAshProjector.thinNormFromQuantProjection(bitDepth, quant);
        var dSqrt = Math.sqrt(quant.length);
        var scaleScale = dSqrt / thinNorm;
        var scale = (float) (baseScale * scaleScale);

        // binarizeIntoFromQuantized(quant);
        this.landmarks[ordinal] = landmark;
        this.offsets[ordinal] = offset;
        this.scales[ordinal] = scale;
        driver.packInts(quant, 0, this.packed, ordinal);
    }

    @Override
    public void setZero(int ordinal) {
        this.scales[ordinal] = 0;
        this.offsets[ordinal] = 0;
        this.landmarks[ordinal] = 0;

        // TODO create a setZero method for driver?
        final int[] zero = new int[ash.quantizedDim];
        Arrays.fill(zero, 0);
        driver.packInts(zero, ordinal, packed, ordinal);
    }

    // ---------------------------------
    // Functions used for graph building - symmetric scoring and BuildScoreProvider
    // ---------------------------------

    public ApproximateScoreFunction symmetricScoreFunctionFor(int baseNode, VectorSimilarityFunction vsf) {
        // var base = compressedVectors[baseNode];
        var scale0 = scales[baseNode];
        var bitDepth = ash.getBitDepth();
        var phi = 2f / ((float) (1 << bitDepth) - 1f);
        var d = ash.quantizedDim;
        // var ash = this.ash;

        var baseLandmark = landmarks[baseNode];

        var baseMuProjSum = ash.landmarkProjSum[baseLandmark];
        var baseBitSum = driver.getRawComponentSum(packed, baseNode);

        var mu0 = ash.landmarks[baseLandmark];

        var vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

        float[] mu0DotMuPool = new float[ash.landmarkCount];
        float[] baseDotMuProjPool = new float[ash.landmarkCount];
        for (int c = 0; c < ash.landmarkCount; c++) {
            mu0DotMuPool[c] = vecUtil.dotProduct(mu0, ash.landmarks[c]);

            var muProj = ash.landmarkProj[c];
            assert muProj.length == d;
            // baseDotMuProjPool[c] = calcInnerDot(base, muProj, 0, d);
            baseDotMuProjPool[c] = driver.asymmetricScorePackedInts(packed, baseNode, muProj, 0);
        }

        // TODO add special case for Ash with landmarkCount = 1
        // which has a much simpler equation and should be much faster as well
        return new ScoreFunction.ApproximateScoreFunction() {

            @Override
            public float similarityTo(int otherNode) {

                // var v = compressedVectors[otherNode];
                var scale = scales[otherNode];
                var otherLandmark = landmarks[otherNode];

                var otherMuProjSum = ash.landmarkProjSum[otherLandmark];
                var mu0DotMu = mu0DotMuPool[otherLandmark];
                var baseDotOtherMuProj = baseDotMuProjPool[otherLandmark];

                var otherBitSum = driver.getRawComponentSum(packed, otherNode);
                // var otherDotBaseMuProj = calcInnerDot(v, ash.landmarkProj[v.getLandmark()], 0, d);
                var otherDotBaseMuProj = driver.asymmetricScorePackedInts(packed, otherNode, ash.landmarkProj[otherLandmark], 0);
                // var quantDot = calcSymmetricInnerDot(base, v);
                var quantDot = driver.symmetricScorePackedInts(packed, baseNode, packed, otherNode);

                float dualScaleCoefficient = phi * phi * quantDot - phi * (baseBitSum + otherBitSum) + d;
                float scale0Coefficient = phi * baseDotOtherMuProj - otherMuProjSum;
                float scaleCoefficient = phi * otherDotBaseMuProj - baseMuProjSum;

                return (scale0 * scale) * dualScaleCoefficient + scale0 * scale0Coefficient + scale * scaleCoefficient + mu0DotMu;
            }
        };
    }

    public BuildScoreProvider createBuildScoreProvider(VectorSimilarityFunction vsf) {
        return new BuildScoreProvider() {

			@Override
			public boolean isExact() {
                return false;
			}

			@Override
			public VectorFloat<?> approximateCentroid() {
				// TODO Auto-generated method stub
				throw new UnsupportedOperationException("Unimplemented method 'approximateCentroid'");
			}

			@Override
			public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return new DefaultSearchScoreProvider(scoreFunctionFor(vector, vsf));
			}

			@Override
			public SearchScoreProvider searchProviderFor(int node1) {
                return new DefaultSearchScoreProvider(symmetricScoreFunctionFor(node1, vsf));
			}

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                // TODO diversityProvider may not need so much pre-compute
                // Keep search/diversity consistent for approximate scoring.
                return searchProviderFor(node1);
            }
        };
    }

    // ---------------------------------

    @AllArgsConstructor
    public class AshVector {
        private final int ordinal;

        public void encodeFrom(VectorFloat<?> vector) {
            AshVectors.this.encodeAndSet(ordinal, vector);
        }
    }
}
