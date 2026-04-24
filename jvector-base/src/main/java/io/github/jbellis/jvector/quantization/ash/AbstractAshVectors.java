package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public abstract class AbstractAshVectors<T extends AshQuantizedVector> implements CompressedVectors {
    protected final AsymmetricHashing ash;
    protected final T[] compressedVectors;

    // TODO storing both T[] and these values is a waste of memory
    // Cached scalar headers (avoid object chasing in blocked scorer)
    protected final float[] scales;   // scale_i = ||x_i − μ|| / sqrt(d)
    protected final float[] offsets;  // offset_i = <x_i, μ> − ||μ||^2
    protected final byte[] landmarks;

    /**
     * Initialize ASHVectors with an array of ASH-compressed vectors.
     *
     * <p>
     * The array is treated as immutable after construction.
     */
    public AbstractAshVectors(AsymmetricHashing ash, T[] compressedVectors) {
        this.ash = ash;
        this.compressedVectors = compressedVectors;

        int n = compressedVectors.length;
        this.scales = new float[n];
        this.offsets = new float[n];
        this.landmarks = new byte[n];
        for (int i = 0; i < n; i++) {
            var v = compressedVectors[i];
            scales[i]    = v.getScale();
            offsets[i]   = v.getOffset();
            landmarks[i] = v.getLandmark();
        }

    }

    public T get(int node) {
        return compressedVectors[node];
    }

    /**
     * Serialize the compressor followed by the compressed vectors.
     *
     * <p>
     * NOTE:
     * This format is intentionally simple and versioned only at the compressor level.
     * Additional per-vector metadata (e.g., multiple landmarks) must be versioned
     * carefully when introduced.
     */
    @Override
    public void write(IndexWriter out, int version) throws IOException {
        // Write ASH compressor first
        ash.write(out, version);

        // Write vector count
        out.writeInt(compressedVectors.length);

        // Write vectors
        for (var v : compressedVectors) {
            v.write(out, ash.quantizedDim);
        }
    }

    @Override
    public int getOriginalSize() {
        return ash.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return ash.compressedVectorSize();
    }

    @Override
    public AsymmetricHashing getCompressor() {
        return ash;
    }

    /**
     * For ASH, precomputed and non-precomputed scoring are currently identical.
     *
     * <p>
     * TODO:
     *  - Evaluate caching projected queries or other query-dependent state.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
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
    public int count() {
        return compressedVectors.length;
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new IllegalArgumentException("ASH quantizer only supports dot product");
        }

        var qp = AshQueryPrecompute.precomputeQuery(ash, q);
        var sf = createScoreFunction(qp);
        return node -> sf.similarityTo(compressedVectors[node]);
    }

    public ScoreFunction.ApproximateScoreFunction symmetricScoreFunctionFor(int baseNode, VectorSimilarityFunction vsf) {

        var base = compressedVectors[baseNode];
        var scale0 = base.getScale();
        var bitDepth = ash.getBitDepth();
        var phi = 2f / ((float) (1 << bitDepth) - 1f);
        var d = ash.quantizedDim;
        // var ash = this.ash;

        var baseMuProjSum = ash.landmarkProjSum[base.getLandmark()];
        var baseBitSum = base.getRawComponentSum();

        var mu0 = ash.landmarks[base.getLandmark()];

        var vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

        float[] mu0DotMuPool = new float[ash.landmarkCount];
        float[] baseDotMuProjPool = new float[ash.landmarkCount];
        for (int c = 0; c < ash.landmarkCount; c++) {
            mu0DotMuPool[c] = vecUtil.dotProduct(mu0, ash.landmarks[c]);

            var muProj = ash.landmarkProj[c];
            assert muProj.length == d;
            baseDotMuProjPool[c] = calcInnerDot(base, muProj, 0, d);
        }

        // TODO add special case for Ash with landmarkCount = 1
        // which has a much simpler equation and should be much faster as well
        return new ScoreFunction.ApproximateScoreFunction() {

            @Override
            public float similarityTo(int otherNode) {

                var v = compressedVectors[otherNode];
                var scale = v.getScale();

                var otherMuProjSum = ash.landmarkProjSum[v.getLandmark()];
                var mu0DotMu = mu0DotMuPool[v.getLandmark()];
                var baseDotOtherMuProj = baseDotMuProjPool[v.getLandmark()];

                var otherBitSum = v.getRawComponentSum();
                var otherDotBaseMuProj = calcInnerDot(v, ash.landmarkProj[v.getLandmark()], 0, d);
                var quantDot = calcSymmetricInnerDot(base, v);

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

    @Override
    public abstract long ramBytesUsed();

    public abstract AbstractAshScoreFunction<T> createScoreFunction(AshQueryPrecompute qp);

    public abstract float calcInnerDot(T v, float[] queryPool, int queryPoolOffset, int quantizedDim);

    public abstract float calcSymmetricInnerDot(T v1, T v2);
}
