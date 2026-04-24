package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.quantization.AsymmetricHashing.StiefelTransform;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import lombok.Getter;

public abstract class AshQuantizedVector {
    @Getter
    private float scale = Float.NaN;

    @Getter
    private float offset = Float.NaN;

    @Getter
    private byte landmark = 0;

    protected AshQuantizedVector(float scale, float offset, byte landmark) {
        this.scale = scale;
        this.offset = offset;
        this.landmark = landmark;
    }

    public void encodeInto(
            VectorFloat<?> vector,
            VectorFloat<?> mu,
            byte landmarkId,
            float landmarkNormSq,
            StiefelTransform stiefel)
    {
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
        final float scale = residualNorm * invSqrtD;
        final float offset = dotXMu - landmarkNormSq;

        // Write the new header fields into dest (requires QuantizedVector fields renamed, see below)
        this.scale = scale;
        this.offset = offset;
        this.landmark = landmarkId;

        // Binary body: sign(A · (x − μ)) with Eq. 6 normalization inside quantizer
        this.quantizeVectorInto(vector, residualNorm, stiefel, mu);
    }

    /**
     * Writes the binary body into {@code dest.binaryVector}.
     * Also <em>updates</em> the existing scale to make it multi-bit compatible
     * (which is a no-op for the one-bit case).
     *
     * <p>Header fields ({@code scale}, {@code offset}, {@code landmark}) must be
     * set by the caller prior to calling this method.</p>
     *
     * @param residualNorm  ||x - μ||_2, required for Eq. 6 normalization during binarization
     */
    // public abstract void quantizeVectorInto(
    //         VectorFloat<?> vector,
    //         float residualNorm,
    //         StiefelTransform stiefel,
    //         VectorFloat<?> mu);
    
    public void quantizeVectorInto(VectorFloat<?> vector, float residualNorm, StiefelTransform stiefel,
            VectorFloat<?> mu) {
        var out = new float[stiefel.quantizedDim()];
        NbAshProjector.projectTo(vector, residualNorm, stiefel, mu, out);

        int[] quant = NbAshProjector.quantizeProjection(getBitDepth(), out);

        var thinNorm = NbAshProjector.thinNormFromQuantProjection(getBitDepth(), quant);
        var dSqrt = Math.sqrt(quant.length);
        var scaleScale = dSqrt / thinNorm;
        this.scale *= scaleScale;

        binarizeIntoFromQuantized(quant);
    }
    
    // TODO really need to pass in quantizedDim?
    public void write(IndexWriter out, int quantizedDim) throws IOException {
        out.writeFloat(scale);
        out.writeFloat(offset);
        out.writeByte(landmark);

        writeBinaryVector(out, quantizedDim);
    }

    public abstract void writeBinaryVector(IndexWriter out, int quantizedDim) throws IOException;

    public abstract int getBitDepth();

    public abstract void binarizeIntoFromQuantized(int[] quantized);

    /** Dot product of unsigned dims with <pre>\vec{1}</pre> */
    public abstract int getRawComponentSum();

    public int[] toRawComponents(int quantizedDim) {
        throw new UnsupportedOperationException("unimplemented optional method");
    }
}
