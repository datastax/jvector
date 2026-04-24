package io.github.jbellis.jvector.quantization.ash;

public abstract class AbstractAshScoreFunction<V extends AshQuantizedVector> {

    private final AshQueryPrecompute qp;

    public AbstractAshScoreFunction(AshQueryPrecompute qp) {
        this.qp = qp;
    }

    public float similarityTo(V v) {

        final int c = v.getLandmark() & 0xFF; // unsigned [0,C)
        assert c < qp.C : "Invalid landmark id " + c + " for landmarkCount=" + qp.C;

        // scale = ||x − μ_c|| / √d  (precomputed at encode-time)
        final float scale = v.getScale();

        // offset = <x, μ_c> − ||μ_c||²  (precomputed at encode-time)
        final float offset = v.getOffset();

        final float sumTildeQ = qp.sumTildeQ[c];
        final float dotQMu = qp.dotQMu[c];

        int bitDepth = getBitDepth();
        float scaleDifferential = 2f / ((float) (1 << bitDepth) - 1f);

        final var d = qp.d;
        final int tildeBase = c * d;
        var innerDot = calcInnerDot(v, qp.tildeQPool, tildeBase, d);

        // ASH dot-product approximation (multi-landmark, {0,1} encoding):
        //
        //   scale * (2⟨q̃_c, b⟩ − ⟨q̃_c, 1⟩) + ⟨q, μ_c⟩ + (⟨x, μ_c⟩ − ||μ_c||²)
        //
        // where q̃_c = A(q − μ_c) is precomputed once per query and per landmark c.
        return scale * (scaleDifferential * innerDot - sumTildeQ) + dotQMu + offset;
    }

    public abstract float calcInnerDot(V v, float[] queryPool, int queryPoolOffset, int quantizedDim);

    public abstract int getBitDepth();
}
