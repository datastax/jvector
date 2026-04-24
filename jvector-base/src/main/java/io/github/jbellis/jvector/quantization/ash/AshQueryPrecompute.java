package io.github.jbellis.jvector.quantization.ash;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class AshQueryPrecompute {
    final int d;
    final int C;
    final float[] tildeQPool;          // [C * d]
    final float[] sumTildeQ;            // [C]
    final float[] dotQMu;               // [C]

    private AshQueryPrecompute(int d, int C,
                    float[] tildeQPool,
                    float[] sumTildeQ,
                    float[] dotQMu) {
        this.d = d;
        this.C = C;
        this.tildeQPool = tildeQPool;
        this.sumTildeQ = sumTildeQ;
        this.dotQMu = dotQMu;
    }

    public static AshQueryPrecompute precomputeQuery(AsymmetricHashing ash, VectorFloat<?> query) {
        final int d = ash.quantizedDim;
        final int D = ash.originalDimension;
        final int C = ash.landmarkCount;

        final float[][] A = ash.stiefelTransform.AFloat;

        final var vecUtil =
                VectorizationProvider.getInstance().getVectorUtilSupport();

        // q materialized once
        final float[] qArr = new float[D];
        for (int i = 0; i < D; i++) {
            qArr[i] = query.get(i);
        }

        // Aq
        final float[] qProj = new float[d];
        float sumQProj = 0f;
        for (int j = 0; j < d; j++) {
            float v = vecUtil.ashDotRow(A[j], qArr);
            qProj[j] = v;
            sumQProj += v;
        }

        final float[] tildeQPool = new float[C * d];
        final float[] sumTildeQ = new float[C];
        final float[] dotQMu = new float[C];

        for (int c = 0; c < C; c++) {
            dotQMu[c] = VectorUtil.dotProduct(query, ash.landmarks[c]);
            sumTildeQ[c] = sumQProj - ash.landmarkProjSum[c];

            final float[] muProj = ash.landmarkProj[c];
            final int base = c * d;
            for (int j = 0; j < d; j++) {
                tildeQPool[base + j] = qProj[j] - muProj[j];
            }
        }

        return new AshQueryPrecompute(d, C, tildeQPool, sumTildeQ, dotQMu);
    }
}
