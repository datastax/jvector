package io.github.jbellis.jvector.quantization.ash;

import io.github.jbellis.jvector.quantization.AsymmetricHashing.StiefelTransform;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

public class NbAshProjector {
    private static final VectorUtilSupport vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();
    // private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    // ThreadLocal to hold the reusable buffers.
    // Using a simple container class to avoid multiple ThreadLocal lookups.
    private static final ThreadLocal<Workspace> WORKSPACE = ThreadLocal.withInitial(Workspace::new);

    private static final class Workspace {
        float[] x;
        float[] muArr;
        float[] xhat;

        void ensureCapacity(int dim) {
            if (x == null || x.length < dim) {
                x = new float[dim];
                muArr = new float[dim];
                xhat = new float[dim];
            }
        }
    }

    public static void projectTo(VectorFloat<?> vector, float residualNorm, StiefelTransform stiefel, VectorFloat<?> mu, float[] out) {
        final float[][] A = stiefel.AFloat;
        final int originalDim = stiefel.cols;
        final int quantizedDim = stiefel.rows;

        // Get workspace for this thread and ensure correct sizing (originalDim may have changed)
        Workspace ws = WORKSPACE.get();
        ws.ensureCapacity(originalDim);

        // Alias for readability (no new allocations)
        final float[] x = ws.x;
        final float[] muArr = ws.muArr;
        final float[] xhat = ws.xhat;

        // Copy vector and mean once into workspace
        for (int d = 0; d < originalDim; d++) {
            x[d] = vector.get(d);
            muArr[d] = mu.get(d);
        }

        // ASH paper, Eq. 6, normalization
        final float invNorm = (residualNorm > 0f) ? (1.0f / residualNorm) : 0.0f;

        // Compute normalized residual x̂
        for (int d = 0; d < originalDim; d++) {
            xhat[d] = (x[d] - muArr[d]) * invNorm;
        }

        for (int i = 0; i < quantizedDim; i++) {
            float acc = vecUtil.ashDotRow(A[i], xhat);
            out[i] = acc;
        }
    }

    public static float thinNormFromQuantProjection(int bitDepth, int[] quantized) {
        var normSq = Arrays.stream(quantized)
            .mapToDouble(q -> (q * 2. / ((float) (1 << bitDepth) - 1.)) - 1.)
            .map(x -> x * x)
            .sum();
        return (float) Math.sqrt(normSq);
    }

    public static int[] quantizeProjection(int bitDepth, float[] proj) {
        final int dim = proj.length;

        // Step 1: Extract signs and compute absolute values
        boolean[] vecSigns = new boolean[dim];
        float[] posVec = new float[dim];
        for (int i = 0; i < dim; i++) {
            vecSigns[i] = proj[i] >= 0.0f;
            posVec[i] = Math.abs(proj[i]);
        }

        // Step 2: Find best scale factor
        int exBits = bitDepth - 1;
        float scaleFactor = bestScaleFactor(exBits, posVec);
        
        // Step 3: Quantize positive values with scale factor
        int[] quant = quantizePosScaled(exBits, posVec, scaleFactor);
        
        // Step 4: Apply sign encoding
        int posMid = 1 << exBits;  // 2^(bitDepth-1)
        int posMax = (1 << exBits) - 1;  // 2^(bitDepth - 1) - 1

        int[] result = new int[dim];
        for (int i = 0; i < dim; i++) {
            int q = quant[i];
            assert q < posMid : "expected q(" + q + ") < posMid(" + posMid + ")";
            
            if (vecSigns[i]) {
                // Positive: add "1" to MSB to indicate +ve range
                result[i] = posMid + q;
            } else {
                // Negative: flip the codes for -ve range
                // Keeps "0" at MSB to indicate -ve range
                result[i] = posMax - q;
            }
        }
        
        return result;
    }

    private static int[] quantizePosScaled(int exBits, float[] vec, float scaleFactor) {
        int hi = (1 << exBits) - 1;  // !((!0) << exBits) in Rust
        int[] result = new int[vec.length];
        
        for (int i = 0; i < vec.length; i++) {
            float scaled = vec[i] * scaleFactor;
            float floored = (float) Math.floor(scaled);
            float clamped = Math.max(0.0f, Math.min(floored, (float) hi));
            result[i] = (int) clamped;
        }
        
        return result;
    }

    private static float bestScaleFactor(int exBits, float[] posVec) {
        final float EPS = 1e-6f;
        final int dim = posVec.length;
        final int qMax = (1 << exBits) - 1;
        
        // Initialize quantization array
        int[] oBar = quantizePosScaled(exBits, posVec, 0.0f);
        
        // Priority queue for updates (min-heap by scale_factor)
        PriorityQueue<Update> next = new PriorityQueue<>(Comparator.comparingDouble(u -> u.scaleFactor));
        
        // Populate initial updates
        for (int i = 0; i < dim; i++) {
            if (oBar[i] < qMax && posVec[i] > EPS) {
                next.add(new Update((oBar[i] + 1.0f) / posVec[i], i));
            }
        }
        
        // Initialize accumulators
        float num = 0.0f;
        float denSq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float qs = oBar[i] + 0.5f;
            num += qs * posVec[i];
            denSq += qs * qs;
        }
        
        float ipMax = num / (float) Math.sqrt(denSq);
        float bestScale = 0.0f;
        
        // Process updates
        while (!next.isEmpty()) {
            Update update = next.poll();
            float scaleFactor = update.scaleFactor;
            int updateIndex = update.updateId;
            
            float x = posVec[updateIndex];
            int qPrev = oBar[updateIndex];
            oBar[updateIndex]++;
            int q = oBar[updateIndex];
            
            // Update accumulators
            num += x;
            denSq += 2.0f * (qPrev + 0.5f) + 1.0f;
            
            float ip = num / (float) Math.sqrt(denSq);
            if (ip > ipMax) {
                ipMax = ip;
                bestScale = scaleFactor;
            }
            
            // Add next update if not at max
            if (q < qMax) {
                float newScale = (q + 1.0f) / x;
                next.add(new Update(newScale, updateIndex));
            }
        }
        
        return bestScale + EPS;
    }

    private static class Update {
        final float scaleFactor;
        final int updateId;
        
        Update(float scaleFactor, int updateId) {
            this.scaleFactor = scaleFactor;
            this.updateId = updateId;
        }
    }

    private NbAshProjector() {
        throw new UnsupportedOperationException("Utility class should not be instantiated");
    }
}
