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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Container for ASH-compressed vectors with a minimal scoring interface.
 *
 * <p>
 * This class implements the functionality required for:
 * <ul>
 *   <li>Scalar scoring</li>
 *   <li>Block scoring</li>
 *   <li>Vectorized versions of scoring</li>
 * </ul>
 *
 * <p>
 * TODO (future work):
 * <ul>
 *   <li>Add a full ASH ScoreFunction aligned with graph search semantics</li>
 *   <li>Support similarityToNeighbor(...) where appropriate</li>
 *   <li>Integrate ASH as a Feature for on-disk graph indexes</li>
 *   <li>Define reranking strategy</li>
 *   <li>Support multi-landmark ASH </li>
 * </ul>
 */
public class ASHVectors implements CompressedVectors {

    final AsymmetricHashing ash;
    final AsymmetricHashing.QuantizedVector[] compressedVectors;
    final ASHScorer scorer;

    enum AshBlockKernel {
        AUTO,
        SCALAR,
        SIMD;

        static AshBlockKernel fromProperty() {
            String v = System.getProperty("jvector.ash.blockKernel", "auto").toLowerCase();
            if ("scalar".equals(v)) return SCALAR;
            if ("simd".equals(v))   return SIMD;
            return AUTO;
        }
    }

    // Cached scalar headers (avoid object chasing in blocked scorer)
    private final float[] scales;   // scale_i = ||x_i − μ|| / sqrt(d)
    private final float[] offsets;  // offset_i = <x_i, μ> − ||μ||^2

    // Quantized dimensions
    private final int d;
    private final int words;

    // Packed block-column-major bits (built lazily per blockSize)
    private int packedBlockSize = -1;
    private long[] packedBits = null;
    private int packedBlockCount = 0;

    /**
     * Initialize ASHVectors with an array of ASH-compressed vectors.
     *
     * <p>
     * The array is treated as immutable after construction.
     */
    public ASHVectors(AsymmetricHashing ash,
                      AsymmetricHashing.QuantizedVector[] compressedVectors) {
        this.ash = ash;
        this.compressedVectors = compressedVectors;
        this.scorer = new ASHScorer(ash);

        this.d = ash.quantizedDim;
        this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

        int n = compressedVectors.length;
        this.scales = new float[n];
        this.offsets = new float[n];
        for (int i = 0; i < n; i++) {
            var v = compressedVectors[i];
            scales[i] = v.scale;
            offsets[i] = v.offset;
        }
    }

    @Override
    public int count() {
        return compressedVectors.length;
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
    public void write(DataOutput out, int version) throws IOException {
        // Write ASH compressor first
        ash.write(out, version);

        // Write vector count
        out.writeInt(compressedVectors.length);

        // Write vectors
        for (var v : compressedVectors) {
            v.write(out, ash.quantizedDim);
        }
    }

    public static ASHVectors load(RandomAccessReader in) throws IOException {
        var ash = AsymmetricHashing.load(in);

        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }

        var compressedVectors = new AsymmetricHashing.QuantizedVector[size];
        for (int i = 0; i < size; i++) {
            compressedVectors[i] =
                    AsymmetricHashing.QuantizedVector.load(in, ash.quantizedDim);
        }

        return new ASHVectors(ash, compressedVectors);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {

        // Scorer follows ASH paper (single landmark, DOT_PRODUCT only)
        final ASHScorer.ASHScoreFunction f = scorer.scoreFunctionFor(query, similarityFunction);

        // Wrap QuantizedVector scorer into ordinal-based score function
        return node -> f.similarityTo(compressedVectors[node]);
    }

// ============================================================================
// ASH block-oriented scoring
// ============================================================================

    /**
     * Scalar reference block scorer.
     *
     * <p>
     * This scorer computes ASH scores by invoking the scalar per-vector
     * score function for each vector in the requested range.
     * </p>
     *
     * <p>
     * It serves as a correctness reference and baseline implementation.
     * No block-level optimizations are applied.
     * </p>
     */
    private static final class ScalarASHBlockScorer implements ASHBlockScorer {

        private final ASHScorer.ASHScoreFunction scorer;
        private final AsymmetricHashing.QuantizedVector[] vectors;

        ScalarASHBlockScorer(
                ASHScorer.ASHScoreFunction scorer,
                AsymmetricHashing.QuantizedVector[] vectors) {
            this.scorer = scorer;
            this.vectors = vectors;
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > vectors.length) {
                throw new IllegalArgumentException(
                        "Range out of bounds: [" + start + ", " + (start + count) + ")"
                );
            }
            if (out.length < count) {
                throw new IllegalArgumentException("out.length < count");
            }
            if (count == 0) return;

            final int end = start + count;
            for (int i = start, o = 0; i < end; i++, o++) {
                out[o] = scorer.similarityTo(vectors[i]);
            }
        }
    }

    /**
     * Returns a scalar reference block scorer.
     *
     * <p>
     * The returned scorer matches {@link #scoreFunctionFor} exactly
     * and computes each score independently.
     * </p>
     */
    public ASHBlockScorer blockScorerFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException(
                    "ASH block scorer supports DOT_PRODUCT only");
        }

        ASHScorer.ASHScoreFunction f =
                scorer.scoreFunctionFor(query, similarityFunction);

        return new ScalarASHBlockScorer(f, compressedVectors);
    }

    /**
     * Returns a block-oriented ASH scorer for contiguous vectors.
     *
     * <p>
     * This factory selects between a portable scalar implementation and a
     * SIMD implementation of ASH block scoring (ASH paper, Figure 1 / Table 1).
     * </p>
     *
     * <p>
     * Selection is controlled by the JVM property:
     * </p>
     *
     * <pre>
     *   -Djvector.ash.blockKernel=auto|scalar|simd
     * </pre>
     *
     * <ul>
     *   <li><b>auto</b> (default): the active VectorUtilSupport backend determines whether execution is vectorized or scalar</li>
     *   <li><b>scalar</b>: force the portable scalar register-accumulating implementation</li>
     *   <li><b>simd</b>: simd: force the block scorer defined in the ASH paper; execution may still be scalar if the active backend does not support masked-load SIMD</li>
     * </ul>
     *
     * <p>
     * In all cases, accumulation of the masked-add term ⟨q̃, b⟩ is performed in
     * local scalar registers, ensuring numerical equivalence across backends.
     * </p>
     */
    public ASHBlockScorer blockScorerFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction,
            int blockSize) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException("DOT_PRODUCT only");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be > 0");
        }

        final AshBlockKernel kernel = AshBlockKernel.fromProperty();

        // ------------------------------------------------------------
        // Forced scalar path (portable register-accumulator)
        // ------------------------------------------------------------
        if (kernel == AshBlockKernel.SCALAR) {
            return new BlockScalarASHBlockScorer(
                    ash,
                    compressedVectors,
                    scales,
                    offsets,
                    query
            );
        }

        // ------------------------------------------------------------
        // SIMD or AUTO path (paper-faithful block kernel)
        // ------------------------------------------------------------
        // AUTO and SIMD intentionally share the same implementation.
        // VectorUtilSupport decides whether masked-load is vectorized
        // or falls back to scalar internally.
        ensurePackedBits(blockSize);

        return new BlockSimdASHBlockScorer(
                ash,
                compressedVectors,
                scales,
                offsets,
                packedBits,
                blockSize,
                query
        );
    }

    private void ensurePackedBits(int blockSize) {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be > 0");
        }
        if (packedBits != null && packedBlockSize == blockSize) {
            return;
        }

        final int n = compressedVectors.length;
        packedBlockSize = blockSize;
        packedBlockCount = (n + blockSize - 1) / blockSize;
        packedBits = new long[packedBlockCount * words * blockSize];

        for (int b = 0; b < packedBlockCount; b++) {
            int baseOrd = b * blockSize;
            int blockLen = Math.min(blockSize, n - baseOrd);

            for (int w = 0; w < words; w++) {
                int dstBase = (b * words + w) * blockSize;
                for (int lane = 0; lane < blockLen; lane++) {
                    packedBits[dstBase + lane] =
                            compressedVectors[baseOrd + lane].binaryVector[w];
                }
                // remaining lanes default to 0
            }
        }
    }


    /**
     * Block-oriented ASH scorer using register-local accumulation.
     *
     * <p>
     * This is the preferred block implementation. It avoids temporary
     * per-block arrays and accumulates the masked-add term directly
     * into scalar registers.
     * </p>
     */
    private static final class BlockScalarASHBlockScorer implements ASHBlockScorer {

        private final AsymmetricHashing ash;
        private final AsymmetricHashing.QuantizedVector[] vectors;
        private final float[] scales;
        private final float[] offsets;


        private final float[] tildeQ;
        private final float sumTildeQ;
        private final float dotQMu;

        private final int d;
        private final int words;

        BlockScalarASHBlockScorer(
                AsymmetricHashing ash,
                AsymmetricHashing.QuantizedVector[] vectors,
                float[] scales,
                float[] offsets,
                VectorFloat<?> query) {

            this.ash = ash;
            this.vectors = vectors;
            this.scales = scales;
            this.offsets = offsets;

            this.d = ash.quantizedDim;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            final int D = ash.originalDimension;
            final VectorFloat<?> mu = ash.landmarks[0];

            final float[] qArr = new float[D];
            final float[] muArr = new float[D];
            for (int i = 0; i < D; i++) {
                qArr[i] = query.get(i);
                muArr[i] = mu.get(i);
            }

            this.tildeQ = new float[d];
            final float[][] A = ash.stiefelTransform.AFloat;
            for (int r = 0; r < d; r++) {
                float s = 0f;
                float[] Arow = A[r];
                for (int j = 0; j < D; j++) {
                    s += Arow[j] * (qArr[j] - muArr[j]);
                }
                tildeQ[r] = s;
            }

            float sum = 0f;
            for (int i = 0; i < d; i++) sum += tildeQ[i];
            this.sumTildeQ = sum;
            this.dotQMu = VectorUtil.dotProduct(query, mu);
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {

            for (int i = 0; i < count; i++) {
                int idx = start + i;

                float maskedAdd = 0f;
                long[] bits = vectors[idx].binaryVector;

                for (int w = 0; w < words; w++) {
                    long word = bits[w];
                    int baseDim = w * 64;

                    while (word != 0L) {
                        int bit = Long.numberOfTrailingZeros(word);
                        int dim = baseDim + bit;
                        if (dim < d) maskedAdd += tildeQ[dim];
                        word &= (word - 1);
                    }
                }

                float scale = scales[idx];
                float offset = offsets[idx];

                out[i] =
                        scale * (2f * maskedAdd - sumTildeQ)
                                + dotQMu
                                + offset;
            }
        }
    }

    /**
     * Block-oriented ASH scorer using SIMD masked-loads with register-local
     * accumulation.
     *
     * <p>
     * This scorer is a faithful implementation of Figure 1
     * and Table 1 kernel from the ASH paper
     * </p>
     *
     * <ul>
     *   <li>binary codes are stored in block column-major layout</li>
     *   <li>SIMD lanes correspond to dimensions</li>
     *   <li>masked loads select active dimensions</li>
     *   <li>horizontal sums are accumulated per vector</li>
     * </ul>
     *
     * <p>
     * This implementation accumulates masked-add results directly into scalar registers,
     * avoiding per-lane memory traffic and restoring performance.
     * </p>
     */
    private static final class BlockSimdASHBlockScorer implements ASHBlockScorer {

        private final AsymmetricHashing ash;
        private final AsymmetricHashing.QuantizedVector[] vectors;

        private final float[] scales;
        private final float[] offsets;

        private final long[] packedBits;
        private final int blockSize;

        private final int d;
        private final int words;

        // Query precompute
        private final float[] tildeQ;
        private final float sumTildeQ;
        private final float dotQMu;

        // Scratch (one per scorer instance)
        private final float[] maskedAdds;

        private final VectorUtilSupport vecUtil;

        BlockSimdASHBlockScorer(
                AsymmetricHashing ash,
                AsymmetricHashing.QuantizedVector[] vectors,
                float[] scales,
                float[] offsets,
                long[] packedBits,
                int blockSize,
                VectorFloat<?> query) {

            this.vecUtil =
                    io.github.jbellis.jvector.vector.VectorizationProvider
                            .getInstance()
                            .getVectorUtilSupport();
            this.ash = ash;
            this.vectors = vectors;
            this.scales = scales;
            this.offsets = offsets;
            this.packedBits = packedBits;
            this.blockSize = blockSize;

            this.d = ash.quantizedDim;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            this.maskedAdds = new float[blockSize];

            // ---- Query precompute (identical to ASHScorer) ----
            final int D = ash.originalDimension;
            final VectorFloat<?> mu = ash.landmarks[0];

            final float[] qArr = new float[D];
            final float[] muArr = new float[D];
            for (int i = 0; i < D; i++) {
                qArr[i] = query.get(i);
                muArr[i] = mu.get(i);
            }

            this.tildeQ = new float[d];
            final float[][] A = ash.stiefelTransform.AFloat;
            for (int r = 0; r < d; r++) {
                float s = 0f;
                float[] Arow = A[r];
                for (int j = 0; j < D; j++) {
                    s += Arow[j] * (qArr[j] - muArr[j]);
                }
                tildeQ[r] = s;
            }

            float sum = 0f;
            for (int i = 0; i < d; i++) sum += tildeQ[i];
            this.sumTildeQ = sum;
            this.dotQMu = VectorUtil.dotProduct(query, mu);
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count == 0) return;

            int ord = start;
            int end = start + count;

            while (ord < end) {
                final int blockId = ord / blockSize;
                final int blockBase = blockId * blockSize;
                final int laneStart = ord - blockBase;
                final int blockLen = Math.min(blockSize - laneStart, end - ord);

                final int blockWordBase = blockId * words * blockSize;

                // SIMD masked-add with register accumulation
                vecUtil.ashMaskedAddBlockAllWords(
                        tildeQ,
                        d,
                        packedBits,
                        blockWordBase,
                        words,
                        blockSize,
                        laneStart,
                        blockLen,
                        maskedAdds
                );

                // Finish Eq. 11 per vector
                for (int lane = 0; lane < blockLen; lane++) {
                    int idx = ord + lane;
                    float scale = scales[idx];
                    float offset = offsets[idx];
                    float m = maskedAdds[lane];

                    out[idx - start] =
                            scale * (2f * m - sumTildeQ)
                                    + dotQMu
                                    + offset;
                }

                ord += blockLen;
            }
        }
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

    public AsymmetricHashing.QuantizedVector get(int ordinal) {
        return compressedVectors[ordinal];
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

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long vectorsArraySize =
                AH_BYTES + (long) REF_BYTES * compressedVectors.length;
        long vectorsDataSize =
                (long) ash.compressedVectorSize() * compressedVectors.length;

        return ash.ramBytesUsed() + vectorsArraySize + vectorsDataSize;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ASHVectors that = (ASHVectors) o;
        return Objects.equals(ash, that.ash)
                && Arrays.equals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(ash);
        result = 31 * result + Arrays.hashCode(compressedVectors);
        return result;
    }

    @Override
    public String toString() {
        return "ASHVectors{count=" + compressedVectors.length +
                ", ash=" + ash + '}';
    }
}
