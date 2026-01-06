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
    private final byte[] landmarks;

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
        this.landmarks = new byte[n];
        for (int i = 0; i < n; i++) {
            var v = compressedVectors[i];
            scales[i]    = v.scale;
            offsets[i]   = v.offset;
            landmarks[i] = v.landmark;
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

        // Compute query precompute ONCE (shared by scalar + simd)
        final QueryPrecompute qp = precomputeQuery(query);

        // ------------------------------------------------------------
        // Forced scalar path (portable register-accumulator)
        // ------------------------------------------------------------
        if (kernel == AshBlockKernel.SCALAR) {
            return new BlockScalarASHBlockScorer(
                    compressedVectors,
                    scales,
                    offsets,
                    landmarks,
                    qp
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
                compressedVectors,
                scales,
                offsets,
                landmarks,
                packedBits,
                blockSize,
                qp
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

        private final AsymmetricHashing.QuantizedVector[] vectors;
        private final float[] scales;
        private final float[] offsets;
        private final byte[] landmarks;

        private final float[] tildeQPool;   // [C * d]
        private final float[] sumTildeQ;    // [C]
        private final float[] dotQMu;       // [C]
        private final int d;
        private final int words;

        BlockScalarASHBlockScorer(
                AsymmetricHashing.QuantizedVector[] vectors,
                float[] scales,
                float[] offsets,
                byte[] landmarks,
                QueryPrecompute qp)
        {

            this.vectors = vectors;
            this.scales = scales;
            this.offsets = offsets;
            this.landmarks = landmarks;

            this.d = qp.d;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            this.tildeQPool = qp.tildeQPool;
            this.sumTildeQ = qp.sumTildeQ;
            this.dotQMu = qp.dotQMu;
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > vectors.length) {
                throw new IllegalArgumentException("Range out of bounds: [" + start + ", " + (start + count) + ")");
            }
            if (out.length < count) throw new IllegalArgumentException("out.length < count");
            if (count == 0) return;

            for (int i = 0; i < count; i++) {
                final int idx = start + i;

                final int c = landmarks[idx] & 0xFF; // unsigned [0,C)
                final int base = c * d;

                float maskedAdd = 0f;

                // bit-walk against landmark-specific q̃
                final long[] bits = vectors[idx].binaryVector;
                int baseDim = 0;
                for (int w = 0; w < words && baseDim < d; w++, baseDim += 64) {
                    long word = bits[w];
                    while (word != 0L) {
                        int bit = Long.numberOfTrailingZeros(word);
                        int j = baseDim + bit;
                        if (j < d) maskedAdd += tildeQPool[base + j];
                        word &= (word - 1);
                    }
                }

                out[i] =
                        scales[idx] * (2f * maskedAdd - sumTildeQ[c])
                                + dotQMu[c]
                                + offsets[idx];
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

        private final AsymmetricHashing.QuantizedVector[] vectors;
        private final float[] scales;
        private final float[] offsets;
        private final byte[] landmarks;

        private final long[] packedBits;
        private final int blockSize;

        private final int d;
        private final int words;

        private final float[] maskedAdds;     // [blockSize]

        private final float[] tildeQPool;     // qp.tildeQPool [C*d]
        private final float[] tildeQScratch;  // [d]
        private final float[] sumTildeQ;      // qp.sumTildeQ [C]
        private final float[] dotQMu;         // qp.dotQMu [C]

        private final VectorUtilSupport vecUtil;

        private static final java.util.concurrent.atomic.AtomicBoolean PRINTED_MIXED_ONCE =
                new java.util.concurrent.atomic.AtomicBoolean(false);

        BlockSimdASHBlockScorer(
                AsymmetricHashing.QuantizedVector[] vectors,
                float[] scales,
                float[] offsets,
                byte[] landmarks,
                long[] packedBits,
                int blockSize,
                QueryPrecompute qp) {

            this.vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

            this.vectors = vectors;
            this.scales = scales;
            this.offsets = offsets;
            this.landmarks = landmarks;

            this.packedBits = packedBits;
            this.blockSize = blockSize;

            this.d = qp.d;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            this.maskedAdds = new float[blockSize];

            this.tildeQPool = qp.tildeQPool;
            this.tildeQScratch = new float[d];
            this.sumTildeQ = qp.sumTildeQ;
            this.dotQMu = qp.dotQMu;
        }

        // SIMD kernel is invoked on homogeneous landmark runs; this method splits mixed ranges into runs.
        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > vectors.length) {
                throw new IllegalArgumentException("Range out of bounds: [" + start + ", " + (start + count) + ")");
            }
            if (out.length < count) throw new IllegalArgumentException("out.length < count");
            if (count == 0) return;

            int ord = start;
            final int end = start + count;

            while (ord < end) {
                // Identify landmark for this run
                final int c = landmarks[ord] & 0xFF;
                final int base = c * d;

                // Compute packed-block boundary info
                final int blockId = ord / blockSize;
                final int blockBase = blockId * blockSize;
                final int laneStart = ord - blockBase;

                // Max we can do without crossing packed-block boundary or request end
                final int maxLen = Math.min(blockSize - laneStart, end - ord);

                // Clamp further so we do NOT cross a landmark boundary inside this packed block
                int runLen = 1;
                while (runLen < maxLen && ((landmarks[ord + runLen] & 0xFF) == c)) {
                    runLen++;
                }

                // Catch mixed-landmark calls for debug.  TODO remove later
                if (runLen != maxLen && PRINTED_MIXED_ONCE.compareAndSet(false, true)) {
                    int nextLm = (ord + runLen < end) ? (landmarks[ord + runLen] & 0xFF) : -1;
                    System.out.println(
                            "\t[ASH SIMD debug] mixed-landmark boundary inside packed-block slice: " +
                                    "start=" + start +
                                    " count=" + count +
                                    " ord=" + ord +
                                    " blockId=" + blockId +
                                    " laneStart=" + laneStart +
                                    " runLen=" + runLen +
                                    " maxLen=" + maxLen +
                                    " c=" + c +
                                    " nextC=" + nextLm
                    );
                }

                assert runLen >= 1 && runLen <= maxLen;
                assert laneStart >= 0 && laneStart + runLen <= blockSize;

                final float runSumTildeQ = sumTildeQ[c];
                final float runDotQMu = dotQMu[c];
                final int blockWordBase = blockId * words * blockSize;

                // Load q̃_c into contiguous scratch for the unpooled kernel
                System.arraycopy(tildeQPool, base, tildeQScratch, 0, d);

                vecUtil.ashMaskedAddBlockAllWords(
                        tildeQScratch,
                        d,
                        packedBits,
                        blockWordBase,
                        words,
                        blockSize,
                        laneStart,
                        runLen,
                        maskedAdds
                );


                for (int lane = 0; lane < runLen; lane++) {
                    int idx = ord + lane;
                    float m = maskedAdds[lane];
                    out[idx - start] =
                            scales[idx] * (2f * m - runSumTildeQ)
                                    + runDotQMu
                                    + offsets[idx];
                }

                ord += runLen;
            }
        }
    }

    private QueryPrecompute precomputeQuery(VectorFloat<?> query) {
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

        return new QueryPrecompute(d, C, tildeQPool, sumTildeQ, dotQMu);
    }

    static final class QueryPrecompute {
        final int d;
        final int C;
        final float[] tildeQPool;          // [C * d]
        final float[] sumTildeQ;            // [C]
        final float[] dotQMu;               // [C]

        QueryPrecompute(int d, int C,
                        float[] tildeQPool,
                        float[] sumTildeQ,
                        float[] dotQMu) {
            this.d = d;
            this.C = C;
            this.tildeQPool = tildeQPool;
            this.sumTildeQ = sumTildeQ;
            this.dotQMu = dotQMu;
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
