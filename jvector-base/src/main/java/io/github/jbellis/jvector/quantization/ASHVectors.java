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

    // Cached scalar headers (avoid object chasing in blocked scorer)
    private final float[] residualNorms;
    private final float[] dotXMu;

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
        this.residualNorms = new float[n];
        this.dotXMu = new float[n];
        for (int i = 0; i < n; i++) {
            var v = compressedVectors[i];
            residualNorms[i] = v.residualNorm;
            dotXMu[i] = v.dotWithLandmark;
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
     * Block-structured scalar ASH scorer.
     *
     * <p>
     * This scorer reorganizes the computation to operate on contiguous
     * blocks of vectors while remaining fully scalar.
     * </p>
     *
     * <p>
     * The loop structure is:
     * </p>
     * <ul>
     *   <li>outer loop over blocks of vectors</li>
     *   <li>outer loop over binary words</li>
     *   <li>inner loop over vectors within the block</li>
     * </ul>
     *
     * <p>
     * This improves cache locality and provides a stable structural
     * foundation for block-oriented evaluation.
     * </p>
     */
    public ASHBlockScorer blockScorerFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction,
            int blockSize) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException(
                    "ASH block scorer supports DOT_PRODUCT only");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be > 0");
        }

        return new BlockedScalarASHBlockScorer(ash, compressedVectors, query, blockSize);
    }

    /**
     * Block-oriented ASH scorer using register-local accumulation.
     *
     * <p>
     * This implementation evaluates ASH scores for a contiguous block
     * of vectors while accumulating intermediate values directly into
     * local scalar variables.
     * </p>
     *
     * <p>
     * Compared to array-based accumulation, this approach:
     * </p>
     * <ul>
     *   <li>reduces memory traffic</li>
     *   <li>keeps hot values in registers</li>
     *   <li>preserves exact ASH scoring semantics</li>
     * </ul>
     *
     * <p>
     * This scorer is suitable for brute-force scans and graph neighborhood
     * evaluation where vectors are processed contiguously.
     * </p>
     */
    public ASHBlockScorer blockScorerRegisterAccFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction,
            int blockSize) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException("DOT_PRODUCT only");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be > 0");
        }

        return new RegisterAccASHBlockScorer(
                ash,
                compressedVectors,
                residualNorms,
                dotXMu,
                query
        );
    }

    /**
     * Block-structured scalar ASH scorer.
     *
     * <p>
     * This scorer performs scalar bit-walk accumulation for each vector,
     * but organizes the computation by blocks and binary words.
     * </p>
     *
     * <p>
     * It preserves exact correctness and serves as a clear baseline
     * for block-based implementations.
     * </p>
     */
    private static final class BlockedScalarASHBlockScorer implements ASHBlockScorer {

        private final AsymmetricHashing ash;
        private final AsymmetricHashing.QuantizedVector[] vectors;
        private final int blockSize;

        private final float[] tildeQ;
        private final float sumTildeQ;
        private final float dotQMu;
        private final float muNormSq;
        private final float invSqrtD;

        private final int d;
        private final int words;
        private final int originalDim;

        BlockedScalarASHBlockScorer(
                AsymmetricHashing ash,
                AsymmetricHashing.QuantizedVector[] vectors,
                VectorFloat<?> query,
                int blockSize) {

            this.ash = ash;
            this.vectors = vectors;
            this.blockSize = blockSize;

            this.d = ash.quantizedDim;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);
            this.originalDim = ash.originalDimension;

            final VectorFloat<?> mu = ash.globalMean;

            final float[] qArr = new float[originalDim];
            final float[] muArr = new float[originalDim];
            for (int i = 0; i < originalDim; i++) {
                qArr[i] = query.get(i);
                muArr[i] = mu.get(i);
            }

            this.tildeQ = new float[d];
            final float[][] A = ash.stiefelTransform.AFloat;
            for (int r = 0; r < d; r++) {
                float acc = 0f;
                float[] Arow = A[r];
                for (int j = 0; j < originalDim; j++) {
                    acc += Arow[j] * (qArr[j] - muArr[j]);
                }
                tildeQ[r] = acc;
            }

            float s = 0f;
            for (int i = 0; i < d; i++) s += tildeQ[i];
            this.sumTildeQ = s;

            this.dotQMu = VectorUtil.dotProduct(query, mu);
            this.muNormSq = VectorUtil.dotProduct(mu, mu);
            this.invSqrtD = (float) (1.0 / Math.sqrt(d));
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

            final float[] acc = new float[Math.min(blockSize, count)];

            int baseOrd = start;
            int remaining = count;

            while (remaining > 0) {
                final int blockLen = Math.min(blockSize, remaining);

                for (int i = 0; i < blockLen; i++) acc[i] = 0f;

                for (int w = 0; w < words; w++) {
                    final int baseDim = w * 64;

                    for (int lane = 0; lane < blockLen; lane++) {
                        long word = vectors[baseOrd + lane].binaryVector[w];

                        while (word != 0L) {
                            int bit = Long.numberOfTrailingZeros(word);
                            int idx = baseDim + bit;
                            if (idx < d) acc[lane] += tildeQ[idx];
                            word &= (word - 1);
                        }
                    }
                }

                for (int lane = 0; lane < blockLen; lane++) {
                    final AsymmetricHashing.QuantizedVector v = vectors[baseOrd + lane];
                    final float scale = invSqrtD * v.residualNorm;

                    out[(baseOrd - start) + lane] =
                            scale * (2f * acc[lane] - sumTildeQ)
                                    + dotQMu
                                    + (v.dotWithLandmark - muNormSq);
                }

                baseOrd += blockLen;
                remaining -= blockLen;
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
    private static final class RegisterAccASHBlockScorer implements ASHBlockScorer {

        private final AsymmetricHashing ash;
        private final AsymmetricHashing.QuantizedVector[] vectors;
        private final float[] residualNorms;
        private final float[] dotXMu;

        private final float[] tildeQ;
        private final float sumTildeQ;
        private final float dotQMu;
        private final float muNormSq;
        private final float invSqrtD;

        private final int d;
        private final int words;

        RegisterAccASHBlockScorer(
                AsymmetricHashing ash,
                AsymmetricHashing.QuantizedVector[] vectors,
                float[] residualNorms,
                float[] dotXMu,
                VectorFloat<?> query) {

            this.ash = ash;
            this.vectors = vectors;
            this.residualNorms = residualNorms;
            this.dotXMu = dotXMu;

            this.d = ash.quantizedDim;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            final int D = ash.originalDimension;
            final VectorFloat<?> mu = ash.globalMean;

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
            this.muNormSq = VectorUtil.dotProduct(mu, mu);
            this.invSqrtD = (float) (1.0 / Math.sqrt(d));
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

                float scale = invSqrtD * residualNorms[idx];
                out[i] =
                        scale * (2f * maskedAdd - sumTildeQ)
                                + dotQMu
                                + (dotXMu[idx] - muNormSq);
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
