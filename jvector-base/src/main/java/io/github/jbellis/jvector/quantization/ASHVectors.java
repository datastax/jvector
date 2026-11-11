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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.Objects;

/**
 * Container for ASH-compressed vectors with scalar, SIMD, and block scoring.
 * All public node and block scorers return nonnegative DOT_PRODUCT similarities,
 * {@code max(0, (1 + rawDotProduct) / 2)}, rather than raw ASH dot products.
 *
 * <p>
 * This class implements the functionality required for:
 * <ul>
 *   <li>Scalar scoring</li>
 *   <li>Block scoring</li>
 *   <li>Vectorized versions of scoring</li>
 * </ul>
 *
 * <p>Codes are held in one compact {@link ASHCodes} store with no per-vector objects;
 * every scorer reads them in place. Fused graph neighborhoods are scored separately by
 * {@link FusedASHDecoder}, from the on-disk index, using the same similarity scale and
 * shared numerical kernels.</p>
 */
@Experimental
public class ASHVectors implements CompressedVectors {

    final AsymmetricHashing ash;
    /** The only copy of the stored codes and headers; see {@link ASHCodes}. */
    final ASHCodes codes;
    final ASHScorer scorer;
    private final long containerBytes;
    // Immutable, optional construction cache produced by the bounded encoding workers.
    // Raw input vectors are never retained. Loaded/individually encoded collections use
    // the scorer's existing on-demand preparation instead.
    final float[] symmetricOffsets;
    private final java.util.concurrent.atomic.AtomicLong projectionCacheBytes = new java.util.concurrent.atomic.AtomicLong();

    // Conservatively include a key, node and initial-sized table per cached layout.
    // Arrays dominate this charge; no map traversal is needed when Cassandra polls it.
    private static final long PROJECTION_CACHE_ENTRY_BYTES =
            RamUsageEstimator.shallowSizeOfInstance(Integer.class)
            + RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
                    + Integer.BYTES + 4L * RamUsageEstimator.NUM_BYTES_OBJECT_REF)
            + RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                    + 16L * RamUsageEstimator.NUM_BYTES_OBJECT_REF);

    private volatile ASHSymmetricScorer diversityScorer;

    enum AshSingleKernel {
        AUTO,
        SCALAR,
        SIMD;

        static AshSingleKernel fromProperty() {
            String v = System.getProperty("jvector.ash.singleKernel", "auto").trim().toLowerCase(java.util.Locale.ROOT);
            if ("scalar".equals(v)) return SCALAR;
            if ("simd".equals(v))   return SIMD;
            if ("auto".equals(v)) return AUTO;
            throw new IllegalArgumentException("jvector.ash.singleKernel must be auto, scalar or simd: " + v);
        }
    }

    enum AshBlockKernel {
        AUTO,
        SCALAR,
        SIMD;

        static AshBlockKernel fromProperty() {
            String v = System.getProperty("jvector.ash.blockKernel", "auto").trim().toLowerCase(java.util.Locale.ROOT);
            if ("scalar".equals(v)) return SCALAR;
            if ("simd".equals(v))   return SIMD;
            if ("auto".equals(v)) return AUTO;
            throw new IllegalArgumentException("jvector.ash.blockKernel must be auto, scalar or simd: " + v);
        }
    }

    // Decoded headers, owned by codes; aliased for the scorers' hot loops.
    private final float[] scales;   // scale_i = ||x_i − μ|| / ||code_i||
    private final float[] offsets;  // offset_i = <x_i, μ> − ||μ||^2
    private final byte[] landmarks;

    // Projected dimensions and sign-bit word count
    private final int d;
    private final int words;

    // Packed block-column-major sign bits (built lazily per blockSize, 1-bit path only). Published
    // whole, after it is filled, so concurrent queries never see a partial array or another size.
    private static final class PackedBits {
        final int blockSize;
        final long[] words;

        PackedBits(int blockSize, long[] words) {
            this.blockSize = blockSize;
            this.words = words;
        }
    }

    private volatile PackedBits packedBits;
    private final Object packedBitsLock = new Object();

    // Immutable, body-only nibble blocks shared by queries. Header arrays contain
    // decoded binary16 values; the serialized header remains 40 bits per vector.
    private final java.util.concurrent.ConcurrentMap<Integer, byte[][]> projectionBlocks =
            new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Initialize ASHVectors by copying an array of ASH-compressed vectors into compact storage.
     * The vectors are not retained, so later changes to them are not seen.
     *
     * @throws IllegalArgumentException if a vector does not have this compressor's code shape
     */
    public ASHVectors(AsymmetricHashing ash,
                      AsymmetricHashing.QuantizedVector[] compressedVectors) {
        this(ash, ASHCodes.copyOf(compressedVectors, ash.quantizedDim, ash.bitsPerDimension), null);
    }

    /** Takes ownership of {@code codes}, which must not change afterward. */
    ASHVectors(AsymmetricHashing ash, ASHCodes codes, float[] symmetricOffsets) {
        if (codes.quantizedDim != ash.quantizedDim || codes.bitsPerDimension != ash.bitsPerDimension)
            throw new IllegalArgumentException("ASH codes do not match the compressor's code shape");
        if (symmetricOffsets != null && symmetricOffsets.length != codes.size)
            throw new IllegalArgumentException("Symmetric offsets must match encoded vector count");
        this.symmetricOffsets = symmetricOffsets;
        this.ash = ash;
        this.codes = codes;
        this.scorer = new ASHScorer(ash);

        this.d = ash.quantizedDim;
        this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

        this.scales = codes.scales;
        this.offsets = codes.offsets;
        this.landmarks = codes.landmarks;
        this.containerBytes = containerBytes();
    }

    @Override
    public int count() {
        return codes.size;
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
        out.writeInt(codes.size);

        // Write vectors, each in the AsymmetricHashing.QuantizedVector format
        codes.write(out);
    }

    public static ASHVectors load(RandomAccessReader in) throws IOException {
        var ash = AsymmetricHashing.load(in);

        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }

        return new ASHVectors(ash, ASHCodes.load(in, size, ash.quantizedDim, ash.bitsPerDimension), null);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException("ASH scorer supports DOT_PRODUCT only");
        }

        // ASHScorer dispatches canonical 2/4-bit projection scoring to SIMD when available.
        // The packed masked-add path below is specific to one-bit codes.
        if (ash.bitsPerDimension != 1) {
            return nodeScoreFunction(scorer.codeScorerFor(scorer.precomputeQuery(query)));
        }

        final AshSingleKernel kernel = AshSingleKernel.fromProperty();
        final VectorUtilSupport vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

        // Scalar forced
        if (kernel == AshSingleKernel.SCALAR) {
            return nodeScoreFunction(scorer.codeScorerFor(scorer.precomputeQuery(query)));
        }

        // AUTO/SIMD: use SIMD scorer only when backend supports masked-load kernel;
        // otherwise fall back to the existing scalar behavior.
        if ((kernel == AshSingleKernel.SIMD || kernel == AshSingleKernel.AUTO)
                && vecUtil.supportsAshMaskedLoad()) {
            final QueryPrecompute qp = precomputeQuery(query);
            return new SimdASHScoreFunction(qp);
        }

        // Fallback: existing scalar behavior (stable reference)
        if (kernel == AshSingleKernel.SIMD) {
            throw new IllegalStateException("ASH single-vector SIMD is unavailable on " + vecUtil.getClass().getName());
        }
        return nodeScoreFunction(scorer.codeScorerFor(scorer.precomputeQuery(query)));
    }

    /**
     * Applies {@code f} to stored codes in place, in graph similarity units. One lambda per storage
     * shape keeps each call site free of null checks for the array a bit width does not use.
     */
    private ScoreFunction.ApproximateScoreFunction nodeScoreFunction(ASHScorer.CodeScorer f) {
        final ASHCodes c = codes;
        if (c.signWords == 0) {
            return node -> ASHScorer.toSimilarity(f.dot(null, 0, c.bodyPage(node), c.bodyOffset(node),
                    scales[node], offsets[node], landmarks[node] & 0xFF));
        }
        if (c.bodyBytes == 0) {
            return node -> ASHScorer.toSimilarity(f.dot(c.signPage(node), c.signOffset(node), null, 0,
                    scales[node], offsets[node], landmarks[node] & 0xFF));
        }
        return node -> ASHScorer.toSimilarity(f.dot(c.signPage(node), c.signOffset(node),
                c.bodyPage(node), c.bodyOffset(node), scales[node], offsets[node], landmarks[node] & 0xFF));
    }

    // ============================================================================
    // ASH single vector scoring (1-bit only, reads the stored sign pages in place)
    // ============================================================================

    private final class SimdASHScoreFunction implements ScoreFunction.ApproximateScoreFunction {
        private final QueryPrecompute qp;
        private final VectorUtilSupport vecUtil;

        // Query state and the stored sign pages, read in place
        private final float[] qProj;
        private final ASHCodes codes;

        private final int d;
        private final int words;

        SimdASHScoreFunction(QueryPrecompute qp) {
            if (ash.bitsPerDimension != 1) {
                throw new IllegalStateException("SIMD ASH scorer supports only 1-bit ASH");
            }

            this.qp = qp;
            this.vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();
            this.d = qp.d;
            this.words = ASHVectors.this.words; // access outer class field

            this.qProj = qp.qProj;

            this.codes = ASHVectors.this.codes;
        }

        @Override
        public float similarityTo(int node2) {

            final int c = landmarks[node2] & 0xFF;

            // Run existing optimized 1-bit kernel on the node's sign words in its page.
            final float maskedAdd = vecUtil.ashMaskedAdd_512(
                    qProj,
                    0,
                    codes.signPage(node2),
                    codes.signOffset(node2),
                    d,
                    words
            );

            return ASHScorer.toSimilarity(scales[node2] * (2f * maskedAdd - qp.sumQProj)
                    + qp.dotQMu[c] + offsets[node2]);
        }

        @Override
        public boolean isExact() {
            return false;
        }
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
    private static final class LegacyASHBlockScorer implements ASHBlockScorer {

        /** Per-node scorer already in graph similarity units. */
        private final ScoreFunction.ApproximateScoreFunction scorer;
        private final int size;

        LegacyASHBlockScorer(ScoreFunction.ApproximateScoreFunction scorer, int size) {
            this.scorer = scorer;
            this.size = size;
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > size) {
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
                out[o] = scorer.similarityTo(i);
            }
        }
    }

    /**
     * Returns the independent scalar reference scorer, regardless of kernel properties.
     * Use the overload accepting a block size for automatic or forced SIMD dispatch.
     */
    public ASHBlockScorer blockScorerFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {

        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException(
                    "ASH block scorer supports DOT_PRODUCT only");
        }

        return new LegacyASHBlockScorer(nodeScoreFunction(scorer.scalarCodeScorerFor(query)), codes.size);
    }

    /**
     * Returns a block-oriented ASH scorer for contiguous vectors.
     *
     * <p>
     * This factory selects between a portable scalar implementation and a
     * SIMD implementation. One-bit codes use masked adds; 2/4-bit codes use
     * nibble lookup tables over interleaved blocks of 8, 16 or 32 vectors.
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
     *   <li><b>auto</b> (default): use SIMD when supported by the active backend</li>
     *   <li><b>scalar</b>: force portable scalar accumulation</li>
     *   <li><b>simd</b>: require SIMD; fail if unavailable</li>
     * </ul>
     *
     * <p>
     * Scorers are numerically equivalent within floating-point accumulation error.
     * Packed multi-bit bodies are cached per block size and shared across queries.
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

        // Every multibit width scores 4-bit groups of its canonical code through the query LUT.
        if (ash.bitsPerDimension != 1) {
            FusedASHLayout.validateBlockSize(blockSize);
            return new LutASHBlockScorer(query, blockSize);
        }

        final AshBlockKernel kernel = AshBlockKernel.fromProperty();
        final boolean supportsSimd = VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshMaskedLoad();
        if (kernel == AshBlockKernel.SIMD && !supportsSimd) {
            throw new IllegalStateException("ASH masked-add SIMD is unavailable on the active backend");
        }

        // Compute query precompute ONCE (shared by scalar + simd)
        final QueryPrecompute qp = precomputeQuery(query);

        // ------------------------------------------------------------
        // Forced scalar path (portable register-accumulator)
        // ------------------------------------------------------------
        if (kernel == AshBlockKernel.SCALAR || !supportsSimd) {
            return new ScalarASHBlockScorer(
                    codes,
                    scales,
                    offsets,
                    landmarks,
                    qp
            );
        }

        // ------------------------------------------------------------
        // SIMD path
        // ------------------------------------------------------------
        return new SimdASHBlockScorer(
                codes.size,
                scales,
                offsets,
                landmarks,
                packedBitsFor(blockSize),
                blockSize,
                qp
        );
    }

    // Package-private prepared-state entry points shared by symmetric scoring.
    ScoreFunction.ApproximateScoreFunction scoreFunctionFor(ASHScorer.QueryPrecompute qp, boolean simd) {
        if (ash.bitsPerDimension == 1 && simd) {
            return new SimdASHScoreFunction(new QueryPrecompute(qp.d, qp.C, qp.qProj,
                    qp.sumQProj, qp.dotQMuByLandmark));
        }
        return nodeScoreFunction(scorer.codeScorerFor(qp, simd));
    }

    ASHBlockScorer blockScorerFor(ASHScorer.QueryPrecompute prepared, int blockSize, boolean simd) {
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize must be > 0");
        if (ash.bitsPerDimension != 1) {
            FusedASHLayout.validateBlockSize(blockSize);
            return new LutASHBlockScorer(prepared, blockSize, simd);
        }
        var qp = new QueryPrecompute(prepared.d, prepared.C, prepared.qProj,
                prepared.sumQProj, prepared.dotQMuByLandmark);
        if (!simd) return new ScalarASHBlockScorer(codes, scales, offsets, landmarks, qp);
        return new SimdASHBlockScorer(codes.size, scales, offsets, landmarks, packedBitsFor(blockSize), blockSize, qp);
    }

    private byte[][] packProjectionBlocks(int blockSize) {
        int count = codes.size;
        int bytes = FusedASHLayout.canonicalCodeBytes(d, ash.bitsPerDimension);
        if (bytes > codes.bodyBytes) throw new IllegalStateException("ASH projection codes are shorter than their layout");
        byte[][] blocks = new byte[count / blockSize + (count % blockSize == 0 ? 0 : 1)][];
        long bytesUsed = RamUsageEstimator.shallowSizeOf(blocks);
        for (int block = 0; block < blocks.length; block++) {
            byte[] body = new byte[Math.multiplyExact(bytes, blockSize)];
            int base = block * blockSize;
            for (int lane = 0; lane < Math.min(blockSize, count - base); lane++) {
                byte[] page = codes.bodyPage(base + lane);
                int at = codes.bodyOffset(base + lane);
                for (int i = 0; i < bytes; i++) body[i * blockSize + lane] = page[at + i];
            }
            blocks[block] = body;
            bytesUsed = Math.addExact(bytesUsed, RamUsageEstimator.sizeOf(body));
        }
        // computeIfAbsent serializes creation for a layout. Failed packing is not charged.
        projectionCacheBytes.accumulateAndGet(Math.addExact(bytesUsed, PROJECTION_CACHE_ENTRY_BYTES), Math::addExact);
        return blocks;
    }

    byte[][] symmetricProjectionBlocks(int blockSize) {
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize must be positive");
        return projectionBlocks.computeIfAbsent(blockSize, this::packProjectionBlocks);
    }

    /** Query-local LUT and dispatch over immutable, container-owned packed blocks. */
    private final class LutASHBlockScorer implements ASHBlockScorer {
        private final ASHLutKernel kernel;
        private final int blockSize;
        private final byte[][] blocks;
        private final float[] lut;
        private final float[] dotQMu;
        private final int groups = FusedASHLayout.codeGroups(d, ash.bitsPerDimension);

        LutASHBlockScorer(VectorFloat<?> query, int blockSize) {
            this.kernel = new ASHLutKernel();
            this.blockSize = blockSize;
            this.blocks = projectionBlocks.computeIfAbsent(blockSize, ASHVectors.this::packProjectionBlocks);
            float[] q = new float[ash.originalDimension];
            for (int i = 0; i < q.length; i++) q[i] = query.get(i);
            float[] projected = new float[d];
            ash.stiefelTransform.projectQuery(q, projected);
            lut = new float[Math.multiplyExact(groups, 16)];
            FusedASHLayout.buildQueryLut(projected, d, ash.bitsPerDimension, lut);
            dotQMu = new float[ash.landmarkCount];
            for (int i = 0; i < dotQMu.length; i++) dotQMu[i] = VectorUtil.dotProduct(query, ash.landmarks[i]);
        }

        LutASHBlockScorer(ASHScorer.QueryPrecompute prepared, int blockSize, boolean simd) {
            this.kernel = new ASHLutKernel(simd);
            this.blockSize = blockSize;
            this.blocks = projectionBlocks.computeIfAbsent(blockSize, ASHVectors.this::packProjectionBlocks);
            this.lut = new float[Math.multiplyExact(groups, 16)];
            FusedASHLayout.buildQueryLut(prepared.qProj, d, ash.bitsPerDimension, lut);
            this.dotQMu = prepared.dotQMuByLandmark;
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            Objects.checkFromIndexSize(start, count, codes.size);
            Objects.checkFromIndexSize(0, count, out.length);
            for (int done = 0; done < count;) {
                int ordinal = start + done;
                int lane = ordinal % blockSize;
                int lanes = Math.min(count - done, blockSize - lane);
                kernel.score(blocks[ordinal / blockSize], 0, groups, blockSize,
                        lane, lanes, lut, out, done);
                for (int i = 0; i < lanes; i++) {
                    int node = ordinal + i;
                    out[done + i] = ASHScorer.toSimilarity(scales[node] * out[done + i]
                            + dotQMu[landmarks[node] & 255] + offsets[node]);
                }
                done += lanes;
            }
        }

        @Override
        public String toString() { return "LutASHBlockScorer[" + kernel + ", blockSize=" + blockSize + "]"; }

        @Override
        public String description() { return toString(); }
    }

    /** The packed sign words for {@code blockSize}; callers score with exactly the array returned. */
    private long[] packedBitsFor(int blockSize) {
        if (ash.bitsPerDimension != 1) {
            throw new IllegalStateException("Packed sign-bit block cache supports only 1-bit ASH");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be > 0");
        }
        PackedBits current = packedBits;
        if (current != null && current.blockSize == blockSize) {
            return current.words;
        }
        synchronized (packedBitsLock) {
            current = packedBits;
            if (current == null || current.blockSize != blockSize) {
                current = new PackedBits(blockSize, packBits(blockSize));
                packedBits = current;
            }
            return current.words;
        }
    }

    private long[] packBits(int blockSize) {
        final int n = codes.size;
        final int packedBlockCount = (n + blockSize - 1) / blockSize;
        final long[] packedBits = new long[packedBlockCount * words * blockSize];

        for (int b = 0; b < packedBlockCount; b++) {
            int baseOrd = b * blockSize;
            int blockLen = Math.min(blockSize, n - baseOrd);

            for (int w = 0; w < words; w++) {
                int dstBase = (b * words + w) * blockSize;
                for (int lane = 0; lane < blockLen; lane++) {
                    int node = baseOrd + lane;
                    packedBits[dstBase + lane] = codes.signPage(node)[codes.signOffset(node) + w];
                }
                // remaining lanes default to 0
            }
        }
        return packedBits;
    }

    /**
     * Block-oriented 1-bit ASH scorer using register-local accumulation.
     *
     * <p>
     * This is the preferred scalar block implementation for 1-bit ASH. It avoids
     * temporary per-block arrays and accumulates the masked-add term directly
     * into scalar registers.
     * </p>
     */
    private static final class ScalarASHBlockScorer implements ASHBlockScorer {

        private final ASHCodes codes;
        private final float[] scales;
        private final float[] offsets;
        private final byte[] landmarks;

        private final float[] qProj;        // [d]
        private final float sumQProj;
        private final float[] dotQMu;       // [C]
        private final int d;
        private final int words;

        ScalarASHBlockScorer(
                ASHCodes codes,
                float[] scales,
                float[] offsets,
                byte[] landmarks,
                QueryPrecompute qp)
        {

            this.codes = codes;
            this.scales = scales;
            this.offsets = offsets;
            this.landmarks = landmarks;

            this.d = qp.d;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            this.qProj = qp.qProj;
            this.sumQProj = qp.sumQProj;
            this.dotQMu = qp.dotQMu;
        }

        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > codes.size) {
                throw new IllegalArgumentException("Range out of bounds: [" + start + ", " + (start + count) + ")");
            }
            if (out.length < count) throw new IllegalArgumentException("out.length < count");
            if (count == 0) return;

            for (int i = 0; i < count; i++) {
                final int idx = start + i;

                final int c = landmarks[idx] & 0xFF; // unsigned [0,C)

                float maskedAdd = 0f;

                // bit-walk against Aq; the landmark term is folded into the offset
                final long[] bits = codes.signPage(idx);
                final int bitsOffset = codes.signOffset(idx);
                int baseDim = 0;
                for (int w = 0; w < words && baseDim < d; w++, baseDim += 64) {
                    long word = bits[bitsOffset + w];
                    while (word != 0L) {
                        int bit = Long.numberOfTrailingZeros(word);
                        int j = baseDim + bit;
                        if (j < d) maskedAdd += qProj[j];
                        word &= (word - 1);
                    }
                }

                out[i] = ASHScorer.toSimilarity(
                        scales[idx] * (2f * maskedAdd - sumQProj)
                                + dotQMu[c]
                                + offsets[idx]);
            }
        }
    }

    /**
     * Block-oriented 1-bit ASH scorer using SIMD masked-loads with register-local
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
    private static final class SimdASHBlockScorer implements ASHBlockScorer {

        private final int size;
        private final float[] scales;
        private final float[] offsets;
        private final byte[] landmarks;

        private final long[] packedBits;
        private final int blockSize;

        private final int d;
        private final int words;

        private final float[] maskedAdds;     // [blockSize]

        private final float[] qProj;          // qp.qProj [d]
        private final float sumQProj;
        private final float[] dotQMu;         // qp.dotQMu [C]

        private final VectorUtilSupport vecUtil;

        SimdASHBlockScorer(
                int size,
                float[] scales,
                float[] offsets,
                byte[] landmarks,
                long[] packedBits,
                int blockSize,
                QueryPrecompute qp) {

            this.vecUtil = VectorizationProvider.getInstance().getVectorUtilSupport();

            this.size = size;
            this.scales = scales;
            this.offsets = offsets;
            this.landmarks = landmarks;

            this.packedBits = packedBits;
            this.blockSize = blockSize;

            this.d = qp.d;
            this.words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

            this.maskedAdds = new float[blockSize];

            this.qProj = qp.qProj;
            this.sumQProj = qp.sumQProj;
            this.dotQMu = qp.dotQMu;
        }

        // The masked add reads only Aq, so one kernel call covers each packed block's lanes in the range,
        // whatever their landmarks; each lane adds its own landmark's <q, mu_c>.
        @Override
        public void scoreRange(int start, int count, float[] out) {
            if (count < 0) throw new IllegalArgumentException("count must be >= 0");
            if (start < 0 || start + count > size) {
                throw new IllegalArgumentException("Range out of bounds: [" + start + ", " + (start + count) + ")");
            }
            if (out.length < count) throw new IllegalArgumentException("out.length < count");
            if (count == 0) return;

            int ord = start;
            final int end = start + count;

            while (ord < end) {
                final int blockId = ord / blockSize;
                final int laneStart = ord - blockId * blockSize;
                final int runLen = Math.min(blockSize - laneStart, end - ord);
                final int blockWordBase = blockId * words * blockSize;

                vecUtil.ashMaskedAddBlockAllWords(
                        qProj,
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
                    out[idx - start] = ASHScorer.toSimilarity(
                            // scale the 0/1 stored bits to +/-1 for processing.
                            scales[idx] * (2f * maskedAdds[lane] - sumQProj)
                                    + dotQMu[landmarks[idx] & 0xFF]
                                    + offsets[idx]);
                }

                ord += runLen;
            }
        }
    }

    private QueryPrecompute precomputeQuery(VectorFloat<?> query) {
        if (ash.bitsPerDimension != 1) {
            throw new IllegalStateException("ASHVectors precomputeQuery supports only 1-bit ASH");
        }

        final int d = ash.quantizedDim;
        final int D = ash.originalDimension;
        final int C = ash.landmarkCount;

        // q materialized once
        final float[] qArr = new float[D];
        for (int i = 0; i < D; i++) {
            qArr[i] = query.get(i);
        }

        // Aq
        final float[] qProj = new float[d];
        ash.stiefelTransform.projectQuery(qArr, qProj);
        final float[] dotQMu = new float[C];
        for (int c = 0; c < C; c++) {
            dotQMu[c] = VectorUtil.dotProduct(query, ash.landmarks[c]);
        }

        return new QueryPrecompute(d, C, qProj, ASHScorer.sum(qProj), dotQMu);
    }

    static final class QueryPrecompute {
        final int d;
        final int C;
        final float[] qProj;               // [d], Aq
        final float sumQProj;              // <Aq, 1>
        final float[] dotQMu;              // [C]

        QueryPrecompute(int d, int C,
                        float[] qProj,
                        float sumQProj,
                        float[] dotQMu) {
            this.d = d;
            this.C = C;
            this.qProj = qProj;
            this.sumQProj = sumQProj;
            this.dotQMu = dotQMu;
        }
    }

    public static final class LandmarkOrder {
        public final ASHVectors vectors;
        public final int[] newToOld;
        public final int[] oldToNew;
        public final int[] landmarkOffsets; // length C+1, last == n

        LandmarkOrder(ASHVectors vectors, int[] newToOld, int[] oldToNew, int[] landmarkOffsets) {
            this.vectors = vectors;
            this.newToOld = newToOld;
            this.oldToNew = oldToNew;
            this.landmarkOffsets = landmarkOffsets;
        }
    }

    /**
     * Stable, linear-time reorder by landmark id ({@code C <= 256}).
     * The result holds its own reordered copy of the codes.
     */
    public LandmarkOrder reorderByLandmarkFast() {
        final int n = codes.size;
        final int C = ash.landmarkCount;

        // Count per landmark
        final int[] counts = new int[C];
        for (int i = 0; i < n; i++) {
            int c = this.landmarks[i] & 0xFF;
            if (c >= C) {
                throw new IllegalStateException("Invalid landmark id " + c + " at ordinal " + i + " (C=" + C + ")");
            }
            counts[c]++;
        }

        // Prefix sums: landmarkOffsets[c] is start of landmark c, length C+1
        final int[] landmarkOffsets = new int[C + 1];
        landmarkOffsets[0] = 0;
        for (int c = 0; c < C; c++) {
            landmarkOffsets[c + 1] = landmarkOffsets[c] + counts[c];
        }
        if (landmarkOffsets[C] != n) {
            throw new IllegalStateException("landmarkOffsets[C] != n: " + landmarkOffsets[C] + " != " + n);
        }

        final int[] writePos = landmarkOffsets.clone();

        final float[] newSymmetricOffsets = symmetricOffsets == null ? null : new float[n];
        final int[] newToOld = new int[n];

        // Preserve an ordinal map so we can go back and forth
        for (int oldOrd = 0; oldOrd < n; oldOrd++) {
            final int c = this.landmarks[oldOrd] & 0xFF;
            final int newOrd = writePos[c]++;
            if (newSymmetricOffsets != null) newSymmetricOffsets[newOrd] = symmetricOffsets[oldOrd];
            newToOld[newOrd] = oldOrd;
        }

        final int[] oldToNew = new int[n];
        for (int newOrd = 0; newOrd < n; newOrd++) {
            oldToNew[newToOld[newOrd]] = newOrd;
        }

        ASHVectors reordered = new ASHVectors(ash, codes.permute(newToOld), newSymmetricOffsets);
        return new LandmarkOrder(reordered, newToOld, oldToNew, landmarkOffsets);
    }

    public static final class LandmarkRunStats {
        public final int n;
        public final int runCount;
        public final double avgRunLength;
        public final int maxRunLength;
        public final double stddevRunLength;

        LandmarkRunStats(int n, int runCount, double avgRunLength, int maxRunLength, double stddevRunLength) {
            this.n = n;
            this.runCount = runCount;
            this.avgRunLength = avgRunLength;
            this.maxRunLength = maxRunLength;
            this.stddevRunLength = stddevRunLength;
        }

        @Override
        public String toString() {
            return String.format(
                    java.util.Locale.ROOT,
                    "runs=%d, avgRun=%.2f, maxRun=%d, stddev=%.2f (n=%d)",
                    runCount, avgRunLength, maxRunLength, stddevRunLength, n
            );
        }
    }

    public LandmarkRunStats landmarkRunStats() {
        return computeLandmarkRunStats(this.landmarks);
    }

    public static LandmarkRunStats computeLandmarkRunStats(byte[] landmarks) {
        final int n = landmarks.length;
        if (n == 0) {
            return new LandmarkRunStats(0, 0, 0.0, 0, 0.0);
        }

        int runCount = 0;
        long sum = 0;
        long sumSq = 0;
        int max = 0;

        int i = 0;
        while (i < n) {
            final byte c = landmarks[i];
            int j = i + 1;
            while (j < n && landmarks[j] == c) j++;
            final int len = j - i;

            runCount++;
            sum += len;
            sumSq += (long) len * (long) len;
            if (len > max) max = len;

            i = j;
        }

        final double avg = (double) sum / (double) runCount;
        final double meanSq = (double) sumSq / (double) runCount;
        final double var = Math.max(0.0, meanSq - avg * avg);
        final double stddev = Math.sqrt(var);

        return new LandmarkRunStats(n, runCount, avg, max, stddev);
    }

    /**
     * Same as {@link #scoreFunctionFor}: ASH prepares the query once, projection included,
     * when the score function is created, so there is nothing further to precompute.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
    }

    /** Symmetric node-to-node dot-product similarity from the stored encoded vectors. */
    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(
            int node1,
            VectorSimilarityFunction similarityFunction) {
        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT)
            throw new UnsupportedOperationException("ASH diversity scoring supports DOT_PRODUCT only");
        ASHSymmetricScorer symmetric = diversityScorer;
        if (symmetric == null) {
            synchronized (this) {
                symmetric = diversityScorer;
                if (symmetric == null) diversityScorer = symmetric = new ASHSymmetricScorer(this);
            }
        }
        return symmetric.scoreFunctionFor(node1);
    }

    /**
     * A new, independent copy of the stored vector. Allocates; for repeated reads into reusable
     * storage use {@link #copyTo}. Scorers read the stored codes in place.
     */
    public AsymmetricHashing.QuantizedVector get(int ordinal) {
        return codes.get(ordinal);
    }

    /**
     * Copies the stored vector into {@code dest}, replacing its arrays only if their lengths do not
     * match this compressor's code shape.
     */
    public void copyTo(int ordinal, AsymmetricHashing.QuantizedVector dest) {
        codes.copyTo(ordinal, dest);
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

    private long containerBytes() {
        return RamUsageEstimator.shallowSizeOfInstance(ASHVectors.class)
                + RamUsageEstimator.shallowSizeOf(scorer)
                + RamUsageEstimator.shallowSizeOf(symmetricOffsets)
                + RamUsageEstimator.shallowSizeOf(projectionBlocks)
                + RamUsageEstimator.shallowSizeOf(projectionCacheBytes);
    }

    /**
     * Retained compressor, code storage and published caches, analogous to PQVectors.
     * Query-local scorers and worker scratch are owned by callers and excluded here.
     * Code storage is fixed at construction; lazy cache totals are added at allocation.
     */
    @Override
    public long ramBytesUsed() {
        ASHSymmetricScorer symmetric = diversityScorer;
        PackedBits packed = packedBits;
        return ash.ramBytesUsed() + containerBytes + codes.ramBytesUsed() + projectionCacheBytes.get()
                + (packed == null ? 0 : RamUsageEstimator.shallowSizeOf(packed.words))
                + (symmetric == null ? 0 : symmetric.ramBytesUsed());
    }

    /** Equal compressors and the same stored vectors in the same order. */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ASHVectors that = (ASHVectors) o;
        return Objects.equals(ash, that.ash) && codes.contentEquals(that.codes);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(ash);
        result = 31 * result + codes.contentHash();
        return result;
    }

    @Override
    public String toString() {
        return "ASHVectors{count=" + codes.size +
                ", ash=" + ash + '}';
    }
}
