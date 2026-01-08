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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import dev.ludovic.netlib.blas.BLAS;
import dev.ludovic.netlib.lapack.LAPACK;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.netlib.util.intW;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Asymmetric Hashing (ASH) for float vectors.
 * Encodes each vector into a fixed-length binary code using a learned or random
 * orthonormal projection and sign thresholding.
 */
public class AsymmetricHashing implements VectorCompressor<AsymmetricHashing.QuantizedVector>, Accountable {
    public static final int ITQ = 1, LANDING = 2, RANDOM = 3;
    private static final int MAGIC = 0x75EC4012;  // TODO update the magic number?

    // ---------------------------------------------------------------------
    // Training configuration (reference defaults; TODO tune later)
    // ---------------------------------------------------------------------

    /**
     * Training sample size for ITQ. Default mirrors a practical upper bound for large datasets.
     */
    private static final int ITQ_N_TRAIN = 100_000;

    /**
     * Training sample size for LANDING. Reduce if LANDING becomes too expensive at larger D/d.
     */
    private static final int LANDING_N_TRAIN = 100_000;

    /** Tune for tradeoff between convergence rate and processing speed. */
    private static final int LANDING_BATCH_SIZE = 256;

    /** Final setting TBD. */
    private static final int TRAINING_ITERS = 25;

    // Physical header size, reflecting actual stored fields:
    //  - scale: float (32 bits), where scale = ||x − μ|| / sqrt(d)
    //  - offset: float (32 bits), where offset = <x, μ> − ||μ||_2^2
    //  - landmark id: byte (8 bits) in [0, C)
    //
    // NOTE: residualNorm (= ||x − μ||) is still computed during encoding for Eq. 6,
    // but it is not stored explicitly once scale is stored.
    private static final int HEADER_BITS =
            (Float.BYTES + Float.BYTES + Byte.BYTES) * 8; // 72 bits currently

    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    // ---------------------------------------------------------------------
    // Index-wide immutable state
    // ---------------------------------------------------------------------

    /** Number of landmarks (clusters), C<= 64 */
    public final int landmarkCount;

    /** Landmark centroids μ_0... μ_{C−1} */
    public final VectorFloat<?>[] landmarks;

    // One entry per landmark
    private final float[] landmarkNormSq;

    /** Original (uncompressed) dimensionality */
    public final int originalDimension;

    /** Total bits per encoded vector (header + body) */
    public final int encodedBits;

    /** Number of bits in the binary body */
    public final int quantizedDim;

    /** Optimizer / learning mode */
    public final int optimizer;

    /** Learned or random Stiefel transform */
    public final StiefelTransform stiefelTransform;

    /**
     * Precomputed landmark projections in the reduced space:
     *   landmarkProj[c][j] = <A[j], μ_c>  where A is [d][D] and μ_c is length D.
     *
     * Shape: [landmarkCount][quantizedDim]
     */
    public final float[][] landmarkProj;

    /**
     * Precomputed sums of the landmark projections:
     *   landmarkProjSum[c] = Σ_j landmarkProj[c][j]
     *
     * Shape: [landmarkCount]
     */
    public final float[] landmarkProjSum;

    /** Debug hook to disable learning paths */
    @VisibleForTesting
    public boolean learn = true;

    // ---------------------------------------------------------------------
    // Quantizer instances (cached)
    // ---------------------------------------------------------------------

    private final BinaryQuantizer randomQuantizer = new RandomBinaryQuantizer();
    private final BinaryQuantizer itqQuantizer = new ItqBinaryQuantizer();
    private final BinaryQuantizer landingQuantizer = new LandingBinaryQuantizer();

    private BinaryQuantizer quantizer() {
        switch (optimizer) {
            case RANDOM:
                return randomQuantizer;
            case ITQ:
                return itqQuantizer;
            case LANDING:
                return landingQuantizer;
            default:
                throw new IllegalStateException("Unknown optimizer " + optimizer);
        }
    }

    /**
     * Class constructor.
     * @param originalDim the dimensionality of the vectors to be quantized
     * @param landmarks the landmarks using for vectors (up to C in landmarkCount)
     */
    private AsymmetricHashing(int originalDim,
                              int encodedBits,
                              int quantizedDim,
                              int optimizer,
                              VectorFloat<?>[] landmarks,
                              StiefelTransform stiefelTransform,
                              float[][] landmarkProj,
                              float[] landmarkProjSum) {
        this.originalDimension = originalDim;
        this.encodedBits = encodedBits;
        this.quantizedDim = quantizedDim;

        // Defensive check for consistency
        if (encodedBits - HEADER_BITS != quantizedDim) {
            throw new IllegalArgumentException(
                    "Invalid ASH configuration: encodedBits=" + encodedBits +
                            ", quantizedDim=" + quantizedDim +
                            ", HEADER_BITS=" + HEADER_BITS
            );
        }

        this.optimizer = optimizer;
        this.stiefelTransform = Objects.requireNonNull(stiefelTransform);

        this.landmarks = Objects.requireNonNull(landmarks);
        assert landmarks.length > 0;
        this.landmarkCount = landmarks.length;

        // Precompute ||μ_c||^2 for each landmark (even when C = 1)
        this.landmarkNormSq = new float[landmarkCount];
        for (int c = 0; c < landmarkCount; c++) {
            this.landmarkNormSq[c] =
                    VectorUtil.dotProduct(landmarks[c], landmarks[c]);
        }

        this.landmarkProj = Objects.requireNonNull(landmarkProj, "landmarkProj");
        this.landmarkProjSum = Objects.requireNonNull(landmarkProjSum, "landmarkProjSum");

        if (landmarkProj.length != landmarkCount) {
            throw new IllegalArgumentException("landmarkProj outer dimension mismatch: "
                    + landmarkProj.length + " != landmarkCount=" + landmarkCount);
        }
        if (landmarkProjSum.length != landmarkCount) {
            throw new IllegalArgumentException("landmarkProjSum length mismatch: "
                    + landmarkProjSum.length + " != landmarkCount=" + landmarkCount);
        }
        for (int c = 0; c < landmarkCount; c++) {
            if (landmarkProj[c] == null || landmarkProj[c].length != quantizedDim) {
                throw new IllegalArgumentException("landmarkProj[" + c + "] length mismatch: "
                        + (landmarkProj[c] == null ? "null" : landmarkProj[c].length)
                        + " != quantizedDim=" + quantizedDim);
            }
        }

        if (landmarkNormSq.length != landmarkCount) {
            throw new IllegalArgumentException("landmarkNormSq length mismatch");
        }

        if (landmarkCount < 1 || landmarkCount > 64) {
            throw new IllegalArgumentException(
                    "landmarkCount must be in [1,64], got " + landmarkCount);
        }

        for (VectorFloat<?> mu : landmarks) {
            if (mu.length() != originalDimension) {
                throw new IllegalArgumentException(
                        "Landmark dimensionality does not match original dimensionality");
            }
        }

        if (quantizedDim < 1) {
            throw new IllegalArgumentException("quantizedDim must be > 0");
        }
    }

    /**
     * Initialize ASH index-wide parameters.
     *
     * @param ravv the vectors to quantize
     * @param optimizer the optimizer to use
     * @param encodedBits the number of bits used to encode vector, including the header
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount) {

        final int quantizedDim = encodedBits - HEADER_BITS;

        if (landmarkCount < 1 || landmarkCount > 64) {
            throw new IllegalArgumentException(
                    "landmarkCount must be in [1,64], got " + landmarkCount);
        }

        System.out.println(
                "\tASH initialized with landmarkCount=" + landmarkCount +
                        ", quantizedDim=" + quantizedDim
        );

        var ravvCopy = ravv.threadLocalSupplier().get();
        int originalDim = ravvCopy.getVector(0).length();
        // payload bits are always 64-bit aligned by validateEncodedBits()
        validateEncodedBits(encodedBits, HEADER_BITS, originalDim);

        // NOTE: points are treated as read-only by KMeansPlusPlusClusterer.  Materialize once.
        VectorFloat<?>[] points = new VectorFloat<?>[ravvCopy.size()];
        for (int i = 0; i < ravvCopy.size(); i++) {
            points[i] = ravvCopy.getVector(i);
        }

        VectorFloat<?>[] landmarks;
        if (landmarkCount == 1) {
            // Mean centroid
            VectorFloat<?> mu0 = vectorTypeSupport.createFloatVector(originalDim);
            for (int i = 0; i < points.length; i++) {
                VectorUtil.addInPlace(mu0, points[i]);
            }
            VectorUtil.scale(mu0, 1.0f / points.length);
            landmarks = new VectorFloat<?>[] { mu0 };
        } else {
            long kmStart = System.nanoTime();

            // KMeans++ in JVector returns centroids packed as one big vector [C * D]
            KMeansPlusPlusClusterer km = new KMeansPlusPlusClusterer(points, landmarkCount);
            VectorFloat<?> packed = km.cluster(/*unweightedIterations*/ 20, /*anisotropicIterations*/ 0);

            long kmEnd = System.nanoTime();

            System.out.printf(
                    "\tKMeans++ (C=%d, N=%d, D=%d) took %.3f seconds%n",
                    landmarkCount,
                    points.length,
                    originalDim,
                    (kmEnd - kmStart) / 1e9
            );

            landmarks = new VectorFloat<?>[landmarkCount];
            for (int c = 0; c < landmarkCount; c++) {
                VectorFloat<?> mu = vectorTypeSupport.createFloatVector(originalDim);
                mu.copyFrom(packed, c * originalDim, 0, originalDim);
                landmarks[c] = mu;
            }
        }

        // Keep determinism consistent within ASH.  TODO define seed constant or remove.
        final Random rng = new Random(42);

        StiefelTransform stiefelTransform;
        if (optimizer == RANDOM) {
            stiefelTransform = runWithoutTraining(originalDim, quantizedDim, rng);
        } else if (optimizer == ITQ || optimizer == LANDING) {
            // Build training inputs from a random sample of points.
            TrainingData td = prepareTrainingData(points, landmarks, optimizer, originalDim, rng);

            if (optimizer == ITQ) {
                stiefelTransform = runItqTrainer(td.xTrainNorm, quantizedDim, rng, TRAINING_ITERS);
            } else {
                stiefelTransform = runLandingTrainer(td.xTrainNorm, quantizedDim, rng, TRAINING_ITERS);
            }
        } else {
            throw new IllegalArgumentException("Unknown optimizer " + optimizer);
        }

        LandmarkProjections lp =
                computeLandmarkProjections(
                        landmarks,
                        stiefelTransform,
                        quantizedDim,
                        originalDim
                );

        return new AsymmetricHashing(
                originalDim,
                encodedBits,
                quantizedDim,
                optimizer,
                landmarks,
                stiefelTransform,
                lp.proj,
                lp.sum
        );
    }

    /**
     * Validates that the requested {@code encodedBits} value is compatible with
     * ASH’s block-SIMD scoring kernel.
     *
     * <p>
     * For performance reasons, the binary payload size
     * {@code (encodedBits - headerBits)} must be a multiple of 64. Non-aligned
     * payloads introduce partial-word tails that significantly degrade SIMD
     * efficiency.
     * </p>
     *
     * @throws IllegalArgumentException if the payload bit count is invalid or
     *                                  not 64-bit aligned
     */
    private static void validateEncodedBits(
            int encodedBits,
            int headerBits,
            int originalDim
    ) {
        int payloadBits = encodedBits - headerBits;

        if (payloadBits < 1) {
            throw new IllegalArgumentException(
                    "Illegal ASH payload bits: " + payloadBits +
                            " (encodedBits=" + encodedBits + ", headerBits=" + headerBits + ")"
            );
        }

        if ((payloadBits & 63) != 0) {
            throw new IllegalArgumentException(
                    "Invalid encodedBits=" + encodedBits +
                            ". ASH requires (encodedBits - headerBits) to be a multiple of 64 " +
                            "for aligned binary payload. " +
                            "Got payloadBits=" + payloadBits +
                            " (headerBits=" + headerBits + ")."
            );
        }

        if (payloadBits > originalDim) {
            throw new IllegalArgumentException(
                    "Invalid ASH payloadBits=" + payloadBits +
                            " exceeds originalDim=" + originalDim
            );
        }
    }

    /**
     * Precompute landmark projections in the reduced space:
     *
     *   landmarkProj[c][j] = <A[j], μ_c>
     *   landmarkProjSum[c] = Σ_j landmarkProj[c][j]
     *
     * This is used by both initialize() and load() to ensure
     * identical query-time math regardless of construction path.
     */
    private static LandmarkProjections computeLandmarkProjections(
            VectorFloat<?>[] landmarks,
            StiefelTransform stiefelTransform,
            int quantizedDim,
            int originalDim
    ) {
        final int C = landmarks.length;
        final float[][] A = stiefelTransform.AFloat; // [d][D]

        final var vecUtil =
                io.github.jbellis.jvector.vector.VectorizationProvider
                        .getInstance()
                        .getVectorUtilSupport();

        final float[][] landmarkProj = new float[C][quantizedDim];
        final float[] landmarkProjSum = new float[C];

        final float[] muArr = new float[originalDim];

        for (int c = 0; c < C; c++) {
            VectorFloat<?> mu = landmarks[c];

            // materialize μ once
            for (int i = 0; i < originalDim; i++) {
                muArr[i] = mu.get(i);
            }

            float sum = 0f;
            float[] proj = landmarkProj[c];

            for (int j = 0; j < quantizedDim; j++) {
                float v = vecUtil.ashDotRow(A[j], muArr);
                proj[j] = v;
                sum += v;
            }

            landmarkProjSum[c] = sum;
        }

        return new LandmarkProjections(landmarkProj, landmarkProjSum);
    }

    private static final class LandmarkProjections {
        final float[][] proj;   // [C][d]
        final float[] sum;      // [C]

        LandmarkProjections(float[][] proj, float[] sum) {
            this.proj = proj;
            this.sum = sum;
        }
    }

    /**
     * Loads an ASH instance from a RandomAccessReader.
     * This must mirror the ASH header writer exactly.
     */
    public static AsymmetricHashing load(RandomAccessReader in) throws IOException {

        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new IllegalArgumentException(
                    String.format("Invalid ASH magic: 0x%08X", magic)
            );
        }

        int version = in.readInt();
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported ASH version " + version);
        }

        int originalDimension = in.readInt();
        int encodedBits = in.readInt();
        validateEncodedBits(encodedBits, HEADER_BITS, originalDimension);
        int quantizedDim = in.readInt();

        // HEADER_BITS compatibility check
        int expectedQuantizedDim = encodedBits - HEADER_BITS;
        if (quantizedDim != expectedQuantizedDim) {
            throw new IOException(
                    "ASH format mismatch: encodedBits=" + encodedBits +
                            ", quantizedDim=" + quantizedDim +
                            ", but runtime HEADER_BITS=" + HEADER_BITS +
                            " implies quantizedDim=" + expectedQuantizedDim
            );
        }

        int optimizer = in.readInt();

        // Multi-landmark support
        int landmarkCount = in.readInt();
        if (landmarkCount < 1 || landmarkCount > 64) {
            throw new IOException("Invalid landmarkCount=" + landmarkCount);
        }

        VectorFloat<?>[] landmarks = new VectorFloat<?>[landmarkCount];
        for (int c = 0; c < landmarkCount; c++) {
            int dim = in.readInt();
            if (dim != originalDimension) {
                throw new IOException(
                        "Landmark dimension mismatch: expected " +
                                originalDimension + ", got " + dim);
            }
            landmarks[c] = vectorTypeSupport.readFloatVector(in, dim);
        }

        // NOTE: StiefelTransform loading will be added in Step 2.
        // For now, construct a placeholder or re-run the RANDOM initializer.
        StiefelTransform stiefelTransform;

        if (optimizer == RANDOM) {
            // Deterministic re-construction for now
            stiefelTransform = runWithoutTraining(
                    originalDimension,
                    quantizedDim,
                    new Random(42)
            );
        } else {
            throw new IllegalArgumentException(
                    "ASH optimizer " + optimizer + " requires learned transform; not yet supported in load()"
            );
        }

        LandmarkProjections lp =
                computeLandmarkProjections(
                        landmarks,
                        stiefelTransform,
                        quantizedDim,
                        originalDimension
                );

        return new AsymmetricHashing(
                originalDimension,
                encodedBits,
                quantizedDim,
                optimizer,
                landmarks,
                stiefelTransform,
                lp.proj,
                lp.sum
        );
    }


    /**
     * TODO confirm handling versioning correctly
     * TODO confirm handling magic number correctly
     * Writes the ASH index header to DataOutput (used for on-disk indexing)
     * @param out the DataOutput into which to write the object
     * @throws IOException if there is a problem writing to out.
     */
    public void write(DataOutput out, int version) throws IOException {
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported version " + version);
        }

        out.writeInt(MAGIC);
        out.writeInt(version);

        out.writeInt(originalDimension);
        out.writeInt(encodedBits);
        out.writeInt(quantizedDim);
        out.writeInt(optimizer);

        // Multi-landmark serialization
        out.writeInt(landmarkCount);

        for (int c = 0; c < landmarkCount; c++) {
            out.writeInt(landmarks[c].length());
            vectorTypeSupport.writeFloatVector(out, landmarks[c]);
        }

        // NOTE: stiefelTransform serialization will be added later
    }

    // ---------------------------------------------------------------------
    // Index-level quantization entry point
    // ---------------------------------------------------------------------

    public void quantizeVector(VectorFloat<?> vector,
                               float residualNorm,
                               byte landmark,
                               QuantizedVector dest) {

        if (quantizedDim > originalDimension) {
            throw new IllegalArgumentException(
                    "quantizedDim (" + quantizedDim +
                            ") exceeds original dimension (" + originalDimension + ")");
        }

        if ((landmark & 0xFF) >= landmarkCount) {
            throw new IllegalArgumentException("Invalid landmark id " + (landmark & 0xFF)
                    + " for landmarkCount=" + landmarkCount);
        }

        // Landmark-specific mean (C=1 → landmarks[0] == dataset mean)
        final VectorFloat<?> mu = landmarks[landmark];

        QuantizedVector.quantizeTo(
                vector,
                residualNorm,
                quantizedDim,
                mu,
                stiefelTransform,
                quantizer(),
                dest
        );
    }

    // ---------------------------------------------------------------------
    // Binary quantization strategies
    // ---------------------------------------------------------------------

    public interface BinaryQuantizer {
        void quantizeBody(VectorFloat<?> vector,
                          VectorFloat<?> mu,
                          float residualNorm,
                          int quantizedDim,
                          StiefelTransform stiefel,
                          long[] outWords);
    }

    public static final class RandomBinaryQuantizer implements BinaryQuantizer {

        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {

            final float[][] A = stiefel.AFloat;
            final int originalDim = stiefel.cols;

            final var vecUtil =
                    io.github.jbellis.jvector.vector.VectorizationProvider
                            .getInstance()
                            .getVectorUtilSupport();

            // Copy vector and mean once (no virtual calls in inner loop)
            final float[] x = new float[originalDim];
            final float[] muArr = new float[originalDim];
            for (int d = 0; d < originalDim; d++) {
                x[d] = vector.get(d);
                muArr[d] = mu.get(d);
            }

            // ASH paper, Eq. 6, normalization
            final float invNorm = (residualNorm > 0f) ? (1.0f / residualNorm) : 0.0f;

            // Normalized residual x̂
            final float[] xhat = new float[originalDim];
            for (int d = 0; d < originalDim; d++) {
                xhat[d] = (x[d] - muArr[d]) * invNorm;
            }

            // Binarize directly from per-row projection
            final int words = QuantizedVector.wordsForDims(quantizedDim);
            for (int w = 0; w < words; w++) {
                long bits = 0L;
                int base = w << 6;
                int rem = Math.min(64, quantizedDim - base);

                for (int j = 0; j < rem; j++) {
                    int bitIndex = base + j;
                    float acc = vecUtil.ashDotRow(A[bitIndex], xhat);
                    if (acc > 0.0f) {
                        bits |= (1L << j);
                    }
                }
                outWords[w] = bits;
            }
        }
    }

    public static final class ItqBinaryQuantizer implements BinaryQuantizer {
        // Same binarization math for all optimizers; only the learned transform differs.
        // TODO consolidate quantizeBody across learners/random
        private final RandomBinaryQuantizer delegate = new RandomBinaryQuantizer();

        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {
            delegate.quantizeBody(vector, mu, residualNorm, quantizedDim, stiefel, outWords);
        }
    }

    public static final class LandingBinaryQuantizer implements BinaryQuantizer {
        // Same binarization math for all optimizers; only the learned transform differs.
        private final RandomBinaryQuantizer delegate = new RandomBinaryQuantizer();

        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {
            delegate.quantizeBody(vector, mu, residualNorm, quantizedDim, stiefel, outWords);
        }
    }

    // ---------------------------------------------------------------------
    // Quantized vector (per-vector data)
    // ---------------------------------------------------------------------

    public static class QuantizedVector {
        public float scale; // d^{-1/2} ||x_i - μ_i*||_2
        public float offset; // offset_i = <x_i, μ_i*> - ||μ_i*||^2
        public byte landmark; // c_i*, unsigned [0, C)
        public long[] binaryVector; // bin(x̂_i)

        private QuantizedVector(float scale,
                                float offset,
                                byte landmark,
                                long[] binaryVector) {
            this.scale = scale;
            this.offset = offset;
            this.landmark = landmark;
            this.binaryVector = binaryVector;
        }

        public static int wordsForDims(int quantizedDim) {
            return (quantizedDim + 63) >>> 6;
        }

        public static QuantizedVector createEmpty(int quantizedDim) {
            return new QuantizedVector(
                    Float.NaN,
                    Float.NaN,
                    (byte) 0,
                    new long[wordsForDims(quantizedDim)]
            );
        }

        public static int serializedSizeBytes(int quantizedDim) {
            return Float.BYTES + Float.BYTES + Byte.BYTES
                    + wordsForDims(quantizedDim) * Long.BYTES;
        }

        /**
         * Writes only the binary body into {@code dest.binaryVector}.
         *
         * <p>Header fields ({@code scale}, {@code offset}, {@code landmark}) must be
         * set by the caller prior to calling this method.</p>
         *
         * @param residualNorm  ||x - μ||_2, required for Eq. 6 normalization during binarization
         */
        static void quantizeTo(VectorFloat<?> vector,
                               float residualNorm,
                               int quantizedDim,
                               VectorFloat<?> mu,
                               StiefelTransform stiefel,
                               BinaryQuantizer quantizer,
                               QuantizedVector dest) {

            // encodeTo() must have initialized header
            if (Float.isNaN(dest.scale) || Float.isNaN(dest.offset)) {
                throw new IllegalStateException(
                        "QuantizedVector header not initialized before quantizeTo()"
                );
            }

            assert Float.isFinite(dest.scale) : "scale is not finite";
            assert Float.isFinite(dest.offset) : "offset is not finite";

            int words = wordsForDims(quantizedDim);
            if (dest.binaryVector.length < words) {
                throw new IllegalArgumentException("binaryVector too short");
            }

            Arrays.fill(dest.binaryVector, 0, words, 0L);
            quantizer.quantizeBody(
                    vector,
                    mu,
                    residualNorm,
                    quantizedDim,
                    stiefel,
                    dest.binaryVector
            );
        }

        public void write(DataOutput out, int quantizedDim) throws IOException {
            out.writeFloat(scale);
            out.writeFloat(offset);
            out.writeByte(landmark);

            int words = wordsForDims(quantizedDim);
            for (int i = 0; i < words; i++) {
                out.writeLong(binaryVector[i]);
            }
        }

        public static QuantizedVector load(RandomAccessReader in,
                                           int quantizedDim) throws IOException {
            float scale = in.readFloat();
            float offset = in.readFloat();
            byte landmark;
            if (in instanceof io.github.jbellis.jvector.disk.ByteReadable) {
                landmark = ((io.github.jbellis.jvector.disk.ByteReadable) in).readByte();
            } else {
                throw new IOException("ASH requires ByteReadable reader for landmark");
            }

            int words = wordsForDims(quantizedDim);
            long[] body = new long[words];
            for (int i = 0; i < words; i++) {
                body[i] = in.readLong();
            }

            return new QuantizedVector(scale, offset, landmark, body);
        }

        public static void loadInto(RandomAccessReader in,
                                    QuantizedVector dest,
                                    int quantizedDim) throws IOException {
            dest.scale = in.readFloat();
            dest.offset = in.readFloat();
            if (in instanceof io.github.jbellis.jvector.disk.ByteReadable) {
                dest.landmark = ((io.github.jbellis.jvector.disk.ByteReadable) in).readByte();
            } else {
                throw new IOException("ASH requires ByteReadable reader for landmark");
            }

            int words = wordsForDims(quantizedDim);
            for (int i = 0; i < words; i++) {
                dest.binaryVector[i] = in.readLong();
            }
        }

        /**
         * Compares two ASH quantized vectors for exact encoded equality.
         *
         * <p>
         * Equality is defined as bitwise equality of all encoded components:
         * <ul>
         *   <li>Binary code payload</li>
         *   <li>Associated landmark index</li>
         *   <li>Auxiliary scalar values (residual norm, dot-with-landmark)</li>
         * </ul>
         *
         * <p>
         * Floating-point values are compared using their raw bit representations
         * rather than tolerance-based comparisons. This is intentional: quantized
         * vectors are expected to be deterministic outputs of the encoder, and
         * approximate equality would be ambiguous and unsafe for hashing.
         * </p>
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuantizedVector that = (QuantizedVector) o;
            return Float.floatToIntBits(scale) == Float.floatToIntBits(that.scale)
                    && Float.floatToIntBits(offset) == Float.floatToIntBits(that.offset)
                    && landmark == that.landmark
                    && Arrays.equals(binaryVector, that.binaryVector);
        }

        /**
         * Hash code consistent with {@link #equals(Object)}.
         *
         * <p>
         * This hash reflects the full encoded state of the vector and is suitable
         * for use in hash-based collections during testing, debugging, or caching.
         * </p>
         */
        @Override
        public int hashCode() {
            int result = Integer.hashCode(Float.floatToIntBits(scale));
            result = 31 * result + Integer.hashCode(Float.floatToIntBits(offset));
            result = 31 * result + Short.hashCode(landmark);
            result = 31 * result + Arrays.hashCode(binaryVector);
            return result;
        }
    }

    /**
     * Private helper to provide byte storage for binary body
     * @param nBits the number of bits to store
     * @return the byte sequence
     */
    private static ByteSequence<?> createBodyByteSequenceForBits(int nBits) {
        int nBytes = (nBits + 7) >>> 3;
        return vectorTypeSupport.createByteSequence(nBytes);
    }

    /**
     * Orthonormal linear transform on the Stiefel manifold.
     * W  : originalDim × quantizedDim
     * A  : quantizedDim × originalDim, where A = Wᵀ
     * The transform is applied as:
     *     y = A · (x − μ)
     */
    public static final class StiefelTransform {

        /** Forward projection (column-orthonormal). Shape: [originalDim × quantizedDim]. */
        public final RealMatrix W;

        /**
         * Cached row-major backing for A in double precision (A = Wᵀ).
         * Used for training, reference, and correctness-sensitive math.
         *
         * AData must not be mutated after construction.
         * Shape: [quantizedDim × originalDim].
         */
        public final double[][] AData;

        /**
         * Cached row-major backing for A in float precision (A = Wᵀ).
         * Used for encoding and scoring fast paths.
         * AFloat must not be mutated after construction.
         * Shape: [quantizedDim × originalDim].
         */
        public final float[][] AFloat;

        /** Number of rows in A (quantizedDim = d). */
        public final int rows;

        /** Number of columns in A (originalDim = D). */
        public final int cols;

        public StiefelTransform(RealMatrix W) {
            this.W = Objects.requireNonNull(W, "W");

            // Dimensions: W is [D × d]
            final int originalDim = W.getRowDimension();     // D
            final int quantizedDim = W.getColumnDimension(); // d

            this.rows = quantizedDim;
            this.cols = originalDim;

            // Explicitly materialize A = Wᵀ into row-major arrays.
            // We do this directly instead of relying on RealMatrix.transpose()+getData(),
            // because some RealMatrix implementations return views where getData()
            // can be surprisingly expensive or shaped unexpectedly.
            this.AData = new double[quantizedDim][originalDim];
            for (int i = 0; i < quantizedDim; i++) {
                final double[] row = this.AData[i];
                for (int j = 0; j < originalDim; j++) {
                    // A[i][j] = W[j][i]
                    row[j] = W.getEntry(j, i);
                }
            }

            this.AFloat = new float[quantizedDim][originalDim];
            for (int i = 0; i < quantizedDim; i++) {
                final double[] src = this.AData[i];
                final float[] dst = this.AFloat[i];
                for (int j = 0; j < originalDim; j++) {
                    dst[j] = (float) src[j];
                }
            }

            // Hard invariants: these are programmer-contract checks, not user input checks.
            if (AFloat.length != rows) {
                throw new AssertionError("AFloat rows " + AFloat.length + " != quantizedDim " + rows);
            }
            if (rows > 0 && AFloat[0].length != cols) {
                throw new AssertionError("AFloat cols " + AFloat[0].length + " != originalDim " + cols);
            }
            if (AData.length != rows) {
                throw new AssertionError("AData rows " + AData.length + " != quantizedDim " + rows);
            }
            if (rows > 0 && AData[0].length != cols) {
                throw new AssertionError("AData cols " + AData[0].length + " != originalDim " + cols);
            }
        }
    }

    /**
     * RANDOM (untrained) Stiefel initialization.
     * Generates a random orthonormal projection from R^originalDim → R^quantizedDim
     * by sampling a Gaussian matrix and orthonormalizing via QR decomposition.
     * A deterministic sign normalization is applied so results are reproducible
     * across runs and linear algebra backends.
     * @param originalDim dimensionality of the input vectors
     * @param quantizedDim dimensionality of the binary embedding
     * @param rng random number generator
     */
    public static StiefelTransform runWithoutTraining(int originalDim,
                                                      int quantizedDim,
                                                      Random rng) {

        if (quantizedDim <= 0 || quantizedDim > originalDim) {
            throw new IllegalArgumentException(
                    "Invalid quantizedDim=" + quantizedDim +
                            " for originalDim=" + originalDim
            );
        }

        // Sample Gaussian matrix G ∈ R^{originalDim × quantizedDim}
        double[][] gaussian = new double[originalDim][quantizedDim];
        for (int i = 0; i < originalDim; i++) {
            for (int j = 0; j < quantizedDim; j++) {
                gaussian[i][j] = rng.nextGaussian();
            }
        }

        RealMatrix G = MatrixUtils.createRealMatrix(gaussian);

        // QR decomposition: G = Q R
        QRDecomposition qr = new QRDecomposition(G);

        // Commons Math returns full Q (m×m). We need thin Q (m×n).
        RealMatrix Qfull = qr.getQ();  // originalDim × originalDim
        RealMatrix Q = Qfull.getSubMatrix(0, originalDim - 1, 0, quantizedDim - 1); // originalDim × quantizedDim

        RealMatrix R = qr.getR(); // originalDim × quantizedDim

        // Normalize QR sign ambiguity for determinism (RANDOM only)
        int diag = Math.min(R.getRowDimension(), R.getColumnDimension()); // == quantizedDim
        for (int j = 0; j < diag; j++) {
            if (R.getEntry(j, j) < 0.0) {
                for (int i = 0; i < Q.getRowDimension(); i++) {
                    Q.setEntry(i, j, -Q.getEntry(i, j));
                }
            }
        }

        // W := Q (originalDim × quantizedDim)
        return new StiefelTransform(Q);
    }

    @Override
    @Deprecated
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        throw new UnsupportedOperationException("Deprecated createCompressedVectors() method not supported");
    }


    /**
     * Encodes all vectors using ASH, honoring the caller-provided ForkJoinPool.
     *
     * <p><strong>Concurrency model:</strong></p>
     * <ul>
     *   <li>The work is partitioned into a small number of contiguous ordinal ranges ("chunks").</li>
     *   <li>Each chunk is processed sequentially by a single task.</li>
     *   <li>Chunks are executed in parallel using the provided {@code simdExecutor}.</li>
     * </ul>
     *
     * <p>
     * This design intentionally avoids {@code IntStream.parallel()} and other
     * constructs that implicitly use {@link ForkJoinPool#commonPool()}.
     * All parallelism is derived exclusively from the supplied executor,
     * ensuring predictable CPU usage and isolation from unrelated workloads.
     * </p>
     *
     * <p><strong>Correctness guarantees:</strong></p>
     * <ul>
     *   <li>Each vector ordinal {@code i} is written exactly once to {@code out[i]}.</li>
     *   <li>The final output array is ordered by ordinal, independent of execution order.</li>
     *   <li>Missing vectors ({@code null}) are encoded as zero vectors, per the
     *       {@link VectorCompressor} contract.</li>
     *   <li>Safe publication is guaranteed by joining all submitted tasks
     *       before returning.</li>
     * </ul>
     *
     * <p>
     * This implementation favors clarity and correctness over maximal parallel
     * granularity. More aggressive chunking or SIMD-aware partitioning can be
     * layered on later without changing observable semantics.
     * </p>
     */

    @Override
    public ASHVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        final int n = ravv.size();

        // The caller controls parallelism; we derive a bounded number of tasks
        // directly from the executor rather than using parallel streams.
        final int parallelism = Math.max(1, simdExecutor.getParallelism());

        // Each task processes a contiguous range of ordinals.
        // Chunking by contiguous ranges preserves cache locality and
        // guarantees each ordinal is written exactly once.
        final int chunkSize = (n + parallelism - 1) / parallelism;

        final QuantizedVector[] out = new QuantizedVector[n];
        final var ravvSupplier = ravv.threadLocalSupplier();

        // NOTE:
        // We intentionally do NOT use IntStream.parallel() here.
        // Parallel streams always execute on ForkJoinPool.commonPool(),
        // which would ignore the caller-provided executor and risk
        // oversubscribing CPU resources.
        var tasks = new java.util.ArrayList<java.util.concurrent.ForkJoinTask<?>>(parallelism);

        for (int t = 0; t < parallelism; t++) {
            final int start = t * chunkSize;
            final int end = Math.min(n, start + chunkSize);
            if (start >= end) {
                break;
            }

            tasks.add(simdExecutor.submit(() -> {
                // Each task uses its own thread-local RAVV view.
                var localRavv = ravvSupplier.get();

                for (int i = start; i < end; i++) {
                    VectorFloat<?> v = localRavv.getVector(i);

                    // Contract: missing vectors are encoded as zero vectors.
                    if (v != null) {
                        out[i] = encode(v);
                    } else {
                        out[i] = QuantizedVector.createEmpty(quantizedDim);
                    }
                }
            }));
        }

        // Join establishes a happens-before relationship, ensuring safe publication
        // of all writes to the output array.
        for (var task : tasks) {
            task.join();
        }

        return new ASHVectors(this, out);
    }


    /**
     * Encodes the input vector using ASH.
     * @return the header and binary vector payload
     */
    @Override
    public QuantizedVector encode(VectorFloat<?> vector) {
        var qv = QuantizedVector.createEmpty(this.quantizedDim);
        encodeTo(vector, qv);
        return qv;
    }

    @Override
    public void encodeTo(VectorFloat<?> vector, QuantizedVector dest) {
        // NOTE:
        // Single-landmark baseline configuration (Table 2, C = 1).
        // All vectors are encoded relative to landmark μ_c (C=1 => μ_0 is the dataset mean).
        // The landmark id is always 0 in this mode.

        // Landmark assignment: nearest centroid by L2 distance
        byte landmark = 0;
        float bestDist = Float.POSITIVE_INFINITY;

        for (int c = 0; c < landmarkCount; c++) {
            float dist = VectorUtil.squareL2Distance(vector, landmarks[c]);
            if (dist < bestDist) {
                bestDist = dist;
                landmark = (byte) c;
            }
        }

        final VectorFloat<?> mu = landmarks[landmark & 0xFF];

        // Compute <x, μ>
        final float dotXMu = VectorUtil.dotProduct(vector, mu);

        // Compute ||x_i − μ_i*||_2
        // We store the true L2 norm (not squared), matching the paper’s use in Eq. 6 normalization.
        final float sqDist = VectorUtil.squareL2Distance(vector, mu);
        final float residualNorm = (float) Math.sqrt(sqDist);

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
        final float offset = dotXMu - landmarkNormSq[landmark];

        // Write the new header fields into dest (requires QuantizedVector fields renamed, see below)
        dest.scale = scale;
        dest.offset = offset;
        dest.landmark = landmark;

        // Binary body: sign(A · (x − μ)) with Eq. 6 normalization inside quantizer
        quantizeVector(vector, residualNorm, landmark, dest);
    }

    // ---------------------------------------------------------------------
    // Training of projection matrix (ITQ + LANDING algorithms)
    // ---------------------------------------------------------------------

    /**
     * Training inputs derived from a random sample of the dataset:
     * xTrainNorm holds normalized centered residuals.
     *
     * NOTE: Vectors are read as floats; training matrices are accumulated/stored in double precision.
     */
    private static final class TrainingData {
        final double[][] xTrainNorm; // [N_train][D]

        TrainingData(double[][] xTrainNorm) {
            this.xTrainNorm = xTrainNorm;
        }
    }

    /**
     * Builds normalized centered residuals for training:
     *  - Choose N_train = min(N, cap) (cap differs by optimizer).
     *  - Sample ordinals uniformly at random (deterministic RNG).
     *  - For each sampled ordinal i:
     *      c = argmin ||x_i - mu_c||^2  (same rule as encoding)
     *      r = x_i - mu_c
     *      r_hat = r / ||r||
     *      store r_hat into xTrainNorm
     */
    private static TrainingData prepareTrainingData(
            VectorFloat<?>[] points,
            VectorFloat<?>[] landmarks,
            int optimizer,
            int originalDim,
            Random rng
    ) {
        final int N = points.length;
        final int cap = (optimizer == LANDING) ? LANDING_N_TRAIN : ITQ_N_TRAIN;
        final int nTrain = Math.min(N, cap);

        final int[] sample = reservoirSampleOrdinals(N, nTrain, rng);
        Arrays.sort(sample); // deterministic order for reproducibility

        final double[][] xTrainNorm = new double[nTrain][originalDim];

        int row = 0;
        int samplePos = 0;

        for (int i = 0; i < N && samplePos < nTrain; i++) {
            if (sample[samplePos] != i) {
                continue;
            }
            samplePos++;

            final VectorFloat<?> x = points[i];

            // Nearest landmark assignment (same semantics as encodeTo).
            byte best = 0;
            float bestDist = Float.POSITIVE_INFINITY;
            for (int c = 0; c < landmarks.length; c++) {
                float dist = VectorUtil.squareL2Distance(x, landmarks[c]);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = (byte) c;
                }
            }

            final VectorFloat<?> mu = landmarks[best & 0xFF];

            // Center and normalize into double precision row.
            final double[] out = xTrainNorm[row++];

            double ss = 0.0;
            for (int d = 0; d < originalDim; d++) {
                double v = (double) x.get(d) - (double) mu.get(d);
                out[d] = v;
                ss += v * v;
            }

            // Normalize residuals.  NOTE: Does not handle zero vectors!
            double inv = 1.0 / Math.sqrt(ss);
            for (int d = 0; d < originalDim; d++) {
                out[d] *= inv;
            }
        }

        if (row != nTrain) {
            throw new IllegalStateException("Training sample fill mismatch: got " + row + ", expected " + nTrain);
        }

        return new TrainingData(xTrainNorm);
    }

    /**
     * Reservoir sample k ordinals uniformly from [0, N).
     * Deterministic given rng.
     */
    private static int[] reservoirSampleOrdinals(int N, int k, Random rng) {
        if (k < 0 || k > N) throw new IllegalArgumentException("k out of range: " + k + " for N=" + N);
        int[] r = new int[k];
        for (int i = 0; i < k; i++) r[i] = i;
        for (int i = k; i < N; i++) {
            int j = rng.nextInt(i + 1);
            if (j < k) r[j] = i;
        }
        return r;
    }

    /**
     * ITQ trainer:
     *  - PCA basis from SVD(X_train_norm)
     *  - iterative orthogonal Procrustes updates on temp_mat
     *  - returns W = M_pca @ R
     */
    private static StiefelTransform runItqTrainer(
            double[][] xHdNorm,   // [N][D] normalized residuals
            int quantizedDim,     // d
            Random rng,
            int nTrainingIterations
    ) {
        logProgress("\t[stage] Starting runITQTrainer...");

        final int N = xHdNorm.length;
        final int D = xHdNorm[0].length;
        final int d = quantizedDim;

        // PCA basis M = V[:, :d] where X = U S V^T
        RealMatrix X = new Array2DRowRealMatrix(xHdNorm, false); // [N×D]
        logProgress("\t[stage] Starting Native SVD for PCA...");
        // We only need V, so we tell LAPACK to skip U to save massive amounts of time/memory
        RealMatrix V = computeNativeV(X);
        logProgress("\t[stage] Completed Native SVD...");
        RealMatrix M = V.getSubMatrix(0, D - 1, 0, d - 1); // [D×d]

        // Early stopping based on stabilized binary codes (subject to max iterations)
        RealMatrix prevXbin = null;
        int stable = 0;
        final double STOP_FRAC = 4e-3;   // stop when <0.4% bits flip TODO setting
        final int STOP_PATIENCE = 3;     // for 3 consecutive epochs TODO setting

        // X_norm = X_hd_norm @ M (N x d)
        RealMatrix Xnorm = nativeMultiply(X, M); // [N×d]

        // temp_mat ~ N(0,1) in R^{d×d}. Initial random rotation.
        RealMatrix temp = gaussianMatrix(d, d, rng);

        // Loop: epoch 0..iters inclusive; last epoch computes R but does not update temp_mat.
        RealMatrix R = null;
        logProgress("\t[stage] ITQ training iterations started...");
        for (int epoch = 0; epoch <= nTrainingIterations; epoch++) {
            R = orthogonalize(temp); // R = U @ V^T

            if (epoch < nTrainingIterations) {
                // Use native multiplication for Xtr = Xnorm @ R
                RealMatrix Xtr = nativeMultiply(Xnorm, R);             // [N×d]
                RealMatrix Xbin = sign01(Xtr);                         // [N×d] in {+1,-1}

                if (prevXbin != null) {
                    long changed = 0;
                    long total = (long) N * (long) d;
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < d; j++) {
                            if (Xbin.getEntry(i, j) != prevXbin.getEntry(i, j)) changed++;
                        }
                    }
                    double frac = (double) changed / (double) total;

                    logProgress(String.format(java.util.Locale.ROOT,
                            "\titeration %d/%d - fracBitsChanged=%.6g (stable=%d/%d)",
                            epoch, nTrainingIterations, frac, stable, STOP_PATIENCE));

                    if (frac < STOP_FRAC) {
                        if (++stable >= STOP_PATIENCE) break;
                    } else {
                        stable = 0;
                    }
                }

                // Optimized temp = Xnorm.T @ Xbin
                // This avoids the physical transpose of the large Xnorm matrix
                prevXbin = Xbin;
                temp = nativeMultiplyTransposedA(Xnorm, Xbin);          // [d×d]
            }
        }
        logProgress("\t[stage] ITQ training completed.");

        // W = M @ R  (shape [D×d])
        // Use native multiply for the final transform matrix
        RealMatrix W = nativeMultiply(M, R);
        return new StiefelTransform(W);
    }

    /**
     * LANDING trainer TODO incomplete
     *
     * NOTE: This is a DRAFT (not hardened) implementation using RealMatrix operations.
     * We can SIMD/block-optimize the hot math once correctness is validated.
     */
    private static StiefelTransform runLandingTrainer(
            double[][] xHdNorm,   // [N][D] normalized residuals
            int quantizedDim,     // d
            Random rng,
            int nTrainingIterations
    ) {
        final int N = xHdNorm.length;
        final int D = xHdNorm[0].length;
        final int d = quantizedDim;

        RealMatrix X = new Array2DRowRealMatrix(xHdNorm, false); // [N×D]

        // PCA basis M_pca = V[:, :d]
        SingularValueDecomposition svdX = new SingularValueDecomposition(X);
        RealMatrix V = svdX.getV();
        RealMatrix M_pca = V.getSubMatrix(0, D - 1, 0, d - 1); // [D×d]

        // Init:
        // temp(d×d) -> Q = U V^T
        // A = Q @ M_pca^T  (d×D)
        // W = A^T          (D×d)
        RealMatrix Q = orthogonalize(gaussianMatrix(d, d, rng));
        RealMatrix A = Q.multiply(M_pca.transpose()); // [d×D]
        RealMatrix W = A.transpose();                 // [D×d]

        final RealMatrix I_D = MatrixUtils.createRealIdentityMatrix(D);
        final double invSqrtD = 1.0 / Math.sqrt(d);

        int[] idx = new int[N];
        for (int i = 0; i < N; i++) idx[i] = i;

        for (int epoch = 0; epoch < nTrainingIterations; epoch++) {
            shuffleInPlace(idx, rng);

            for (int off = 0; off < N; off += LANDING_BATCH_SIZE) {
                final int end = Math.min(N, off + LANDING_BATCH_SIZE);
                final int B = end - off;

                // Build batch matrix X_batch (B×D)
                double[][] xb = new double[B][D];
                for (int bi = 0; bi < B; bi++) {
                    xb[bi] = xHdNorm[idx[off + bi]];
                }
                RealMatrix Xb = new Array2DRowRealMatrix(xb, false);

                // X_sign = sign(Xb @ W) / sqrt(d)
                RealMatrix Xproj = Xb.multiply(W);                        // [B×d]
                RealMatrix Xsign = sign01(Xproj).scalarMultiply(invSqrtD);// [B×d]

                // gradA = (X_sign^T X_sign) A - (X_sign^T X_batch)
                RealMatrix XtX = Xsign.transpose().multiply(Xsign);       // [d×d]
                RealMatrix XtB = Xsign.transpose().multiply(Xb);          // [d×D]
                RealMatrix gradA = XtX.multiply(A).subtract(XtB);         // [d×D]

                // gradW = (X_batch^T X_batch) ( W (A A^T) - A^T )
                RealMatrix G = Xb.transpose().multiply(Xb);               // [D×D]
                RealMatrix AAT = A.multiply(A.transpose());               // [d×d]
                RealMatrix term = W.multiply(AAT).subtract(A.transpose());// [D×d]
                RealMatrix gradW = G.multiply(term);                      // [D×d]

                double lr = 0.1 * Math.pow(0.98, epoch);
                double lambda = 1.0;

                // skew = 0.5 * (A^T gradA - (A^T gradA)^T)
                RealMatrix Mmat = A.transpose().multiply(gradA);          // [D×D]
                RealMatrix skew = Mmat.subtract(Mmat.transpose()).scalarMultiply(0.5);

                // A -= lr * A @ ( lambda*(A^T A - I) + skew )
                RealMatrix ortho = A.transpose().multiply(A).subtract(I_D);
                RealMatrix updateCore = ortho.scalarMultiply(lambda).add(skew);
                A = A.subtract(A.multiply(updateCore).scalarMultiply(lr));

                // W -= lr * gradW
                W = W.subtract(gradW.scalarMultiply(lr));
            }
        }

        return new StiefelTransform(W);
    }

    private static void shuffleInPlace(int[] a, Random rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    private static RealMatrix gaussianMatrix(int rows, int cols, Random rng) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] r = m[i];
            for (int j = 0; j < cols; j++) {
                r[j] = rng.nextGaussian();
            }
        }
        return new Array2DRowRealMatrix(m, false);
    }

    /** Orthonormal factor: U @ V^T from SVD(M). */
//    private static RealMatrix orthogonalize(RealMatrix m) {
//        SingularValueDecomposition svd = new SingularValueDecomposition(m);
//        return svd.getU().multiply(svd.getVT());
//    }

    // Basic dgemm routine.
    private static RealMatrix nativeMultiply(RealMatrix A, RealMatrix B) {
        int m = A.getRowDimension();
        int n = B.getColumnDimension();
        int k = A.getColumnDimension(); // Must match B's row dimension

        // 1. Convert to 1D Column-Major for BLAS
        double[] aArr = serializeColumnMajor(A);
        double[] bArr = serializeColumnMajor(B);
        double[] cArr = new double[m * n];

        // 2. Call Native DGEMM: C = alpha*A*B + beta*C
        // "N", "N" means 'No Transpose' for both A and B
        BLAS.getInstance().dgemm(
                "N", "N",
                m, n, k,
                1.0,    // alpha
                aArr, m, // A and its leading dimension
                bArr, k, // B and its leading dimension
                0.0,    // beta (ignores existing values in cArr)
                cArr, m  // C and its leading dimension
        );

        // 3. Convert back to RealMatrix
        return new Array2DRowRealMatrix(deserializeColumnMajor(cArr, m, n), false);
    }

    // Uses the dgesvd routine but tells LAPACK to skip the U matrix (for PCA).  Saves time....
    private static RealMatrix computeNativeV(RealMatrix x) {
        int rows = x.getRowDimension();
        int cols = x.getColumnDimension();
        double[] a = serializeColumnMajor(x);

        double[] s = new double[Math.min(rows, cols)];
        double[] vt = new double[cols * cols]; // Right singular vectors
        intW info = new intW(0);

        // Query optimal workspace size (standard LAPACK practice)
        double[] workQuery = new double[1];
        LAPACK.getInstance().dgesvd("N", "A", rows, cols, a, rows, s, null, 1, vt, cols, workQuery, -1, info);

        int lwork = (int) workQuery[0];
        double[] work = new double[lwork];

        // Actual call: "N" = No U matrix, "A" = All columns of V^T
        LAPACK.getInstance().dgesvd("N", "A", rows, cols, a, rows, s, null, 1, vt, cols, work, lwork, info);

        if (info.val != 0) {
            throw new RuntimeException("Native PCA SVD failed with info=" + info.val);
        }

        // Convert VT (column-major) back to RealMatrix
        return new Array2DRowRealMatrix(deserializeColumnMajor(vt, cols, cols), false);
    }

    /** Computes A^T * B using Native BLAS dgemm */
    private static RealMatrix nativeMultiplyTransposedA(RealMatrix A, RealMatrix B) {
        int m = A.getColumnDimension(); // A.T has rows = A.cols
        int n = B.getColumnDimension();
        int k = A.getRowDimension();    // A.T has cols = A.rows

        double[] aArr = serializeColumnMajor(A);
        double[] bArr = serializeColumnMajor(B);
        double[] cArr = new double[m * n];

        // "T" = Transpose Matrix A, "N" = No Transpose Matrix B
        BLAS.getInstance().dgemm(
                "T", "N",
                m, n, k,
                1.0,
                aArr, k, // Leading dimension is k because A is currently [k x m]
                bArr, k,
                0.0,
                cArr, m
        );

        return new Array2DRowRealMatrix(deserializeColumnMajor(cArr, m, n), false);
    }

    private static double calculateMSE(RealMatrix a, RealMatrix b) {
        double sumSq = 0;
        int rows = a.getRowDimension();
        int cols = a.getColumnDimension();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = a.getEntry(i, j) - b.getEntry(i, j);
                sumSq += diff * diff;
            }
        }
        return sumSq / (rows * cols);
    }

    private static RealMatrix orthogonalize(RealMatrix m) {
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();

        // Convert to column-major for Native LAPACK
        double[] a = serializeColumnMajor(m);

        // Prepare output buffers
        double[] u = new double[rows * rows];
        double[] vt = new double[cols * cols];
        double[] s = new double[Math.min(rows, cols)];

        // int wrapper required....
        intW info = new intW(0);

        // LAPACK Workspace (Required for dgesvd): Pass -1 for lwork and a 1-element array for work
        double[] workQuery = new double[1];
        LAPACK.getInstance().dgesvd(
                "A", "A", rows, cols, a, rows, s, u, rows, vt, cols,
                workQuery, -1, info
        );

        // Extract the optimal lwork and allocate the real workspace
        int lwork = (int) workQuery[0];
        double[] work = new double[lwork];

        // Native SVD call
        LAPACK.getInstance().dgesvd(
                "A", "A",     // Compute all columns of U and all rows of VT
                rows, cols,   // Matrix dimensions
                a, rows,      // Input matrix and its leading dimension
                s,            // Output singular values
                u, rows,      // Output U and its leading dimension
                vt, cols,     // Output VT and its leading dimension
                work, lwork,  // Workspace
                info          // Success/Fail code
        );

        if (info.val != 0) {
            throw new RuntimeException("LAPACK dgesvd failed with info=" + info.val);
        }

        // Convert back to Commons Math
        RealMatrix U = new Array2DRowRealMatrix(deserializeColumnMajor(u, rows, rows), false);
        RealMatrix VT = new Array2DRowRealMatrix(deserializeColumnMajor(vt, cols, cols), false);

        return U.multiply(VT);
    }

    /** Converts an Apache Commons RealMatrix to a column-major 1D array for LAPACK. */
    private static double[] serializeColumnMajor(RealMatrix m) {
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();
        double[] data = new double[rows * cols];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                data[c * rows + r] = m.getEntry(r, c);
            }
        }
        return data;
    }

    /** Converts a column-major 1D array back into a row-major double[][] for Commons Math. */
    private static double[][] deserializeColumnMajor(double[] data, int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                matrix[r][c] = data[c * rows + r];
            }
        }
        return matrix;
    }

    /**
     * Elementwise sign producing +/-1 (no zeros), consistent with ASH encoding convention:
     * bit=1 iff projection > 0, else bit=0.
     */
    private static RealMatrix sign01(RealMatrix m) {
        final int r = m.getRowDimension();
        final int c = m.getColumnDimension();
        double[][] out = new double[r][c];
        for (int i = 0; i < r; i++) {
            double[] row = out[i];
            for (int j = 0; j < c; j++) {
                row[j] = (m.getEntry(i, j) > 0.0) ? 1.0 : -1.0;
            }
        }
        return new Array2DRowRealMatrix(out, false);
    }

    @Override
    public int compressorSize() {
        // Must exactly match write(out, version)
        int size = 0;
        size += Integer.BYTES; // MAGIC
        size += Integer.BYTES; // version
        size += Integer.BYTES; // originalDimension
        size += Integer.BYTES; // encodedBits
        size += Integer.BYTES; // quantizedDim
        size += Integer.BYTES; // optimizer
        size += Integer.BYTES; // landmarkCount
        for (int c = 0; c < landmarkCount; c++) {
            size += Integer.BYTES; // landmark dimension
            size += Float.BYTES * landmarks[c].length();
        }
        return size;
    }

    @Override
    public int compressedVectorSize() {
        int words = QuantizedVector.wordsForDims(quantizedDim);
        return Float.BYTES           // scale
                + Float.BYTES           // offset
                + Byte.BYTES           // landmark
                + words * Long.BYTES;   // binary payload
    }

    @Override
    public double reconstructionError(VectorFloat<?> vector) {
        throw new UnsupportedOperationException("Reconstruction error not defined for ASH");
    }

    @Override
    public long ramBytesUsed() {
        long bytes = 0L;

        if (landmarks != null) {
            for (int i = 0; i < landmarks.length; i++) {
                bytes += (landmarks[i] == null) ? 0L : landmarks[i].ramBytesUsed();
            }
        }

        if (stiefelTransform != null) {
            // Authoritative double-precision model
            bytes += estimateMatrixBytes(stiefelTransform.W);

            // Materialized compute buffers (row-major)
            bytes += estimateArray2DBytes(stiefelTransform.AData);
            bytes += estimateArray2DBytes(stiefelTransform.AFloat);
        }

        return bytes;
    }

    private static long estimateArray2DBytes(double[][] a) {
        if (a == null) return 0L;
        long bytes = 0L;
        for (double[] row : a) {
            bytes += RamUsageEstimator.sizeOf(row);
        }
        return bytes;
    }

    private static long estimateArray2DBytes(float[][] a) {
        if (a == null) return 0L;
        long bytes = 0L;
        for (float[] row : a) {
            bytes += RamUsageEstimator.sizeOf(row);
        }
        return bytes;
    }

    private static long estimateMatrixBytes(RealMatrix m) {
        if (m == null) return 0L;
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();
        return (long) rows * cols * Double.BYTES
                + (long) rows * Long.BYTES;
    }

    // Exact, element-wise comparison of VectorFloat contents.
    // We use Float.floatToIntBits to handle -0.0f, NaN, and ensure
    // stable, bitwise equality suitable for hashing.
    private static boolean vectorEquals(VectorFloat<?> a, VectorFloat<?> b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a.length() != b.length()) return false;

        for (int i = 0; i < a.length(); i++) {
            if (Float.floatToIntBits(a.get(i)) != Float.floatToIntBits(b.get(i))) {
                return false;
            }
        }
        return true;
    }

    // Hash based on the raw bit representation of each float.
    // This mirrors vectorEquals and guarantees hash consistency.
    private static int vectorHash(VectorFloat<?> v) {
        if (v == null) return 0;
        int h = 1;
        for (int i = 0; i < v.length(); i++) {
            h = 31 * h + Integer.hashCode(Float.floatToIntBits(v.get(i)));
        }
        return h;
    }

    // Exact comparison of two RealMatrix instances by dimensions and entries.
    // We intentionally avoid RealMatrix.equals() and tolerance-based checks.
    // ASH initialization is deterministic, so bitwise equality is expected
    // and required for robust hashing semantics.
    private static boolean matrixEquals(RealMatrix a, RealMatrix b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a.getRowDimension() != b.getRowDimension()
                || a.getColumnDimension() != b.getColumnDimension()) {
            return false;
        }

        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                if (Double.doubleToLongBits(a.getEntry(i, j))
                        != Double.doubleToLongBits(b.getEntry(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    // Hash code derived from matrix dimensions and raw double bits.
    // Only the forward projection matrix (W) is hashed; the transpose (A)
    // is derived and does not need to be hashed separately.
    private static int matrixHash(RealMatrix m) {
        if (m == null) return 0;
        int h = 1;
        h = 31 * h + m.getRowDimension();
        h = 31 * h + m.getColumnDimension();

        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                long bits = Double.doubleToLongBits(m.getEntry(i, j));
                h = 31 * h + (int) (bits ^ (bits >>> 32));
            }
        }
        return h;
    }

    /**
     * Compares this ASH compressor to another for encoding equivalence.
     *
     * <p>
     * Two compressors are equal if they would produce identical encoded outputs
     * (including binary codes and auxiliary scalars) for the same input vectors.
     * </p>
     *
     * <p>
     * Equality therefore requires:
     * <ul>
     *   <li>Matching dimensional parameters</li>
     *   <li>Identical optimizer selection</li>
     *   <li>Bitwise-equal landmark vectors</li>
     *   <li>Bitwise-equal Stiefel transform matrices</li>
     * </ul>
     *
     * <p>
     * Note that this is a <em>strong</em> notion of equality. We deliberately
     * avoid approximate floating-point comparisons because ASH initialization
     * and training are deterministic, and weaker equality would break hashing
     * and cache correctness.
     * </p>
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        AsymmetricHashing that = (AsymmetricHashing) o;
        return originalDimension == that.originalDimension
                && encodedBits == that.encodedBits
                && quantizedDim == that.quantizedDim
                && optimizer == that.optimizer
                && landmarksEqual(this.landmarks, that.landmarks)
                && matrixEquals(
                this.stiefelTransform == null ? null : this.stiefelTransform.W,
                that.stiefelTransform == null ? null : that.stiefelTransform.W
        );
    }

    private static boolean landmarksEqual(VectorFloat<?>[] a, VectorFloat<?>[] b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (!vectorEquals(a[i], b[i])) return false;
        }
        return true;
    }

    /**
     * Hash code consistent with {@link #equals(Object)}.
     *
     * <p>
     * The hash incorporates all parameters and learned state that affect encoding,
     * including the contents of landmarks and Stiefel transform. This ensures
     * safe use of ASH compressors as map keys or cache entries.
     * </p>
     */
    @Override
    public int hashCode() {
        int result = Integer.hashCode(originalDimension);
        result = 31 * result + Integer.hashCode(encodedBits);
        result = 31 * result + Integer.hashCode(quantizedDim);
        result = 31 * result + Integer.hashCode(optimizer);
        result = 31 * result + landmarksHash(landmarks);
        result = 31 * result + matrixHash(
                stiefelTransform == null ? null : stiefelTransform.W
        );
        return result;
    }

    private static int landmarksHash(VectorFloat<?>[] a) {
        if (a == null) return 0;
        int h = 1;
        for (VectorFloat<?> v : a) {
            h = 31 * h + vectorHash(v);
        }
        return h;
    }

    private static void logProgress(String msg) {
        System.out.println(msg);
        System.out.flush();
    }

    @Override
    public String toString() {
        String optimizerName;
        switch (optimizer) {
            case RANDOM:
                optimizerName = "RANDOM";
                break;
            case ITQ:
                optimizerName = "ITQ";
                break;
            case LANDING:
                optimizerName = "LANDING";
                break;
            default:
                optimizerName = "UNKNOWN(" + optimizer + ")";
        }

        return String.format(
                "AsymmetricHashing[origDim=%d, encodedBits=%d, quantizedDim=%d, optimizer=%s, landmarks=%d]",
                originalDimension,
                encodedBits,
                quantizedDim,
                optimizerName,
                landmarkCount
        );
    }
}
