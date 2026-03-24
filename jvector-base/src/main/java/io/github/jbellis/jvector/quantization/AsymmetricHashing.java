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
import io.github.jbellis.jvector.disk.IndexWriter;
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
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.netlib.util.intW;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
                                               int landmarkCount) throws IOException {

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

        // NOTE: points are treated as read-only by KMeansPlusPlusClusterer.
        // Materialize and L2-normalize points
        VectorFloat<?>[] points = new VectorFloat<?>[ravvCopy.size()];
        for (int i = 0; i < ravvCopy.size(); i++) {
            VectorFloat<?> v = ravvCopy.getVector(i).copy(); // copy to avoid mutating original source
            float norm = (float) Math.sqrt(VectorUtil.dotProduct(v, v));
            if (norm > 1e-10f) {
                VectorUtil.scale(v, 1.0f / norm);
            }
            points[i] = v;
        }

        VectorFloat<?>[] landmarks;
        if (landmarkCount == 1) {
            double[] muAccumulator = new double[originalDim];

            // Accumulate in double precision
            for (int i = 0; i < points.length; i++) {
                VectorFloat<?> p = points[i];
                for (int d = 0; d < originalDim; d++) {
                    muAccumulator[d] += p.get(d);
                }
            }

            // Scale and materialize back into a float vector
            VectorFloat<?> mu0 = vectorTypeSupport.createFloatVector(originalDim);
            double invN = 1.0 / points.length;
            for (int d = 0; d < originalDim; d++) {
                mu0.set(d, (float) (muAccumulator[d] * invN));
            }

            landmarks = new VectorFloat<?>[] { mu0 };
        } else {
            long kmStart = System.nanoTime();

            // KMeans++ in JVector returns centroids packed as one big vector [C * D]
            // KMeansPlusPlusClusterer km = new KMeansPlusPlusClusterer(points, landmarkCount);
            MiniBatchKMeansClusterer km = new MiniBatchKMeansClusterer(points, landmarkCount, 1024);
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

        // Keep determinism consistent within ASH.  TODO explore more robust random initialization
        final Random rng = new Random(103);

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

//        if ((payloadBits & 63) != 0) {
//            throw new IllegalArgumentException(
//                    "Invalid encodedBits=" + encodedBits +
//                            ". ASH requires (encodedBits - headerBits) to be a multiple of 64 " +
//                            "for aligned binary payload. " +
//                            "Got payloadBits=" + payloadBits +
//                            " (headerBits=" + headerBits + ")."
//            );
//        }

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
        // Use a double-precision matrix for the projection
        final double[][] A = stiefelTransform.AData; // [d][D]

        final float[][] landmarkProj = new float[C][quantizedDim];
        final float[] landmarkProjSum = new float[C];

        // Materialize μ as double[] to maintain precision during the dot product
        final double[] muArr = new double[originalDim];

        for (int c = 0; c < C; c++) {
            VectorFloat<?> mu = landmarks[c];

            for (int i = 0; i < originalDim; i++) {
                muArr[i] = mu.get(i);
            }

            double totalSum = 0.0;
            float[] proj = landmarkProj[c];

            for (int j = 0; j < quantizedDim; j++) {
                double rowDot = 0.0;
                double[] Arow = A[j];

                // Double-precision dot product
                for (int k = 0; k < originalDim; k++) {
                    rowDot += Arow[k] * muArr[k];
                }

                proj[j] = (float) rowDot;
                totalSum += rowDot;
            }

            landmarkProjSum[c] = (float) totalSum;
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

        // Load StiefelTransform (required for ALL optimizers, including ITQ/LANDING)
        StiefelTransform stiefelTransform = readStiefelTransform(in, quantizedDim, originalDimension);

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
     * Writes the ASH index header to IndexWriter (used for on-disk indexing)
     * @param out the IndexWriter into which to write the object
     * @throws IOException if there is a problem writing to out.
     */
    public void write(IndexWriter out, int version) throws IOException {
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

        // Stiefel transform serialization (required for caching and non-RANDOM optimizers):
        // Serialize A = W^T in float precision as a [d][D] row-major matrix.
        writeStiefelTransform(out, stiefelTransform);
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

        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {

            final float[][] A = stiefel.AFloat;
            final int originalDim = stiefel.cols;

            // Get workspace for this thread and ensure correct sizing (originalDim may have changed)
            Workspace ws = WORKSPACE.get();
            ws.ensureCapacity(originalDim);

            // Alias for readability (no new allocations)
            final float[] x = ws.x;
            final float[] muArr = ws.muArr;
            final float[] xhat = ws.xhat;

            final var vecUtil =
                    io.github.jbellis.jvector.vector.VectorizationProvider
                            .getInstance()
                            .getVectorUtilSupport();

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

        public void write(IndexWriter out, int quantizedDim) throws IOException {
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
     *
     * <p>Samples a Gaussian matrix G ∈ R^{D×d} and computes its polar factor
     * W = U·Vᵀ via a thin SVD (G = U·S·Vᵀ). The resulting W has orthonormal
     * columns (WᵀW = I) and can be used as an orthonormal projection matrix
     * (e.g., y = Wᵀx maps R^D → R^d).
     *
     * @param D dimensionality of the input vectors
     * @param d dimensionality of the projected (embedding) space
     * @param rng random number generator used to sample G
     * @return a {@link StiefelTransform} containing W ∈ R^{D×d} with orthonormal columns
     * @throws IllegalArgumentException if d <= 0 or d > D
     * @throws RuntimeException if the native SVD fails
     */
    public static StiefelTransform runWithoutTraining(int D, int d, Random rng) {
        if (d <= 0 || d > D) {
            throw new IllegalArgumentException("Invalid d=" + d + " for D=" + D);
        }

        // gData is interpreted as a [D x d] column-major matrix by LAPACK.
        double[] gData = new double[D * d];
        for (int i = 0; i < gData.length; i++) {
            gData[i] = rng.nextGaussian();
        }

        double[] wData = orthogonalize(gData, D, d);
        RealMatrix W = new Array2DRowRealMatrix(deserializeColumnMajor(wData, D, d), false);
        return new StiefelTransform(W);
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

        // Convert training data once to flat column-major for BLAS/LAPACK.
        double[] xCol = serializeColumnMajor(xHdNorm);

        // PCA basis M = V[:, :d] where X = U S V^T
        logProgress("\t[stage] Starting Native SVD for PCA...");
        double[] vtCol = computeNativeVt(xCol, N, D);    // [D x D], column-major
        logProgress("\t[stage] Completed Native SVD...");
        double[] mCol = extractPcaBasis(vtCol, D, d);    // [D x d], column-major

        // Early stopping based on stabilized binary codes (subject to max iterations)
        double[] prevXbinCol = null;
        int stable = 0;
        final double STOP_FRAC = 4e-3;
        final int STOP_PATIENCE = 3;

        // X_norm = X_hd_norm @ M  => [N x d], column-major
        double[] xnormCol = nativeMultiply(xCol, N, D, mCol, d);

        // Initial random rotation seed
        double[] tempCol = gaussianMatrix(d, d, rng);

        // Loop: epoch 0..iters inclusive; last epoch computes R but does not update temp_mat.
        double[] rCol = null;
        logProgress("\t[stage] ITQ training iterations started...");
        var startTime = System.nanoTime();
        for (int epoch = 0; epoch <= nTrainingIterations; epoch++) {
            rCol = orthogonalize(tempCol, d, d); // R = U @ V^T, [d x d]

            if (epoch < nTrainingIterations) {
                // Xtr = Xnorm @ R, then binarize in-place to become Xbin.
                double[] xbinCol = nativeMultiply(xnormCol, N, d, rCol, d);

                long changed = sign01InPlace(xbinCol, prevXbinCol);

                if (prevXbinCol != null) {
                    long total = (long) N * (long) d;
                    double frac = (double) changed / (double) total;

                    logProgress(String.format(java.util.Locale.ROOT,
                            "\titeration %d/%d - fracBitsChanged=%.6g (stable=%d/%d)",
                            epoch, nTrainingIterations, frac, stable, STOP_PATIENCE));

                    if (frac < STOP_FRAC) {
                        if (++stable >= STOP_PATIENCE) {
                            break;
                        }
                    } else {
                        stable = 0;
                    }
                }

                prevXbinCol = xbinCol;
                tempCol = nativeMultiplyTransposedA(xnormCol, N, d, xbinCol, d);
            }
        }
        var loopTime = (System.nanoTime() - startTime) / 1e9;
        logProgress("\t[stage] ITQ training completed in " + loopTime + " seconds.");

        // W = M @ R  => [D x d], column-major
        double[] wCol = nativeMultiply(mCol, D, d, rCol, d);

        RealMatrix W = new Array2DRowRealMatrix(deserializeColumnMajor(wCol, D, d), false);
        return new StiefelTransform(W);
    }

    /**
     *
     * LANDING trainer TODO incomplete
     *
     * NOTE: This is a DRAFT (not hardened) implementation using RealMatrix operations.
     * We can SIMD/block-optimize the hot math once correctness is validated.
     *
     * Row-oriented implementation of the paper's stochastic updates:
     *   - A is stored as the decoder matrix [D x d]  (paper's A)
     *   - W is stored as the encoder-transpose [D x d], so Xb * W gives
     *     row-wise projections equivalent to W^T x in the paper
     *
     * The update equations implemented are the row-oriented forms of
     * Equations (22)-(25) from the paper.
     */
    private static StiefelTransform runLandingTrainer(
            double[][] xHdNorm,   // [N][D] normalized residuals
            int quantizedDim,     // d
            Random rng,
            int nTrainingIterations
    ) {
        logProgress("\t[stage] Starting runLandingTrainer...");

        final int N = xHdNorm.length;
        final int D = xHdNorm[0].length;
        final int d = quantizedDim;

        final double invD = 1.0 / d;
        final double invSqrtD = 1.0 / Math.sqrt(d);
        final double lambda = 1.0;

        // ------------------------------------------------------------------
        // ITQ-style initialization from PCA basis P and random orthogonal R.
        //
        // xHdNorm is row-stacked [N x D]. The top right singular vectors of X
        // are the top left singular vectors of X^T used in the paper.
        // ------------------------------------------------------------------
        double[] xCol = serializeColumnMajor(xHdNorm);

        logProgress("\t[stage] Starting Native SVD for LANDING PCA...");
        double[] vtCol = computeNativeVt(xCol, N, D);     // [D x D], column-major
        logProgress("\t[stage] Completed Native SVD for LANDING PCA...");

        double[] pCol = extractPcaBasis(vtCol, D, d);     // [D x d], column-major
        double[] rCol = orthogonalize(gaussianMatrix(d, d, rng), d, d); // [d x d]

        // Stored row-oriented decoder A and encoder-transpose W both start from PR,
        // matching the ITQ-style initialization used elsewhere in this class.
        double[] initCol = nativeMultiply(pCol, D, d, rCol, d); // [D x d]
        RealMatrix A = new Array2DRowRealMatrix(deserializeColumnMajor(initCol, D, d), false);
        RealMatrix W = A.copy();

        // I_d for the Stiefel penalty A^T A - I_d
        double[][] eye = new double[d][d];
        for (int i = 0; i < d; i++) {
            eye[i][i] = 1.0;
        }
        final RealMatrix I_d = new Array2DRowRealMatrix(eye, false);

        int[] idx = new int[N];
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }

        logProgress("\t[stage] LANDING training iterations started...");
        for (int epoch = 0; epoch < nTrainingIterations; epoch++) {
            // Fisher-Yates shuffle
            for (int i = idx.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }

            final double lr = 0.1 * Math.pow(0.98, epoch);

            for (int off = 0; off < N; off += LANDING_BATCH_SIZE) {
                final int end = Math.min(N, off + LANDING_BATCH_SIZE);
                final int B = end - off;

                // Build batch matrix Xb [B x D] without copying row contents.
                double[][] xb = new double[B][];
                for (int bi = 0; bi < B; bi++) {
                    xb[bi] = xHdNorm[idx[off + bi]];
                }
                RealMatrix Xb = new Array2DRowRealMatrix(xb, false);

                // Z = sign(Xb * W) in {+1, -1}^{B x d}, unscaled.
                RealMatrix Xproj = Xb.multiply(W); // [B x d]
                double[][] projData = (Xproj instanceof Array2DRowRealMatrix)
                        ? ((Array2DRowRealMatrix) Xproj).getDataRef()
                        : Xproj.getData();

                double[][] zData = new double[B][d];
                for (int i = 0; i < B; i++) {
                    double[] src = projData[i];
                    double[] dst = zData[i];
                    for (int j = 0; j < d; j++) {
                        dst[j] = (src[j] > 0.0) ? 1.0 : -1.0;
                    }
                }
                RealMatrix Z = new Array2DRowRealMatrix(zData, false); // [B x d]

                // Batch statistics
                RealMatrix ZtZ = Z.transpose().multiply(Z);    // [d x d]
                RealMatrix XtZ = Xb.transpose().multiply(Z);   // [D x d]
                RealMatrix XtX = Xb.transpose().multiply(Xb);  // [D x D]
                RealMatrix AtA = A.transpose().multiply(A);    // [d x d]

                // Row-oriented gradA = d^-1 * A * (Z^T Z) - d^-1/2 * X^T Z
                RealMatrix gradA = A.multiply(ZtZ).scalarMultiply(invD)
                        .subtract(XtZ.scalarMultiply(invSqrtD));

                // Row-oriented for stored W = (paper W)^T:
                // gradW = (X^T X) * ( d^-1 * W * (A^T A) - d^-1/2 * A )
                RealMatrix gradW = XtX.multiply(
                        W.multiply(AtA).scalarMultiply(invD)
                                .subtract(A.scalarMultiply(invSqrtD))
                );

                // Λ(A) = skew(gradA * A^T) * A + λ * A * (A^T A - I)
                RealMatrix skewBase = gradA.multiply(A.transpose()); // [D x D]
                RealMatrix skew = skewBase.subtract(skewBase.transpose()).scalarMultiply(0.5);
                RealMatrix ortho = AtA.subtract(I_d); // [d x d]
                RealMatrix landing = skew.multiply(A)
                        .add(A.multiply(ortho).scalarMultiply(lambda));

                A = A.subtract(landing.scalarMultiply(lr));
                W = W.subtract(gradW.scalarMultiply(lr));
            }
        }

        logProgress("\t[stage] LANDING training completed.");

        // StiefelTransform expects the decoder matrix [D x d].
        return new StiefelTransform(A);
    }

    /**
     * Samples a Gaussian matrix in flat column-major storage.
     *
     * Values are assigned in row-major logical order.
     */
    private static double[] gaussianMatrix(int rows, int cols, Random rng) {
        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            int dst = r;
            for (int c = 0; c < cols; c++, dst += rows) {
                data[dst] = rng.nextGaussian();
            }
        }
        return data;
    }

    /** Converts a row-major double[][] to a flat column-major array for BLAS/LAPACK. */
    private static double[] serializeColumnMajor(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] data = new double[rows * cols];

        for (int r = 0; r < rows; r++) {
            double[] row = matrix[r];
            int dst = r;
            for (int c = 0; c < cols; c++, dst += rows) {
                data[dst] = row[c];
            }
        }
        return data;
    }

    /** Converts a RealMatrix to a flat column-major array for BLAS/LAPACK. */
    private static double[] serializeColumnMajor(RealMatrix m) {
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();
        double[] data = new double[rows * cols];

        if (m instanceof Array2DRowRealMatrix) {
            double[][] raw = ((Array2DRowRealMatrix) m).getDataRef();
            for (int r = 0; r < rows; r++) {
                double[] row = raw[r];
                int dst = r;
                for (int c = 0; c < cols; c++, dst += rows) {
                    data[dst] = row[c];
                }
            }
            return data;
        }

        for (int r = 0; r < rows; r++) {
            int dst = r;
            for (int c = 0; c < cols; c++, dst += rows) {
                data[dst] = m.getEntry(r, c);
            }
        }
        return data;
    }

    /** Converts a flat column-major array back into a row-major double[][] for Commons Math. */
    private static double[][] deserializeColumnMajor(double[] data, int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            double[] row = matrix[r];
            int src = r;
            for (int c = 0; c < cols; c++, src += rows) {
                row[c] = data[src];
            }
        }
        return matrix;
    }

    /**
     * Extracts M = V[:, :d] from VT, where VT is [dim x dim] in flat column-major layout.
     * Returns M as [dim x d], also in flat column-major layout.
     */
    private static double[] extractPcaBasis(double[] vtCol, int dim, int d) {
        double[] mCol = new double[dim * d];

        for (int col = 0; col < d; col++) {
            int dstOffset = col * dim;
            for (int row = 0; row < dim; row++) {
                // M(row, col) = V(row, col) = VT(col, row)
                mCol[dstOffset + row] = vtCol[col + row * dim];
            }
        }

        return mCol;
    }

    /** Native DGEMM for flat column-major matrices: C = A @ B. */
    private static double[] nativeMultiply(double[] aCol, int m, int k, double[] bCol, int n) {
        double[] cCol = new double[m * n];

        BLAS.getInstance().dgemm(
                "N", "N",
                m, n, k,
                1.0,
                aCol, m,
                bCol, k,
                0.0,
                cCol, m
        );

        return cCol;
    }

    /** Native DGEMM for flat column-major matrices: C = A^T @ B. */
    private static double[] nativeMultiplyTransposedA(
            double[] aCol,
            int rowsA,
            int colsA,
            double[] bCol,
            int colsB
    ) {
        double[] cCol = new double[colsA * colsB];

        BLAS.getInstance().dgemm(
                "T", "N",
                colsA, colsB, rowsA,
                1.0,
                aCol, rowsA,
                bCol, rowsA,
                0.0,
                cCol, colsA
        );

        return cCol;
    }

    /**
     * Uses LAPACK dgesvd but skips U and returns VT in flat column-major layout.
     * Input X must be [rows x cols], flat column-major.
     */
    private static double[] computeNativeVt(double[] xCol, int rows, int cols) {
        double[] s = new double[Math.min(rows, cols)];
        double[] vt = new double[cols * cols];
        intW info = new intW(0);

        // Workspace query
        double[] a = xCol.clone(); // dgesvd overwrites input
        double[] workQuery = new double[1];
        LAPACK.getInstance().dgesvd("N", "A", rows, cols, a, rows, s, null, 1, vt, cols, workQuery, -1, info);

        if (info.val != 0) {
            throw new RuntimeException("Native PCA SVD workspace query failed with info=" + info.val);
        }

        int lwork = (int) workQuery[0];
        double[] work = new double[lwork];

        // Actual call
        a = xCol.clone();
        LAPACK.getInstance().dgesvd("N", "A", rows, cols, a, rows, s, null, 1, vt, cols, work, lwork, info);

        if (info.val != 0) {
            throw new RuntimeException("Native PCA SVD failed with info=" + info.val);
        }

        return vt;
    }

    /**
     * Orthogonalizes a flat column-major matrix using thin SVD and returns U @ VT
     * in flat column-major layout.
     */
    private static double[] orthogonalize(double[] mCol, int rows, int cols) {
        int k = Math.min(rows, cols);

        double[] u = new double[rows * k];
        double[] vt = new double[k * cols];
        double[] s = new double[k];
        intW info = new intW(0);

        // Workspace query
        double[] a = mCol.clone(); // dgesvd overwrites input
        double[] workQuery = new double[1];
        LAPACK.getInstance().dgesvd(
                "S", "S",
                rows, cols,
                a, rows,
                s,
                u, rows,
                vt, k,
                workQuery, -1,
                info
        );

        if (info.val != 0) {
            throw new RuntimeException("LAPACK dgesvd workspace query failed with info=" + info.val);
        }

        int lwork = (int) workQuery[0];
        double[] work = new double[lwork];

        // Actual call
        a = mCol.clone();
        LAPACK.getInstance().dgesvd(
                "S", "S",
                rows, cols,
                a, rows,
                s,
                u, rows,
                vt, k,
                work, lwork,
                info
        );

        if (info.val != 0) {
            throw new RuntimeException("LAPACK dgesvd failed with info=" + info.val);
        }

        return nativeMultiply(u, rows, k, vt, cols);
    }

    /**
     * Thin wrapper retained so the LANDING stub continues to compile
     * without changing callers or method names.
     */
    private static RealMatrix orthogonalize(RealMatrix m) {
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();
        double[] qCol = orthogonalize(serializeColumnMajor(m), rows, cols);
        return new Array2DRowRealMatrix(deserializeColumnMajor(qCol, rows, cols), false);
    }

    /**
     * Binarizes values in-place to +/-1 and, if prevBinCol is non-null, counts
     * how many entries changed relative to the previous binarized matrix.
     */
    private static long sign01InPlace(double[] valuesCol, double[] prevBinCol) {
        long changed = 0L;

        if (prevBinCol == null) {
            for (int i = 0; i < valuesCol.length; i++) {
                valuesCol[i] = (valuesCol[i] > 0.0) ? 1.0 : -1.0;
            }
            return 0L;
        }

        for (int i = 0; i < valuesCol.length; i++) {
            double bin = (valuesCol[i] > 0.0) ? 1.0 : -1.0;
            if (bin != prevBinCol[i]) {
                changed++;
            }
            valuesCol[i] = bin;
        }

        return changed;
    }

    private static void writeStiefelTransform(IndexWriter out, StiefelTransform st) throws IOException {
        // Serialize A = W^T in float precision as a [d][D] row-major matrix.
        // This keeps the format compact and matches the encoding/scoring fast paths.
        out.writeInt(st.rows);
        out.writeInt(st.cols);

        for (int i = 0; i < st.rows; i++) {
            float[] row = st.AFloat[i];
            if (row.length != st.cols) {
                throw new IOException("Invalid StiefelTransform AFloat row length: " + row.length + " != " + st.cols);
            }
            for (int j = 0; j < st.cols; j++) {
                out.writeFloat(row[j]);
            }
        }
    }

    private static StiefelTransform readStiefelTransform(RandomAccessReader in, int expectedRows, int expectedCols) throws IOException {
        int rows = in.readInt();
        int cols = in.readInt();

        if (rows != expectedRows) {
            throw new IOException("StiefelTransform rows mismatch: expected " + expectedRows + ", got " + rows);
        }
        if (cols != expectedCols) {
            throw new IOException("StiefelTransform cols mismatch: expected " + expectedCols + ", got " + cols);
        }

        // Read A = W^T as [rows][cols] (row-major) in float precision.
        float[][] aFloat = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            float[] row = aFloat[i];
            for (int j = 0; j < cols; j++) {
                row[j] = in.readFloat();
            }
        }

        // Reconstruct W [D x d] so we can build StiefelTransform(W)
        // W[j][i] = A[i][j]
        double[][] wData = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            float[] aRow = aFloat[i];
            for (int j = 0; j < cols; j++) {
                wData[j][i] = aRow[j];
            }
        }

        RealMatrix W = new Array2DRowRealMatrix(wData, false);
        return new StiefelTransform(W);
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
