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

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Asymmetric Hashing (ASH) for float vectors.
 * Encodes each vector into a fixed-length binary code using a learned or random
 * orthonormal projection and sign thresholding.
 */
public class AsymmetricHashing implements VectorCompressor<AsymmetricHashing.QuantizedVector>, Accountable {
    public static final int ITQ = 1, LANDING = 2, RANDOM = 3;
    private static final int MAGIC = 0x75EC4012;  // TODO update the magic number?

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
                              StiefelTransform stiefelTransform) {
        this.originalDimension = originalDim;
        this.encodedBits = encodedBits;
        this.quantizedDim = quantizedDim;
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
                                               int encodedBits) {

        final int quantizedDim = encodedBits - HEADER_BITS;

        var ravvCopy = ravv.threadLocalSupplier().get();
        int originalDim = ravvCopy.getVector(0).length();
        // payload bits are always 64-bit aligned by validateEncodedBits()
        validateEncodedBits(encodedBits, HEADER_BITS, originalDim);

        // C=1 landmarks case
        VectorFloat<?> mu0 = vectorTypeSupport.createFloatVector(originalDim);
        for (int i = 0; i < ravvCopy.size(); i++) {
            VectorUtil.addInPlace(mu0, ravvCopy.getVector(i));
        }
        VectorUtil.scale(mu0, 1.0f / ravvCopy.size());

        VectorFloat<?>[] landmarks = new VectorFloat<?>[] { mu0 };

        StiefelTransform stiefelTransform;
        if (optimizer == RANDOM) {
            stiefelTransform = runWithoutTraining(originalDim, quantizedDim, new Random(42));
        } else if (optimizer == ITQ) {
            throw new IllegalArgumentException("ITQ optimizer not implemented yet");
        } else if (optimizer == LANDING) {
            throw new IllegalArgumentException("LANDING optimizer not implemented yet");
        } else {
            throw new IllegalArgumentException("Unknown optimizer " + optimizer);
        }

        return new AsymmetricHashing(
                originalDim,
                encodedBits,
                quantizedDim,
                optimizer,
                landmarks,
                stiefelTransform
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
        int optimizer = in.readInt();

        int landmark0Length = in.readInt();
        if (landmark0Length <= 0) {
            throw new IOException("ASH header missing landmark[0]");
        }
        VectorFloat<?> mu0 = vectorTypeSupport.readFloatVector(in, landmark0Length);
        VectorFloat<?>[] landmarks = new VectorFloat<?>[] { mu0 };

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

        return new AsymmetricHashing(
                originalDimension,
                encodedBits,
                quantizedDim,
                optimizer,
                landmarks,
                stiefelTransform
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

        // (C=1): write landmark[0] (dataset mean). Multi-landmark serialization TBD.
        out.writeInt(landmarks[0].length());
        vectorTypeSupport.writeFloatVector(out, landmarks[0]);

        // NOTE: stiefelTransform serialization will be added later
    }

    // ---------------------------------------------------------------------
    // Index-level quantization entry point
    // ---------------------------------------------------------------------

    public void quantizeVector(VectorFloat<?> vector,
                               float residualNorm,
                               short landmark,
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
        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {
            throw new UnsupportedOperationException("ITQ not implemented");
        }
    }

    public static final class LandingBinaryQuantizer implements BinaryQuantizer {
        @Override
        public void quantizeBody(VectorFloat<?> vector,
                                 VectorFloat<?> mu,
                                 float residualNorm,
                                 int quantizedDim,
                                 StiefelTransform stiefel,
                                 long[] outWords) {
            throw new UnsupportedOperationException("LANDING not implemented");
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
            return Float.BYTES + Float.BYTES + Short.BYTES
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

        // Landmark assignment (C=1 --> always 0)
        final byte landmark = 0;  // placeholder; will generalize

        final VectorFloat<?> mu = landmarks[landmark];

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
        size += Integer.BYTES; // landmarks[0] length
        size += Float.BYTES * landmarks[0].length();
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
