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
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.netlib.util.intW;

/**
 * Asymmetric Hashing (ASH) for float vectors.
 * Encodes each vector into a fixed-length code using a learned or random
 * orthonormal projection. The 1-bit case stores only sign bits; multibit ASH
 * stores either generic sign/extra bits or the C++-style fast-scan projection code for 2/4-bit ASH.
 */
public class AsymmetricHashing implements VectorCompressor<AsymmetricHashing.QuantizedVector>, Accountable {
    public static final int ITQ = 1, RANDOM = 2;
    private static final int MAGIC = 0x75EC4015;

    // ---------------------------------------------------------------------
    // Training configuration (reference defaults; TODO tune later)
    // ---------------------------------------------------------------------

    /**
     * training_size = min(N, D * training_factor).
     */
    private static final int ITQ_TRAINING_FACTOR = 100;

    /** Final setting TBD. */
    private static final int TRAINING_ITERS = 25;

    /** Current legacy default: one stored bit per projected dimension. */
    private static final int DEFAULT_BITS_PER_DIMENSION = 2;

    // Physical header size, reflecting actual stored fields:
    //  - scale: fp16 (16 bits), where scale = ||x − μ|| / ||code||
    //  - offset: fp16 (16 bits), where offset = <x, μ> − ||μ||_2^2
    //  - landmark id: byte (8 bits) in [0, C)
    public static final int HEADER_BITS =
            (Short.BYTES + Short.BYTES + Byte.BYTES) * 8; // 40 bits currently

    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    // ---------------------------------------------------------------------
    // Index-wide immutable state
    // ---------------------------------------------------------------------

    /** Number of landmarks (clusters), C <= 64. */
    public final int landmarkCount;

    /** Landmark centroids μ_0... μ_{C−1}. */
    public final VectorFloat<?>[] landmarks;

    // One entry per landmark
    private final float[] landmarkNormSq;

    /** Original (uncompressed) dimensionality. */
    public final int originalDimension;

    /** Total bits per encoded vector (header + body). */
    public final int encodedBits;

    /** Number of bits in the encoded body. */
    public final int bodyBits;

    /** Number of projected dimensions in the ASH body. */
    public final int quantizedDim;

    /** Bits stored per projected dimension. */
    public final int bitsPerDimension;

    /** Optimizer / learning mode. */
    public final int optimizer;

    /** Learned or random Stiefel transform. */
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

    /** Debug hook to disable learning paths. */
    @VisibleForTesting
    public boolean learn = true;

    // ---------------------------------------------------------------------
    // Quantizer instances (cached)
    // ---------------------------------------------------------------------

    private final BinaryQuantizer randomQuantizer = new RandomBinaryQuantizer();
    private final BinaryQuantizer itqQuantizer = new ItqBinaryQuantizer();

    private BinaryQuantizer quantizer() {
        switch (optimizer) {
            case RANDOM:
                return randomQuantizer;
            case ITQ:
                return itqQuantizer;
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
                              int bitsPerDimension,
                              int optimizer,
                              VectorFloat<?>[] landmarks,
                              StiefelTransform stiefelTransform,
                              float[][] landmarkProj,
                              float[] landmarkProjSum) {
        this.originalDimension = originalDim;
        this.encodedBits = encodedBits;
        this.quantizedDim = quantizedDim;
        this.bitsPerDimension = bitsPerDimension;
        this.bodyBits = encodedBits - HEADER_BITS;

        validateBitsPerDimension(bitsPerDimension);

        long expectedBodyBits = (long) quantizedDim * (long) bitsPerDimension;
        if (bodyBits != expectedBodyBits) {
            throw new IllegalArgumentException(
                    "Invalid ASH configuration: encodedBits=" + encodedBits +
                            ", quantizedDim=" + quantizedDim +
                            ", bitsPerDimension=" + bitsPerDimension +
                            ", HEADER_BITS=" + HEADER_BITS +
                            ", bodyBits=" + bodyBits
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
        if (quantizedDim > originalDimension) {
            throw new IllegalArgumentException(
                    "quantizedDim (" + quantizedDim + ") exceeds original dimension (" + originalDimension + ")");
        }
    }

    /**
     * Initialize ASH index-wide parameters using the legacy 1-bit body layout.
     *
     * @param ravv the vectors to quantize
     * @param optimizer the optimizer to use
     * @param encodedBits the number of bits used to encode vector, including the header
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount) throws IOException {
        return initialize(ravv, optimizer, encodedBits, landmarkCount, DEFAULT_BITS_PER_DIMENSION);
    }

    /**
     * Initialize ASH index-wide parameters.
     *
     * @param ravv the vectors to quantize
     * @param optimizer the optimizer to use
     * @param encodedBits the number of bits used to encode vector, including the header
     * @param landmarkCount number of landmarks
     * @param bitsPerDimension number of stored bits per projected dimension
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount,
                                               int bitsPerDimension) throws IOException {
        validateBitsPerDimension(bitsPerDimension);

        if (landmarkCount < 1 || landmarkCount > 64) {
            throw new IllegalArgumentException(
                    "landmarkCount must be in [1,64], got " + landmarkCount);
        }

        var ravvCopy = ravv.threadLocalSupplier().get();
        int originalDim = ravvCopy.getVector(0).length();
        final int quantizedDim = validateEncodedBits(encodedBits, HEADER_BITS, originalDim, bitsPerDimension);

        System.out.println(
                "\tASH initialized with landmarkCount=" + landmarkCount +
                        ", quantizedDim=" + quantizedDim +
                        ", bitsPerDimension=" + bitsPerDimension
        );

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
        } else if (optimizer == ITQ) {
            // Build training inputs from a random sample of points.
            TrainingData td = prepareTrainingData(points, landmarks, originalDim, rng);

            stiefelTransform = runItqTrainer(
                    td.xTrainNorm,
                    quantizedDim,
                    bitsPerDimension,
                    rng,
                    TRAINING_ITERS
            );
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
                bitsPerDimension,
                optimizer,
                landmarks,
                stiefelTransform,
                lp.proj,
                lp.sum
        );
    }

    private static void validateBitsPerDimension(int bitsPerDimension) {
        if (bitsPerDimension < 1 || bitsPerDimension > 9) {
            throw new IllegalArgumentException(
                    "bitsPerDimension must be in [1,9], got " + bitsPerDimension);
        }
    }

    /**
     * Validates encodedBits and returns the projected dimension count.
     */
    private static int validateEncodedBits(
            int encodedBits,
            int headerBits,
            int originalDim,
            int bitsPerDimension
    ) {
        validateBitsPerDimension(bitsPerDimension);

        int payloadBits = encodedBits - headerBits;

        if (payloadBits < 1) {
            throw new IllegalArgumentException(
                    "Illegal ASH payload bits: " + payloadBits +
                            " (encodedBits=" + encodedBits + ", headerBits=" + headerBits + ")"
            );
        }

        if (payloadBits % bitsPerDimension != 0) {
            throw new IllegalArgumentException(
                    "Invalid encodedBits=" + encodedBits +
                            ". ASH requires (encodedBits - headerBits) to be divisible by bitsPerDimension. " +
                            "Got payloadBits=" + payloadBits +
                            ", bitsPerDimension=" + bitsPerDimension + "."
            );
        }

        int projectedDims = payloadBits / bitsPerDimension;
        if (projectedDims < 1) {
            throw new IllegalArgumentException("ASH projected dimensions must be > 0");
        }
        if (projectedDims > originalDim) {
            throw new IllegalArgumentException(
                    "Invalid ASH projectedDims=" + projectedDims +
                            " exceeds originalDim=" + originalDim
            );
        }

        return projectedDims;
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
        int quantizedDim = in.readInt();
        int bitsPerDimension = in.readInt();

        int expectedQuantizedDim = validateEncodedBits(
                encodedBits,
                HEADER_BITS,
                originalDimension,
                bitsPerDimension);
        if (quantizedDim != expectedQuantizedDim) {
            throw new IOException(
                    "ASH format mismatch: encodedBits=" + encodedBits +
                            ", quantizedDim=" + quantizedDim +
                            ", bitsPerDimension=" + bitsPerDimension +
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

        // Load StiefelTransform.
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
                bitsPerDimension,
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
        out.writeInt(bitsPerDimension);
        out.writeInt(optimizer);

        // Multi-landmark serialization
        out.writeInt(landmarkCount);

        for (int c = 0; c < landmarkCount; c++) {
            out.writeInt(landmarks[c].length());
            vectorTypeSupport.writeFloatVector(out, landmarks[c]);
        }

        // Stiefel transform serialization:
        // Serialize A = W^T in float precision as a [d][D] row-major matrix.
        writeStiefelTransform(out, stiefelTransform);
    }

    // ---------------------------------------------------------------------
    // Index-level quantization entry point
    // ---------------------------------------------------------------------

    public float quantizeVector(VectorFloat<?> vector,
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

        return QuantizedVector.quantizeTo(
                vector,
                residualNorm,
                quantizedDim,
                bitsPerDimension,
                mu,
                stiefelTransform,
                quantizer(),
                dest
        );
    }


    // ---------------------------------------------------------------------
    // Projection-code layout helpers
    // ---------------------------------------------------------------------

    /**
     * Fast-scan projection codes use C++-compatible 4-bit projection groups.
     * Core ASH supports bitsPerDimension in [1,9], but this layout is only
     * defined for the nibble-friendly multibit cases used by FusedASH.
     */
    static boolean usesFastScanProjectionCode(int bitsPerDimension) {
        return bitsPerDimension == 2 || bitsPerDimension == 4;
    }

    static boolean supportsFusedAshBits(int bitsPerDimension) {
        return bitsPerDimension == 1 || bitsPerDimension == 2 || bitsPerDimension == 4;
    }

    static int projectionDimsPerNibble(int bitsPerDimension) {
        if (bitsPerDimension != 1 && bitsPerDimension != 2 && bitsPerDimension != 4) {
            throw new IllegalArgumentException(
                    "projection fast-scan layout supports bitsPerDimension in {1,2,4}, got "
                            + bitsPerDimension);
        }
        return 4 / bitsPerDimension;
    }

    static int projectionCodeGroups(int quantizedDim, int bitsPerDimension) {
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        return (quantizedDim + groupDims - 1) / groupDims;
    }

    static int projectionCodeBytesForDims(int quantizedDim, int bitsPerDimension) {
        return (projectionCodeGroups(quantizedDim, bitsPerDimension) + 1) >>> 1;
    }

    static void setFlatNibble(byte[] code, int group, int value) {
        int idx = group >>> 1;
        int v = value & 0x0F;
        if ((group & 1) == 0) {
            code[idx] = (byte) ((code[idx] & 0xF0) | v);
        } else {
            code[idx] = (byte) ((code[idx] & 0x0F) | (v << 4));
        }
    }

    static int flatNibble(byte[] code, int group) {
        int v = code[group >>> 1] & 0xFF;
        return ((group & 1) == 0) ? (v & 0x0F) : ((v >>> 4) & 0x0F);
    }

    static int projectionFieldForDimension(
            byte[] projectionCode,
            int dimension,
            int bitsPerDimension
    ) {
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int group = dimension / groupDims;
        int slot = dimension - group * groupDims;
        int mask = (1 << bitsPerDimension) - 1;
        return (flatNibble(projectionCode, group) >>> (slot * bitsPerDimension)) & mask;
    }

    static float decodeProjectionComponent(int field, int bitsPerDimension) {
        if (bitsPerDimension == 1) {
            return field != 0 ? 1.0f : -1.0f;
        }

        int exBits = bitsPerDimension - 1;
        int magMask = (1 << exBits) - 1;
        boolean positive = ((field >>> exBits) & 1) != 0;
        float mag = (field & magMask) + 0.5f;
        return positive ? mag : -mag;
    }

    static float projectionComponent(
            byte[] projectionCode,
            int dimension,
            int bitsPerDimension
    ) {
        return decodeProjectionComponent(
                projectionFieldForDimension(projectionCode, dimension, bitsPerDimension),
                bitsPerDimension);
    }

    static float dotProjectionCode(
            float[] projectedVector,
            byte[] projectionCode,
            int quantizedDim,
            int bitsPerDimension
    ) {
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int groups = projectionCodeGroups(quantizedDim, bitsPerDimension);
        float sum = 0.0f;

        for (int group = 0; group < groups; group++) {
            int nibble = flatNibble(projectionCode, group);
            int baseDim = group * groupDims;
            for (int slot = 0; slot < groupDims; slot++) {
                int dim = baseDim + slot;
                if (dim >= quantizedDim) {
                    break;
                }
                int field = (nibble >>> (slot * bitsPerDimension)) & ((1 << bitsPerDimension) - 1);
                sum += projectedVector[dim] * decodeProjectionComponent(field, bitsPerDimension);
            }
        }

        return sum;
    }

    // ---------------------------------------------------------------------
    // Quantization strategies
    // ---------------------------------------------------------------------

    public interface BinaryQuantizer {
        float quantizeBody(VectorFloat<?> vector,
                           VectorFloat<?> mu,
                           float residualNorm,
                           int quantizedDim,
                           int bitsPerDimension,
                           StiefelTransform stiefel,
                           long[] outSignWords,
                           byte[] outExtraBits);
    }

    public static final class RandomBinaryQuantizer implements BinaryQuantizer {
        // ThreadLocal to hold the reusable buffers.
        // Using a simple container class to avoid multiple ThreadLocal lookups.
        private static final ThreadLocal<Workspace> WORKSPACE = ThreadLocal.withInitial(Workspace::new);

        private static final class Workspace {
            float[] x;
            float[] muArr;
            float[] xhat;
            float[] proj;
            double[] absNorm;

            void ensureCapacity(int originalDim, int quantizedDim) {
                if (x == null || x.length < originalDim) {
                    x = new float[originalDim];
                    muArr = new float[originalDim];
                    xhat = new float[originalDim];
                }
                if (proj == null || proj.length < quantizedDim) {
                    proj = new float[quantizedDim];
                    absNorm = new double[quantizedDim];
                }
            }
        }

        @Override
        public float quantizeBody(VectorFloat<?> vector,
                                  VectorFloat<?> mu,
                                  float residualNorm,
                                  int quantizedDim,
                                  int bitsPerDimension,
                                  StiefelTransform stiefel,
                                  long[] outSignWords,
                                  byte[] outExtraBits) {

            final float[][] A = stiefel.AFloat;
            final int originalDim = stiefel.cols;

            // Get workspace for this thread and ensure correct sizing.
            Workspace ws = WORKSPACE.get();
            ws.ensureCapacity(originalDim, quantizedDim);

            // Alias for readability (no new allocations).
            final float[] x = ws.x;
            final float[] muArr = ws.muArr;
            final float[] xhat = ws.xhat;

            final var vecUtil =
                    io.github.jbellis.jvector.vector.VectorizationProvider
                            .getInstance()
                            .getVectorUtilSupport();

            // Copy vector and mean once into workspace.
            for (int d = 0; d < originalDim; d++) {
                x[d] = vector.get(d);
                muArr[d] = mu.get(d);
            }

            // ASH paper, Eq. 6, normalization.
            final float invNorm = (residualNorm > 0f) ? (1.0f / residualNorm) : 0.0f;

            // Compute normalized residual x̂.
            for (int d = 0; d < originalDim; d++) {
                xhat[d] = (x[d] - muArr[d]) * invNorm;
            }

            if (bitsPerDimension == 1) {
                // Binarize directly from per-row projection.
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
                    outSignWords[w] = bits;
                }

                return (float) Math.sqrt(quantizedDim);
            }

            final int exBits = bitsPerDimension - 1;
            final int maxCode = (1 << exBits) - 1;
            final float[] proj = ws.proj;
            final double[] absNorm = ws.absNorm;

            double projNorm2 = 0.0;
            for (int j = 0; j < quantizedDim; j++) {
                float acc = vecUtil.ashDotRow(A[j], xhat);
                proj[j] = acc;
                projNorm2 += (double) acc * (double) acc;
                if (!usesFastScanProjectionCode(bitsPerDimension) && acc >= 0.0f) {
                    QuantizedVector.setBit(outSignWords, j);
                }
            }

            final double invProjNorm = 1.0 / Math.max(Math.sqrt(projNorm2), 1e-20);
            for (int j = 0; j < quantizedDim; j++) {
                absNorm[j] = Math.abs((double) proj[j]) * invProjNorm;
            }

            final double t = computeOptimalScalingFactor(absNorm, quantizedDim, bitsPerDimension);

            if (usesFastScanProjectionCode(bitsPerDimension)) {
                return quantizeFastScanProjectionCode(
                        proj,
                        absNorm,
                        t,
                        quantizedDim,
                        bitsPerDimension,
                        outExtraBits);
            }

            double codeNorm2 = 0.0;
            for (int j = 0; j < quantizedDim; j++) {
                int q = Math.min((int) (t * absNorm[j] + K_EPS), maxCode);
                double mag = q + 0.5;
                codeNorm2 += mag * mag;

                int extraCode = (proj[j] < 0.0f) ? ((~q) & maxCode) : q;
                QuantizedVector.writeExtraCode(outExtraBits, j, exBits, extraCode);
            }

            return (float) Math.sqrt(Math.max(codeNorm2, 1e-20));
        }

        private static float quantizeFastScanProjectionCode(
                float[] proj,
                double[] absNorm,
                double t,
                int quantizedDim,
                int bitsPerDimension,
                byte[] outProjectionCode
        ) {
            final int groupDims = projectionDimsPerNibble(bitsPerDimension);
            final int groups = projectionCodeGroups(quantizedDim, bitsPerDimension);
            final int exBits = bitsPerDimension - 1;
            final int maxCode = (1 << exBits) - 1;

            double codeNorm2 = 0.0;
            for (int group = 0; group < groups; group++) {
                int nibble = 0;
                int baseDim = group * groupDims;

                for (int slot = 0; slot < groupDims; slot++) {
                    int j = baseDim + slot;
                    if (j >= quantizedDim) {
                        break;
                    }

                    int q = Math.min((int) (t * absNorm[j] + K_EPS), maxCode);
                    boolean positive = proj[j] >= 0.0f;
                    int field = (positive ? (1 << exBits) : 0) | q;
                    nibble |= field << (slot * bitsPerDimension);

                    double mag = q + 0.5;
                    codeNorm2 += mag * mag;
                }

                setFlatNibble(outProjectionCode, group, nibble);
            }

            return (float) Math.sqrt(Math.max(codeNorm2, 1e-20));
        }
    }

    public static final class ItqBinaryQuantizer implements BinaryQuantizer {
        // Same quantization math for all optimizers; only the learned transform differs.
        // TODO consolidate quantizeBody across learners/random.
        private final RandomBinaryQuantizer delegate = new RandomBinaryQuantizer();

        @Override
        public float quantizeBody(VectorFloat<?> vector,
                                  VectorFloat<?> mu,
                                  float residualNorm,
                                  int quantizedDim,
                                  int bitsPerDimension,
                                  StiefelTransform stiefel,
                                  long[] outSignWords,
                                  byte[] outExtraBits) {
            return delegate.quantizeBody(vector, mu, residualNorm, quantizedDim, bitsPerDimension,
                    stiefel, outSignWords, outExtraBits);
        }
    }

    // ---------------------------------------------------------------------
    // Quantized vector (per-vector data)
    // ---------------------------------------------------------------------

    public static class QuantizedVector {
        public float scale; // ||x_i - μ_i*||_2 / ||encoded body||_2
        public float offset; // offset_i = <x_i, μ_i*> - ||μ_i*||^2
        public byte landmark; // c_i*, unsigned [0, C)
        public long[] binaryVector; // sign bits; bit=1 means positive code value
        public byte[] extraBits; // packed extra magnitude/sign-complement bits for multibit ASH

        private QuantizedVector(float scale,
                                float offset,
                                byte landmark,
                                long[] binaryVector,
                                byte[] extraBits) {
            this.scale = scale;
            this.offset = offset;
            this.landmark = landmark;
            this.binaryVector = binaryVector;
            this.extraBits = extraBits;
        }

        public static int wordsForDims(int quantizedDim) {
            return (quantizedDim + 63) >>> 6;
        }

        public static int signWordsForDims(int quantizedDim, int bitsPerDimension) {
            return usesFastScanProjectionCode(bitsPerDimension) ? 0 : wordsForDims(quantizedDim);
        }

        public static int extraBytesForDims(int quantizedDim, int bitsPerDimension) {
            int exBits = bitsPerDimension - 1;
            if (exBits <= 0) {
                return 0;
            }
            return (quantizedDim * exBits + 7) >>> 3;
        }

        public static int bodyBytesForDims(int quantizedDim, int bitsPerDimension) {
            if (usesFastScanProjectionCode(bitsPerDimension)) {
                return projectionCodeBytesForDims(quantizedDim, bitsPerDimension);
            }
            return extraBytesForDims(quantizedDim, bitsPerDimension);
        }

        public static QuantizedVector createEmpty(int quantizedDim) {
            return createEmpty(quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        public static QuantizedVector createEmpty(int quantizedDim, int bitsPerDimension) {
            return new QuantizedVector(
                    Float.NaN,
                    Float.NaN,
                    (byte) 0,
                    new long[signWordsForDims(quantizedDim, bitsPerDimension)],
                    new byte[bodyBytesForDims(quantizedDim, bitsPerDimension)]
            );
        }

        public static int serializedSizeBytes(int quantizedDim) {
            return serializedSizeBytes(quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        public static int serializedSizeBytes(int quantizedDim, int bitsPerDimension) {
            return Short.BYTES + Short.BYTES + Byte.BYTES
                    + signWordsForDims(quantizedDim, bitsPerDimension) * Long.BYTES
                    + bodyBytesForDims(quantizedDim, bitsPerDimension);
        }

        /**
         * Writes only the encoded body into {@code dest}.
         *
         * <p>Header fields ({@code offset}, {@code landmark}) must be set by the caller
         * before calling this method. {@code scale} is set by the caller after this method
         * returns the encoded body norm.</p>
         *
         * @param residualNorm  ||x - μ||_2, required for Eq. 6 normalization during encoding
         * @return norm of the stored centered code body
         */
        static float quantizeTo(VectorFloat<?> vector,
                                float residualNorm,
                                int quantizedDim,
                                int bitsPerDimension,
                                VectorFloat<?> mu,
                                StiefelTransform stiefel,
                                BinaryQuantizer quantizer,
                                QuantizedVector dest) {

            assert Float.isFinite(dest.offset) : "offset is not finite";

            int words = signWordsForDims(quantizedDim, bitsPerDimension);
            if (dest.binaryVector == null || dest.binaryVector.length < words) {
                throw new IllegalArgumentException("binaryVector too short");
            }

            int bodyBytes = bodyBytesForDims(quantizedDim, bitsPerDimension);
            if (dest.extraBits == null || dest.extraBits.length < bodyBytes) {
                throw new IllegalArgumentException("extraBits/projectionCode too short");
            }

            if (words > 0) {
                Arrays.fill(dest.binaryVector, 0, words, 0L);
            }
            if (bodyBytes > 0) {
                Arrays.fill(dest.extraBits, 0, bodyBytes, (byte) 0);
            }

            return quantizer.quantizeBody(
                    vector,
                    mu,
                    residualNorm,
                    quantizedDim,
                    bitsPerDimension,
                    stiefel,
                    dest.binaryVector,
                    dest.extraBits
            );
        }

        public void write(IndexWriter out, int quantizedDim) throws IOException {
            write(out, quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        public void write(IndexWriter out, int quantizedDim, int bitsPerDimension) throws IOException {
            writeFloat16(out, scale);
            writeFloat16(out, offset);
            out.writeByte(landmark);

            int words = signWordsForDims(quantizedDim, bitsPerDimension);
            for (int i = 0; i < words; i++) {
                out.writeLong(binaryVector[i]);
            }

            int bodyBytes = bodyBytesForDims(quantizedDim, bitsPerDimension);
            for (int i = 0; i < bodyBytes; i++) {
                out.writeByte(extraBits[i]);
            }
        }

        public static QuantizedVector load(RandomAccessReader in,
                                           int quantizedDim) throws IOException {
            return load(in, quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        public static QuantizedVector load(RandomAccessReader in,
                                           int quantizedDim,
                                           int bitsPerDimension) throws IOException {
            float scale = readFloat16(in);
            float offset = readFloat16(in);

            byte landmark;
            if (in instanceof io.github.jbellis.jvector.disk.ByteReadable) {
                landmark = ((io.github.jbellis.jvector.disk.ByteReadable) in).readByte();
            } else {
                throw new IOException("ASH requires ByteReadable reader for landmark");
            }

            int words = signWordsForDims(quantizedDim, bitsPerDimension);
            long[] signWords = new long[words];
            for (int i = 0; i < words; i++) {
                signWords[i] = in.readLong();
            }

            int bodyBytes = bodyBytesForDims(quantizedDim, bitsPerDimension);
            byte[] body = new byte[bodyBytes];
            if (bodyBytes > 0) {
                in.readFully(body);
            }

            return new QuantizedVector(scale, offset, landmark, signWords, body);
        }

        public static void loadInto(RandomAccessReader in,
                                    QuantizedVector dest,
                                    int quantizedDim) throws IOException {
            loadInto(in, dest, quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        public static void loadInto(RandomAccessReader in,
                                    QuantizedVector dest,
                                    int quantizedDim,
                                    int bitsPerDimension) throws IOException {
            dest.scale = readFloat16(in);
            dest.offset = readFloat16(in);

            if (in instanceof io.github.jbellis.jvector.disk.ByteReadable) {
                dest.landmark = ((io.github.jbellis.jvector.disk.ByteReadable) in).readByte();
            } else {
                throw new IOException("ASH requires ByteReadable reader for landmark");
            }

            int words = signWordsForDims(quantizedDim, bitsPerDimension);
            if (dest.binaryVector == null || dest.binaryVector.length != words) {
                dest.binaryVector = new long[words];
            }
            for (int i = 0; i < words; i++) {
                dest.binaryVector[i] = in.readLong();
            }

            int bodyBytes = bodyBytesForDims(quantizedDim, bitsPerDimension);
            if (dest.extraBits == null || dest.extraBits.length != bodyBytes) {
                dest.extraBits = new byte[bodyBytes];
            }
            if (bodyBytes > 0) {
                in.readFully(dest.extraBits);
            }
        }

        static void setBit(long[] words, int bitIndex) {
            words[bitIndex >>> 6] |= 1L << (bitIndex & 63);
        }

        static boolean getBit(long[] words, int bitIndex) {
            return ((words[bitIndex >>> 6] >>> (bitIndex & 63)) & 1L) != 0L;
        }

        static void writeExtraCode(byte[] extraBits, int dim, int exBits, int code) {
            int bitPos = dim * exBits;
            for (int bit = 0; bit < exBits; bit++, bitPos++) {
                if ((code & (1 << bit)) != 0) {
                    extraBits[bitPos >>> 3] |= (byte) (1 << (bitPos & 7));
                }
            }
        }

        static int readExtraCode(byte[] extraBits, int dim, int exBits) {
            int code = 0;
            int bitPos = dim * exBits;
            for (int bit = 0; bit < exBits; bit++, bitPos++) {
                code |= ((extraBits[bitPos >>> 3] >>> (bitPos & 7)) & 1) << bit;
            }
            return code;
        }

        /**
         * Compares two ASH quantized vectors for exact encoded equality.
         *
         * <p>
         * Equality is defined as bitwise equality of all encoded components:
         * <ul>
         *   <li>Sign-code payload</li>
         *   <li>Extra-bit payload</li>
         *   <li>Associated landmark index</li>
         *   <li>Auxiliary scalar values</li>
         * </ul>
         *
         * <p>
         * Floating-point values are compared using their raw bit representations
         * rather than tolerance-based comparisons.
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
                    && Arrays.equals(binaryVector, that.binaryVector)
                    && Arrays.equals(extraBits, that.extraBits);
        }

        /**
         * Hash code consistent with {@link #equals(Object)}.
         */
        @Override
        public int hashCode() {
            int result = Integer.hashCode(Float.floatToIntBits(scale));
            result = 31 * result + Integer.hashCode(Float.floatToIntBits(offset));
            result = 31 * result + Short.hashCode(landmark);
            result = 31 * result + Arrays.hashCode(binaryVector);
            result = 31 * result + Arrays.hashCode(extraBits);
            return result;
        }
    }

    /**
     * Private helper to provide byte storage for binary body.
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
     * @param d dimensionality of the projected space
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
     */
    @Override
    public ASHVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        final int n = ravv.size();

        // The caller controls parallelism; we derive a bounded number of tasks
        // directly from the executor rather than using parallel streams.
        final int parallelism = Math.max(1, simdExecutor.getParallelism());

        // Each task processes a contiguous range of ordinals.
        final int chunkSize = (n + parallelism - 1) / parallelism;

        final QuantizedVector[] out = new QuantizedVector[n];
        final var ravvSupplier = ravv.threadLocalSupplier();

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
                        out[i] = QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
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
     * @return the header and encoded vector payload
     */
    @Override
    public QuantizedVector encode(VectorFloat<?> vector) {
        var qv = QuantizedVector.createEmpty(this.quantizedDim, this.bitsPerDimension);
        encodeTo(vector, qv);
        return qv;
    }

    @Override
    public void encodeTo(VectorFloat<?> vector, QuantizedVector dest) {
        // Landmark assignment: nearest centroid by L2 distance.
        byte landmark = 0;
        float bestDist = Float.POSITIVE_INFINITY;

        for (int c = 0; c < landmarkCount; c++) {
            float dist = VectorUtil.squareL2Distance(vector, landmarks[c]);
            if (dist < bestDist) {
                bestDist = dist;
                landmark = (byte) c;
            }
        }

        final int c = landmark & 0xFF;
        final VectorFloat<?> mu = landmarks[c];

        // Compute <x, μ> and ||x_i − μ_i*||_2.
        final float dotXMu = VectorUtil.dotProduct(vector, mu);
        final float sqDist = VectorUtil.squareL2Distance(vector, mu);
        final float residualNorm = (float) Math.sqrt(sqDist);

        assert quantizedDim > 0;

        final float rawOffset = dotXMu - landmarkNormSq[c];

        // Header fields needed before quantization. The final offset may be
        // adjusted after quantization for C++-style fast-scan projection codes.
        dest.offset = rawOffset;
        dest.landmark = landmark;

        // Body: sign bits, generic sign+extra bits, or fast-scan projection code.
        // The returned code norm is sqrt(d) for 1-bit ASH and
        // sqrt(sum_j (q_j + 0.5)^2) for multibit ASH.
        float codeNorm = quantizeVector(vector, residualNorm, landmark, dest);
        final float scale = codeNorm > 0f ? residualNorm / codeNorm : 0f;

        float finalOffset = rawOffset;
        if (usesFastScanProjectionCode(bitsPerDimension)) {
            // C++ projection-mode optimization:
            //   offset = <x, μ> - ||μ||² - scale * <Aμ, code>
            // so query-time scoring can use Aq instead of A(q - μ).
            float landmarkDot = dotProjectionCode(
                    landmarkProj[c],
                    dest.extraBits,
                    quantizedDim,
                    bitsPerDimension);
            finalOffset -= scale * landmarkDot;
        }

        dest.scale = roundToFloat16(scale);
        dest.offset = roundToFloat16(finalOffset);
    }

    // ---------------------------------------------------------------------
    // Training of projection matrix
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
     *  - Choose N_train = min(N, D * training_factor).
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
            int originalDim,
            Random rng
    ) {
        final int N = points.length;

        long learnedCap = Math.max(1L, (long) originalDim * ITQ_TRAINING_FACTOR);
        final int nTrain = (int) Math.min((long) N, learnedCap);

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

            // Normalize residuals while guarding against zero residuals.
            double inv = 1.0 / Math.max(Math.sqrt(ss), 1e-20);
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
     * ITQ / learned projection trainer.
     *
     * Mirrors the current learned_projection path:
     *  - PCA basis P from SVD(X), matching the Python reference
     *  - Xld = X @ P
     *  - iterative Procrustes:
     *      R = polar(M)
     *      Xtr = Xld @ R
     *      Xenc = encodeTrainingDirections(Xtr, bitsPerDimension)
     *      M = Xld^T @ Xenc
     *  - W = P @ R
     */
    private static StiefelTransform runItqTrainer(
            double[][] xHdNorm,       // [N][D] normalized residuals
            int latentDim,            // projected dimensions, d
            int bitsPerDimension,
            Random rng,
            int nTrainingIterations
    ) {
        logProgress("\t[stage] Starting runITQTrainer...");

        validateBitsPerDimension(bitsPerDimension);

        final int N = xHdNorm.length;
        final int D = xHdNorm[0].length;
        final int d = latentDim;

        if (d <= 0 || d > D) {
            throw new IllegalArgumentException("Invalid latentDim=" + d + " for D=" + D);
        }

        // X is [N x D], flat column-major for BLAS/LAPACK.
        double[] xCol = serializeColumnMajor(xHdNorm);

        // PCA basis P = V[:, :d] from SVD(X), matching the Python reference.
        logProgress("\t[stage] Starting Native SVD for PCA...");
        double[] vtCol = computeNativeVt(xCol, N, D);    // [D x D], column-major
        double[] pCol = extractPcaBasis(vtCol, D, d);    // [D x d], column-major
        logProgress("\t[stage] Completed Native SVD for PCA...");

        // Xld = X @ P, shape [N x d].
        double[] xldCol = nativeMultiply(xCol, N, D, pCol, d);

        // Initial random update matrix M, shape [d x d].
        double[] mCol = gaussianMatrix(d, d, rng);

        double[] rCol = null;
        double[] xtrCol = new double[N * d];
        double[] xencCol = new double[N * d];

        logProgress("\t[stage] ITQ training iterations started...");
        long startTime = System.nanoTime();

        for (int epoch = 0; epoch < nTrainingIterations; epoch++) {
            rCol = orthogonalize(mCol, d, d);

            multiplyInto(xldCol, N, d, rCol, d, xtrCol);
            encodeTrainingDirections(xtrCol, N, d, bitsPerDimension, xencCol);

            // ITQ update: M = Xld^T @ Xenc.
            mCol = nativeMultiplyTransposedA(xldCol, N, d, xencCol, d);

            // C++-style cheap loss: trace(R^T M) / N.
            ProjectionTrainingLoss loss = computeTrainingLossFromUpdateMatrix(rCol, mCol, N);
            printTrainingLoss(epoch, loss);
        }

        // Final polar step after the last update.
        rCol = orthogonalize(mCol, d, d);

        double loopTime = (System.nanoTime() - startTime) / 1e9;
        logProgress("\t[stage] ITQ training completed in " + loopTime + " seconds.");

        // W = P @ R, shape [D x d].
        double[] wCol = nativeMultiply(pCol, D, d, rCol, d);

        RealMatrix W = new Array2DRowRealMatrix(deserializeColumnMajor(wCol, D, d), false);
        return new StiefelTransform(W);
    }

    private static final class ProjectionTrainingLoss {
        final double ip;
        final double loss;

        ProjectionTrainingLoss(double ip, double loss) {
            this.ip = ip;
            this.loss = loss;
        }
    }

    /**
     * Java equivalent of C++ compute_training_loss_from_update_matrix(R, M).
     *
     * R and M are [d x d] column-major.
     */
    private static ProjectionTrainingLoss computeTrainingLossFromUpdateMatrix(
            double[] rCol,
            double[] mCol,
            int rows
    ) {
        double ipSum = 0.0;
        for (int i = 0; i < rCol.length; i++) {
            ipSum += rCol[i] * mCol[i];
        }

        double ip = ipSum / rows;
        return new ProjectionTrainingLoss(ip, Math.max(0.0, 1.0 - ip));
    }

    private static void printTrainingLoss(int epoch, ProjectionTrainingLoss loss) {
        logProgress(String.format(java.util.Locale.ROOT,
                "projection_train epoch=%2d ip_loss=%.5f loss=%.5f",
                epoch, loss.ip, loss.loss));
    }

    private static final float[] K_TIGHT_START = {
            0.0f,
            0.15f,
            0.20f,
            0.52f,
            0.59f,
            0.71f,
            0.75f,
            0.77f,
            0.81f
    };

    private static final double K_EPS = 1e-5;

    private static final ThreadLocal<ScalingWorkspace> SCALING_WORKSPACE =
            ThreadLocal.withInitial(ScalingWorkspace::new);

    private static final class ScalingWorkspace {
        float[] invOAbs = new float[0];
        int[] curOBar = new int[0];
        long[] events = new long[0];

        void ensureCapacity(int d, int maxEvents) {
            if (invOAbs.length < d) {
                invOAbs = new float[d];
            }
            if (curOBar.length < d) {
                curOBar = new int[d];
            }
            if (events.length < maxEvents) {
                events = new long[maxEvents];
            }
        }
    }

    /**
     * Java port of ash::compute_optimal_scaling_factor.
     *
     * The input array is double because the Java trainer stores training matrices
     * as double, but this method intentionally performs the search in float
     * precision to match the C++ implementation.
     */
    private static double computeOptimalScalingFactor(
            double[] oAbs,
            int d,
            int bitsPerDimension
    ) {
        final int exBits = bitsPerDimension - 1;
        if (exBits < 1 || exBits > 8) {
            throw new IllegalArgumentException("exBits must be in [1,8], got " + exBits);
        }
        if (d <= 0 || d > oAbs.length) {
            throw new IllegalArgumentException("Invalid d=" + d + " for oAbs.length=" + oAbs.length);
        }

        final int kNEnum = 10;
        final int maxCode = (1 << exBits) - 1;

        float maxO = 0.0f;
        for (int i = 0; i < d; i++) {
            float v = (float) oAbs[i];
            if (v > maxO) {
                maxO = v;
            }
        }

        if (!(maxO > 0.0f)) {
            return 0.0;
        }

        final float tEnd = (float) (maxCode + kNEnum) / maxO;
        final float tStart = tEnd * K_TIGHT_START[exBits];

        final int maxEvents = Math.multiplyExact(d, maxCode + 1);
        ScalingWorkspace ws = SCALING_WORKSPACE.get();
        ws.ensureCapacity(d, maxEvents);

        float sqrDenominator = (float) d * 0.25f;
        float numerator = 0.0f;

        for (int i = 0; i < d; i++) {
            float oi = (float) oAbs[i];

            ws.invOAbs[i] = 1.0f / oi;

            int cur = (int) ((double) (tStart * oi) + K_EPS);
            ws.curOBar[i] = cur;

            sqrDenominator += (float) (cur * cur + cur);
            numerator += (cur + 0.5f) * oi;
        }

        float invSqrtDenom = 1.0f / (float) Math.sqrt(sqrDenominator);

        int eventCount = 0;
        for (int i = 0; i < d; i++) {
            final float invO = ws.invOAbs[i];
            final int first = ws.curOBar[i] + 1;

            // Match C++: always consider the first event, then only generate
            // follow-on events while k <= maxCode.
            float tn = (float) first * invO;
            if (tn >= tEnd) {
                continue;
            }

            ws.events[eventCount++] = packScalingEvent(tn, i);

            for (int k = first + 1; k <= maxCode; k++) {
                tn = (float) k * invO;
                if (tn >= tEnd) {
                    break;
                }

                ws.events[eventCount++] = packScalingEvent(tn, i);
            }
        }

        Arrays.sort(ws.events, 0, eventCount);

        float maxIp = 0.0f;
        float t = 0.0f;

        for (int e = 0; e < eventCount; e++) {
            long event = ws.events[e];
            int id = scalingEventId(event);

            ws.curOBar[id]++;
            int update = ws.curOBar[id];

            float delta = 2.0f * update;
            sqrDenominator += delta;
            numerator += (float) oAbs[id];

            float oldDenom = sqrDenominator - delta;
            invSqrtDenom *=
                    (1.0f - 0.5f * delta / (oldDenom + delta * 0.5f));

            float curIp = numerator * invSqrtDenom;
            if (curIp > maxIp) {
                maxIp = curIp;
                t = scalingEventT(event);
            }
        }

        return t;
    }

    /**
     * Packs event sort key as:
     *   high 32 bits: positive float t bits
     *   low  32 bits: dimension id
     *
     * For positive finite t, raw float bits sort in the same order as numeric t.
     * The low bits reproduce the C++ tie-breaker by id.
     */
    private static long packScalingEvent(float t, int id) {
        return ((long) Float.floatToRawIntBits(t) << 32)
                | (id & 0xFFFF_FFFFL);
    }

    private static float scalingEventT(long event) {
        return Float.intBitsToFloat((int) (event >>> 32));
    }

    private static int scalingEventId(long event) {
        return (int) event;
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
     * PCA basis from eig(X^T X).
     *
     * Input X is [rows x cols] in column-major layout.
     * Returns P = top right singular vectors of X, shape [cols x latentDim],
     * also column-major.
     */
    private static double[] pcaProjectionFromCovariance(
            double[] xCol,
            int rows,
            int cols,
            int latentDim
    ) {
        double[] covCol = new double[cols * cols];

        // cov = X^T @ X, shape [cols x cols].
        BLAS.getInstance().dgemm(
                "T", "N",
                cols, cols, rows,
                1.0,
                xCol, rows,
                xCol, rows,
                0.0,
                covCol, cols
        );

        symmetricEigenvectorsInPlace(covCol, cols);

        // LAPACK dsyev returns eigenvalues ascending and eigenvectors by columns.
        // Select the largest latentDim eigenvectors.
        double[] pCol = new double[cols * latentDim];
        for (int col = 0; col < latentDim; col++) {
            int eigCol = cols - 1 - col;
            System.arraycopy(covCol, eigCol * cols, pCol, col * cols, cols);
        }

        return pCol;
    }

    /**
     * Overwrites a symmetric column-major matrix with eigenvectors by column.
     */
    private static void symmetricEigenvectorsInPlace(double[] aCol, int dim) {
        double[] evals = new double[dim];
        intW info = new intW(0);

        double[] workQuery = new double[1];
        LAPACK.getInstance().dsyev(
                "V", "U",
                dim,
                aCol, dim,
                evals,
                workQuery, -1,
                info
        );

        if (info.val != 0) {
            throw new RuntimeException("LAPACK dsyev workspace query failed with info=" + info.val);
        }

        int lwork = Math.max(1, (int) workQuery[0]);
        double[] work = new double[lwork];

        LAPACK.getInstance().dsyev(
                "V", "U",
                dim,
                aCol, dim,
                evals,
                work, lwork,
                info
        );

        if (info.val != 0) {
            throw new RuntimeException("LAPACK dsyev failed with info=" + info.val);
        }
    }

    /**
     * Native DGEMM into caller-provided output: C = A @ B.
     */
    private static void multiplyInto(
            double[] aCol,
            int m,
            int k,
            double[] bCol,
            int n,
            double[] cCol
    ) {
        if (cCol.length < m * n) {
            throw new IllegalArgumentException("Output matrix too small");
        }

        BLAS.getInstance().dgemm(
                "N", "N",
                m, n, k,
                1.0,
                aCol, m,
                bCol, k,
                0.0,
                cCol, m
        );
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
     * Encodes projected training directions.
     *
     * Input and output are [rows x dims] column-major.
     *
     * bitsPerDimension == 1:
     *   y_j = sign(x_j) / sqrt(d)
     *
     * bitsPerDimension > 1:
     *   y_j uses signed multibit magnitudes and is normalized to unit length.
     */
    private static void encodeTrainingDirections(
            double[] xCol,
            int rows,
            int dims,
            int bitsPerDimension,
            double[] outCol
    ) {
        if (outCol.length < rows * dims) {
            throw new IllegalArgumentException("Output matrix too small");
        }

        if (bitsPerDimension <= 1) {
            final double invSqrtD = 1.0 / Math.sqrt(dims);
            final int size = rows * dims;

            for (int i = 0; i < size; i++) {
                outCol[i] = xCol[i] >= 0.0 ? invSqrtD : -invSqrtD;
            }
            return;
        }

        final int excessBits = bitsPerDimension - 1;
        final int maxCode = (1 << excessBits) - 1;

        double[] absNorm = new double[dims];
        double[] mag = new double[dims];

        for (int r = 0; r < rows; r++) {
            double norm2 = 0.0;
            for (int j = 0; j < dims; j++) {
                double v = xCol[r + j * rows];
                norm2 += v * v;
            }

            double invNorm = 1.0 / Math.sqrt(Math.max(norm2, 1e-20));

            for (int j = 0; j < dims; j++) {
                absNorm[j] = Math.abs(xCol[r + j * rows]) * invNorm;
            }

            double t = computeOptimalScalingFactor(absNorm, dims, bitsPerDimension);

            double encodedNorm2 = 0.0;
            for (int j = 0; j < dims; j++) {
                int q = Math.min((int) (t * absNorm[j] + K_EPS), maxCode);
                double m = q + 0.5;
                mag[j] = m;
                encodedNorm2 += m * m;
            }

            double encodedInvNorm = 1.0 / Math.sqrt(Math.max(encodedNorm2, 1e-20));

            for (int j = 0; j < dims; j++) {
                double signed = xCol[r + j * rows] < 0.0 ? -mag[j] : mag[j];
                outCol[r + j * rows] = signed * encodedInvNorm;
            }
        }
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

        // Reconstruct W [D x d] so we can build StiefelTransform(W).
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
        size += Integer.BYTES; // bitsPerDimension
        size += Integer.BYTES; // optimizer
        size += Integer.BYTES; // landmarkCount
        for (int c = 0; c < landmarkCount; c++) {
            size += Integer.BYTES; // landmark dimension
            size += Float.BYTES * landmarks[c].length();
        }
        size += Integer.BYTES; // stiefel rows
        size += Integer.BYTES; // stiefel cols
        size += Float.BYTES * quantizedDim * originalDimension;
        return size;
    }

    private static float roundToFloat16(float value) {
        return halfBitsToFloat(floatToHalfBits(value));
    }

    private static void writeFloat16(IndexWriter out, float value) throws IOException {
        int bits = floatToHalfBits(value);

        // Explicit big-endian byte order, matching Java DataOutput-style primitives.
        out.writeByte((byte) ((bits >>> 8) & 0xFF));
        out.writeByte((byte) (bits & 0xFF));
    }

    private static float readFloat16(RandomAccessReader in) throws IOException {
        int hi = readUnsignedByte(in);
        int lo = readUnsignedByte(in);
        return halfBitsToFloat((hi << 8) | lo);
    }

    private static int readUnsignedByte(RandomAccessReader in) throws IOException {
        if (in instanceof io.github.jbellis.jvector.disk.ByteReadable) {
            return ((io.github.jbellis.jvector.disk.ByteReadable) in).readByte() & 0xFF;
        }

        byte[] one = ONE_BYTE.get();
        in.readFully(one);
        return one[0] & 0xFF;
    }

    private static final ThreadLocal<byte[]> ONE_BYTE =
            ThreadLocal.withInitial(() -> new byte[1]);

    /**
     * IEEE-754 binary16 round-to-nearest-even conversion.
     */
    private static int floatToHalfBits(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int abs = bits & 0x7FFF_FFFF;

        // NaN / infinity
        if (abs >= 0x7F80_0000) {
            if ((abs & 0x007F_FFFF) == 0) {
                return sign | 0x7C00;
            }
            return sign | 0x7E00; // canonical quiet NaN
        }

        int exp = ((abs >>> 23) & 0xFF) - 127 + 15;
        int mant = abs & 0x007F_FFFF;

        // Overflow to infinity.
        if (exp >= 31) {
            return sign | 0x7C00;
        }

        // Subnormal / underflow.
        if (exp <= 0) {
            if (exp < -10) {
                return sign;
            }

            mant |= 0x0080_0000;
            int shift = 14 - exp;
            int halfMant = mant >>> shift;

            int roundBit = 1 << (shift - 1);
            int remainder = mant & (roundBit - 1);

            if ((mant & roundBit) != 0 && (remainder != 0 || (halfMant & 1) != 0)) {
                halfMant++;
            }

            return sign | halfMant;
        }

        // Normalized.
        int halfMant = mant >>> 13;
        int roundBit = 0x0000_1000;
        int remainder = mant & (roundBit - 1);

        if ((mant & roundBit) != 0 && (remainder != 0 || (halfMant & 1) != 0)) {
            halfMant++;

            if (halfMant == 0x0400) {
                halfMant = 0;
                exp++;

                if (exp >= 31) {
                    return sign | 0x7C00;
                }
            }
        }

        return sign | (exp << 10) | halfMant;
    }

    private static float halfBitsToFloat(int half) {
        int h = half & 0xFFFF;
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1F;
        int mant = h & 0x03FF;

        if (exp == 0) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign);
            }

            // Normalize half subnormal.
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }

            exp++;
            mant &= 0x03FF;
        } else if (exp == 31) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (mant << 13));
        }

        int floatExp = exp + (127 - 15);
        int bits = sign | (floatExp << 23) | (mant << 13);
        return Float.intBitsToFloat(bits);
    }

    @Override
    public int compressedVectorSize() {
        return QuantizedVector.serializedSizeBytes(quantizedDim, bitsPerDimension);
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
    private static int vectorHash(VectorFloat<?> v) {
        if (v == null) return 0;
        int h = 1;
        for (int i = 0; i < v.length(); i++) {
            h = 31 * h + Integer.hashCode(Float.floatToIntBits(v.get(i)));
        }
        return h;
    }

    // Exact comparison of two RealMatrix instances by dimensions and entries.
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
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        AsymmetricHashing that = (AsymmetricHashing) o;
        return originalDimension == that.originalDimension
                && encodedBits == that.encodedBits
                && bodyBits == that.bodyBits
                && quantizedDim == that.quantizedDim
                && bitsPerDimension == that.bitsPerDimension
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
     */
    @Override
    public int hashCode() {
        int result = Integer.hashCode(originalDimension);
        result = 31 * result + Integer.hashCode(encodedBits);
        result = 31 * result + Integer.hashCode(bodyBits);
        result = 31 * result + Integer.hashCode(quantizedDim);
        result = 31 * result + Integer.hashCode(bitsPerDimension);
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
            default:
                optimizerName = "UNKNOWN(" + optimizer + ")";
        }

        return String.format(
                "AsymmetricHashing[origDim=%d, encodedBits=%d, bodyBits=%d, quantizedDim=%d, bitsPerDimension=%d, optimizer=%s, landmarks=%d]",
                originalDimension,
                encodedBits,
                bodyBits,
                quantizedDim,
                bitsPerDimension,
                optimizerName,
                landmarkCount
        );
    }
}
