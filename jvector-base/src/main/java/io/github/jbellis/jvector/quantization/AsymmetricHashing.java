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
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.CallerBatchProcessor;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import dev.ludovic.netlib.blas.BLAS;
import dev.ludovic.netlib.lapack.LAPACK;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.Random;
import java.util.stream.IntStream;
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
    private static final int ITQ_TRAINING_FACTOR = 20;

    /** Shared by sample selection and the existing random projection initialization. */
    private static final long TRAINING_SEED = 103;

    /** Bounds transpose scratch while keeping writes to the training matrix contiguous. */
    private static final int TRAINING_TRANSPOSE_TILE_ROWS = 32;

    /** Final setting TBD. */
    private static final int TRAINING_ITERS = 25;

    /** Default number of stored bits per projected dimension. */
    private static final int DEFAULT_BITS_PER_DIMENSION = 2;

    // Physical header size, reflecting actual stored fields:
    //  - scale: fp16 (16 bits), where scale = ||x − μ|| / ||code||
    //  - offset: fp16 (16 bits), where offset = <x, μ> − ||μ||_2^2
    //  - landmark id: byte (8 bits) in [0, C)
    public static final int HEADER_BYTES = Short.BYTES + Short.BYTES + Byte.BYTES;
    public static final int HEADER_BITS = HEADER_BYTES * Byte.SIZE;

    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport vectorUtilSupport =
            VectorizationProvider.getInstance().getVectorUtilSupport();

    // ---------------------------------------------------------------------
    // Index-wide immutable state
    // ---------------------------------------------------------------------

    /** Number of landmarks (clusters), {@code C <= 256}. */
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
     *   landmarkProj[c][j] = {@code <A[j], μ_c>}  where A is [d][D] and μ_c is length D.
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

    // Immutable retained Java storage; native/query representations are charged lazily.
    private final long heapBytes;

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

        if (landmarkCount < 1 || landmarkCount > 256) {
            throw new IllegalArgumentException(
                    "landmarkCount must be in [1,256], got " + landmarkCount);
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
        long bytes = RamUsageEstimator.shallowSizeOfInstance(AsymmetricHashing.class)
                + RamUsageEstimator.shallowSizeOf(landmarks)
                + RamUsageEstimator.sizeOf(landmarkNormSq)
                + estimateArray2DBytes(landmarkProj)
                + RamUsageEstimator.sizeOf(landmarkProjSum)
                + stiefelTransform.heapBytes
                + RamUsageEstimator.shallowSizeOf(randomQuantizer)
                + RamUsageEstimator.shallowSizeOf(itqQuantizer)
                + RamUsageEstimator.shallowSizeOf(((ItqBinaryQuantizer) itqQuantizer).delegate);
        for (VectorFloat<?> landmark : landmarks) bytes = Math.addExact(bytes, landmark.ramBytesUsed());
        this.heapBytes = bytes;
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
     * <p>With one landmark, the mean and learned projection use the same
     * deterministic sample of {@code min(ravv.size(), 20 * ravv.dimension())}
     * vectors. Training fetches each sampled vector once and leaves the full
     * dataset pass to encoding. The sample size uses the original dimension,
     * independently of the projected dimension and stored bit width.</p>
     *
     * @param ravv the vectors to quantize
     * @param optimizer the optimizer to use
     * @param encodedBits the number of bits used to encode vector, including the header
     * @param landmarkCount number of landmarks
     * @param bitsPerDimension number of stored bits per projected dimension
     * @throws IllegalArgumentException if the configuration is invalid, the source
     *         is empty, or a sampled vector has an invalid dimension or non-finite norm
     * @throws NullPointerException if the source or a sampled vector is null
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount,
                                               int bitsPerDimension) throws IOException {
        validateBitsPerDimension(bitsPerDimension);

        if (landmarkCount < 1 || landmarkCount > 256) {
            throw new IllegalArgumentException(
                    "landmarkCount must be in [1,256], got " + landmarkCount);
        }

        Objects.requireNonNull(ravv, "ravv");
        if (ravv.size() <= 0) {
            throw new IllegalArgumentException("ASH training requires at least one vector");
        }
        if (optimizer != RANDOM && optimizer != ITQ) {
            throw new IllegalArgumentException("Unknown optimizer " + optimizer);
        }
        int originalDim = ravv.dimension();
        if (originalDim <= 0) {
            throw new IllegalArgumentException("ASH training requires a positive vector dimension");
        }
        final int quantizedDim = validateEncodedBits(encodedBits, HEADER_BITS, originalDim, bitsPerDimension);
        var ravvCopy = ravv.threadLocalSupplier().get();

        System.out.println(
                "\tASH initialized with landmarkCount=" + landmarkCount +
                        ", quantizedDim=" + quantizedDim +
                        ", bitsPerDimension=" + bitsPerDimension
        );

        // C=1 uses the same sample for its mean and learned projection.
        // Sample selection preserves ITQ's existing RNG sequence.
        final Random rng = new Random(TRAINING_SEED);
        VectorFloat<?>[] landmarks;
        TrainingData trainingData = null;
        if (landmarkCount == 1) {
            SingleLandmarkTraining prepared = prepareSingleLandmarkTraining(
                    ravvCopy, originalDim, rng, optimizer == ITQ);
            landmarks = new VectorFloat<?>[]{prepared.landmark};
            trainingData = prepared.trainingData;
        } else {
            // Multi-landmark clustering still needs its materialized input.
            VectorFloat<?>[] points = new VectorFloat<?>[ravvCopy.size()];
            for (int i = 0; i < points.length; i++) {
                VectorFloat<?> v = ravvCopy.getVector(i).copy();
                float norm = (float) Math.sqrt(VectorUtil.dotProduct(v, v));
                if (norm > 1e-10f) VectorUtil.scale(v, 1.0f / norm);
                points[i] = v;
            }
            long kmStart = System.nanoTime();
            MiniBatchKMeansClusterer km = new MiniBatchKMeansClusterer(points, landmarkCount, 1024);
            VectorFloat<?> packed = km.cluster(/*unweightedIterations*/ 20, /*anisotropicIterations*/ 0);
            long kmEnd = System.nanoTime();
            System.out.printf("\tKMeans++ (C=%d, N=%d, D=%d) took %.3f seconds%n",
                    landmarkCount, points.length, originalDim, (kmEnd - kmStart) / 1e9);
            landmarks = new VectorFloat<?>[landmarkCount];
            for (int c = 0; c < landmarkCount; c++) {
                VectorFloat<?> mu = vectorTypeSupport.createFloatVector(originalDim);
                mu.copyFrom(packed, c * originalDim, 0, originalDim);
                landmarks[c] = mu;
            }
            if (optimizer == ITQ) trainingData = prepareTrainingData(points, landmarks, originalDim, rng);
        }

        StiefelTransform stiefelTransform = optimizer == RANDOM
                ? runWithoutTraining(originalDim, quantizedDim, rng)
                : runItqTrainer(trainingData, quantizedDim, bitsPerDimension, rng, TRAINING_ITERS);

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
     *   landmarkProj[c][j] = {@code <A[j], μ_c>}
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
        if (landmarkCount < 1 || landmarkCount > 256) {
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

        // Landmark-specific mean (C=1 → landmarks[0] is the sampled mean)
        final VectorFloat<?> mu = landmarks[landmark & 0xFF];

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
        return flatNibble(code, 0, group);
    }

    /** Nibble {@code group} of the code starting at {@code codeOffset}. */
    static int flatNibble(byte[] code, int codeOffset, int group) {
        int v = code[codeOffset + (group >>> 1)] & 0xFF;
        return ((group & 1) == 0) ? (v & 0x0F) : ((v >>> 4) & 0x0F);
    }

    static int projectionFieldForDimension(
            byte[] projectionCode,
            int dimension,
            int bitsPerDimension
    ) {
        return projectionFieldForDimension(projectionCode, 0, dimension, bitsPerDimension);
    }

    static int projectionFieldForDimension(byte[] projectionCode, int codeOffset, int dimension, int bitsPerDimension) {
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int group = dimension / groupDims;
        int slot = dimension - group * groupDims;
        int mask = (1 << bitsPerDimension) - 1;
        return (flatNibble(projectionCode, codeOffset, group) >>> (slot * bitsPerDimension)) & mask;
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
        return projectionComponent(projectionCode, 0, dimension, bitsPerDimension);
    }

    /** Component {@code dimension} of the code starting at {@code codeOffset}. */
    static float projectionComponent(byte[] projectionCode, int codeOffset, int dimension, int bitsPerDimension) {
        return decodeProjectionComponent(
                projectionFieldForDimension(projectionCode, codeOffset, dimension, bitsPerDimension),
                bitsPerDimension);
    }

    static float dotProjectionCode(
            float[] projectedVector,
            byte[] projectionCode,
            int quantizedDim,
            int bitsPerDimension
    ) {
        return dotProjectionCode(projectedVector, projectionCode, 0, quantizedDim, bitsPerDimension);
    }

    /** Scalar projection dot with the code starting at {@code codeOffset}. */
    static float dotProjectionCode(float[] projectedVector, byte[] projectionCode, int codeOffset,
                                   int quantizedDim, int bitsPerDimension) {
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int groups = projectionCodeGroups(quantizedDim, bitsPerDimension);
        float sum = 0.0f;

        for (int group = 0; group < groups; group++) {
            int nibble = flatNibble(projectionCode, codeOffset, group);
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

    /** Header-calibrated correction for symmetric scoring, using the stored code/header. */
    float centeredOffset(QuantizedVector vector) {
        return centeredOffset(vector.scale, vector.offset, vector.landmark, vector.extraBits, 0);
    }

    /** As {@link #centeredOffset(QuantizedVector)} for a header and a body stored at {@code bodyOffset}. */
    float centeredOffset(float scale, float offset, byte landmark, byte[] body, int bodyOffset) {
        int c = landmark & 255;
        if (c >= landmarkCount) throw new IllegalArgumentException("Invalid landmark " + c);
        if (!usesFastScanProjectionCode(bitsPerDimension)) return offset;
        return offset + scale * dotProjectionCode(landmarkProj[c], body, bodyOffset, quantizedDim, bitsPerDimension);
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
            float[] xhat;
            float[] proj;
            double[] absNorm;

            void ensureCapacity(int originalDim, int quantizedDim) {
                if (xhat == null || xhat.length != originalDim) {
                    xhat = new float[originalDim];
                }
                if (proj == null || proj.length != quantizedDim) {
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

            Workspace ws = WORKSPACE.get();
            ws.ensureCapacity(stiefel.cols, quantizedDim);
            normalizeResidual(vector, mu, residualNorm, ws.xhat);
            // Encoding always uses the original FP32 projection, independently
            // of the model's query-only FP16 view.
            stiefel.project(ws.xhat, ws.proj);
            return quantizeProjected(ws.proj, bitsPerDimension, outSignWords,
                    outExtraBits, ws.absNorm);
        }

        /** Shared code generation for singleton and bulk projections. */
        private static float quantizeProjected(float[] proj, int bitsPerDimension,
                                               long[] outSignWords, byte[] outExtraBits,
                                               double[] absNorm) {
            final int quantizedDim = proj.length;
            if (bitsPerDimension == 1) {
                // Binarize directly from per-row projection.
                final int words = QuantizedVector.wordsForDims(quantizedDim);
                for (int w = 0; w < words; w++) {
                    long bits = 0L;
                    int base = w << 6;
                    int rem = Math.min(64, quantizedDim - base);

                    for (int j = 0; j < rem; j++) {
                        int bitIndex = base + j;
                        float acc = proj[bitIndex];
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
            double projNorm2 = 0.0;
            for (int j = 0; j < quantizedDim; j++) {
                float acc = proj[j];
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

    public static class QuantizedVector implements Accountable {
        private static final long SHALLOW_BYTES = RamUsageEstimator.shallowSizeOfInstance(QuantizedVector.class);

        /** Owned heap storage, including empty arrays and their headers; not the disk size. */
        @Override
        public long ramBytesUsed() {
            return SHALLOW_BYTES + RamUsageEstimator.shallowSizeOf(binaryVector) + RamUsageEstimator.shallowSizeOf(extraBits);
        }

        // Decoded arithmetic values, rounded to binary16 by encoding/loading.
        // They are serialized as two bytes each, not as Java float32 fields.
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
            return HEADER_BYTES
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
            return getBit(words, 0, bitIndex);
        }

        /** Bit {@code bitIndex} of the sign words starting at {@code wordOffset}. */
        static boolean getBit(long[] words, int wordOffset, int bitIndex) {
            return ((words[wordOffset + (bitIndex >>> 6)] >>> (bitIndex & 63)) & 1L) != 0L;
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
            return readExtraCode(extraBits, 0, dim, exBits);
        }

        /** As {@link #readExtraCode(byte[], int, int)} for extra bits starting at {@code byteOffset}. */
        static int readExtraCode(byte[] extraBits, int byteOffset, int dim, int exBits) {
            int code = 0;
            int bitPos = dim * exBits;
            for (int bit = 0; bit < exBits; bit++, bitPos++) {
                code |= ((extraBits[byteOffset + (bitPos >>> 3)] >>> (bitPos & 7)) & 1) << bit;
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
     * Concurrent query projection and caller CPU budgeting are described in
     * {@code docs/ash-query-integration.md}.
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

        // Prepared once per immutable model, including models loaded from disk.
        private final VectorUtilSupport.AshProjection projection;

        // Prepared only when this model is used for queries. Encoding-only work
        // must not allocate an extra matrix or change its arithmetic precision.
        private volatile VectorUtilSupport.AshProjection queryProjection;
        private long queryProjectionHeapBytes;
        private final long heapBytes;

        // Default group capacity. Larger configured groups are supported by
        // grow-once caller scratch; native code controls its own micro-tiling.
        private static final int QUERY_BATCH_SIZE = 8;
        // Published together with queryProjection. Training/encoding never
        // initializes query-only dispatch or its retained storage.
        private CallerBatchProcessor<float[], float[]> queryBatches;
        private final int queryBatchSize;
        private static final class QueryBatchRows {
            float[][] input = new float[QUERY_BATCH_SIZE][];
            float[][] output = new float[QUERY_BATCH_SIZE][];
            void ensure(int count) {
                if (input.length >= count) return;
                float[][] nextInput = new float[count][];
                float[][] nextOutput = new float[count][];
                input = nextInput;
                output = nextOutput;
            }
        }
        // Scratch holds row references only during a batch and never a model.
        private static final ThreadLocal<QueryBatchRows> QUERY_BATCH_ROWS =
                ThreadLocal.withInitial(QueryBatchRows::new);

        /** Projects a search query; encoding and training continue through project(). */
        public void projectQuery(float[] query, float[] out) {
            var projection = queryProjection(); // Acquire the matching dispatcher too.
            var batches = queryBatches;
            if (batches == null || !projection.supportsBatchProjection()) projection.project(query, out);
            else batches.process(query, out);
        }

        /** Prepared-query callers already own scheduling; avoid a second combining layer. */
        public void projectQueryDirect(float[] query, float[] out) {
            queryProjection().project(query, out);
        }

        /** Executes a caller-formed batch with the query precision, never the encoding view. */
        public void projectQueryBatch(float[][] queries, float[][] out, int count) {
            queryProjection().projectBatch(queries, out, count);
        }

        /** Capability inspection may lazily prepare the immutable query representation. */
        public boolean supportsQueryBatchProjection() {
            return queryProjection().supportsBatchProjection();
        }

        /** Snapshot of completed/claimed batch counts, or null when combining is disabled. */
        public long[] queryProjectionBatchHistogram() {
            var projection = queryProjection; // Inspection must not prepare a query view.
            return projection == null || queryBatches == null ? null : queryBatches.batchHistogram();
        }

        /** Reports the prepared view without initializing or changing query execution. */
        public String initializedQueryProjectionPrecision() {
            var ready = queryProjection;
            return ready == null ? null : ready.operandPrecision();
        }

        private VectorUtilSupport.AshProjection queryProjection() {
            var ready = queryProjection;
            if (ready == null) {
                synchronized (this) {
                    ready = queryProjection;
                    if (ready == null) {
                        ready = projection.forQueryScoring();
                        long retained = ready == projection ? 0 : RamUsageEstimator.shallowSizeOf(ready);
                        if (queryBatchSize > 1 && ready.supportsBatchProjection()) {
                            final var target = ready;
                            var operation = new CallerBatchProcessor.Operation<float[], float[]>() {
                                @Override
                                public void apply(float[] input, float[] output) { target.project(input, output); }
                                @Override
                                public void applyBatch(List<float[]> inputs, List<float[]> outputs) {
                                    QueryBatchRows rows = QUERY_BATCH_ROWS.get();
                                    int count = inputs.size();
                                    rows.ensure(count);
                                    try {
                                        for (int i = 0; i < count; i++) {
                                            rows.input[i] = inputs.get(i);
                                            rows.output[i] = outputs.get(i);
                                        }
                                        target.projectBatch(rows.input, rows.output, count);
                                    } finally {
                                        Arrays.fill(rows.input, 0, count, null);
                                        Arrays.fill(rows.output, 0, count, null);
                                    }
                                }
                            };
                            int budget = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(),
                                    PhysicalCoreExecutor.getPhysicalCoreCount()));
                            queryBatches = CallerBatchProcessor.opportunistic(
                                    budget, queryBatchSize, operation);
                            retained += queryBatches.ramBytesUsed() + RamUsageEstimator.shallowSizeOf(operation);
                        }
                        queryProjectionHeapBytes = retained;
                        queryProjection = ready; // Publish the representation and its heap charge together.
                    }
                }
            }
            return ready;
        }

        /** Projects one vector with the original FP32 representation, including encoding/training. */
        public void project(float[] query, float[] out) {
            projection.project(query, out);
        }

        /** Projects base vectors in FP32; query precision selection does not apply. */
        public void projectBatch(float[][] vectors, float[][] out, int count) {
            projection.projectBatch(vectors, out, count);
        }

        /** Number of rows in A (quantizedDim = d). */
        public final int rows;

        /** Number of columns in A (originalDim = D). */
        public final int cols;

        public StiefelTransform(RealMatrix W) {
            this.W = Objects.requireNonNull(W, "W");
            String batchSize = System.getProperty("jvector.ash.queryBatchSize", "8");
            try { queryBatchSize = Integer.parseInt(batchSize); }
            catch (NumberFormatException invalid) {
                throw new IllegalArgumentException("jvector.ash.queryBatchSize must be a positive integer: " + batchSize, invalid);
            }
            if (queryBatchSize < 1 || queryBatchSize == Integer.MAX_VALUE)
                throw new IllegalArgumentException("jvector.ash.queryBatchSize must be between 1 and " + (Integer.MAX_VALUE - 1));

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
            this.projection = VectorizationProvider.getInstance().getVectorUtilSupport()
                    .prepareAshProjection(AFloat);

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
            heapBytes = RamUsageEstimator.shallowSizeOfInstance(StiefelTransform.class)
                    + estimateMatrixBytes(W) + estimateArray2DBytes(AData) + estimateArray2DBytes(AFloat)
                    + RamUsageEstimator.shallowSizeOf(projection);
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
     * @throws IllegalArgumentException if {@code d <= 0 or d > D}
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


    // An internal working-set bound, not a projection API limit. Each task owns
    // at most this many normalized vectors and projected results, reused until
    // its range is complete. Native code supplies the SIMD/cache micro-tiling.
    private static final int ENCODING_BATCH_SIZE = 32;

    /**
     * Encodes all base vectors in bounded FP32 projection batches. The supplied
     * pool controls parallelism; native calls execute on the calling worker.
     * Missing vectors retain the existing all-zero encoded representation.
     */
    @Override
    public ASHVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        Objects.requireNonNull(ravv, "ravv");
        Objects.requireNonNull(simdExecutor, "simdExecutor");
        if (ravv.dimension() != originalDimension)
            throw new IllegalArgumentException("ASH encoding vector dimension mismatch");
        final int n = ravv.size();
        final int parallelism = Math.max(1, simdExecutor.getParallelism());
        final int chunkSize = (int) (((long) n + parallelism - 1) / parallelism);
        // Encoded straight into the compact store; no per-vector objects are retained.
        final ASHCodes out = ASHCodes.allocate(n, quantizedDim, bitsPerDimension);
        // Fill while each encoded vector is hot; never reread the raw input for setup.
        final float[] symmetricOffsets = usesFastScanProjectionCode(bitsPerDimension) ? new float[n] : null;
        final var ravvSupplier = ravv.threadLocalSupplier();
        var tasks = new java.util.ArrayList<java.util.concurrent.ForkJoinTask<?>>(parallelism);
        Throwable failure = null;
        try {
            for (int start = 0; start < n;) {
                final int first = start;
                final int end = (int) Math.min((long) n, (long) first + chunkSize);
                tasks.add(simdExecutor.submit(() -> encodeRange(ravvSupplier.get(), out, symmetricOffsets, first, end)));
                start = end;
            }
        } catch (RuntimeException | Error error) {
            failure = error;
        }
        // Await every submitted task, including when the pool rejected a later
        // submission. The caller may close its source when this method exits.
        for (var task : tasks) {
            try { task.join(); }
            catch (RuntimeException | Error error) {
                if (failure == null) failure = error;
                else if (failure != error) failure.addSuppressed(error);
            }
        }
        if (failure instanceof Error) throw (Error) failure;
        if (failure != null) throw (RuntimeException) failure;
        return new ASHVectors(this, out, symmetricOffsets);
    }

    private void encodeRange(RandomAccessVectorValues source, ASHCodes out,
                             float[] symmetricOffsets, int start, int end) {
        int capacity = Math.min(ENCODING_BATCH_SIZE, end - start);
        float[][] inputs = new float[capacity][originalDimension];
        float[][] projected = new float[capacity][quantizedDim];
        float[] residualNorms = new float[capacity];
        int[] ordinals = new int[capacity];
        double[] absNorm = new double[quantizedDim];
        // Per-batch scratch, copied into the store once complete. Never modified: the code for missing vectors.
        QuantizedVector[] scratch = new QuantizedVector[capacity];
        for (int k = 0; k < capacity; k++) scratch[k] = QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
        final QuantizedVector missing = QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
        final float missingOffset = symmetricOffsets == null ? 0f : centeredOffset(missing);
        int count = 0;
        try (var cursor = VectorAccess.openRange(source, start, end)) {
            int expected = start;
            while (cursor.next()) {
                int i = cursor.ordinal();
                if (i != expected++) throw new IllegalStateException("Unexpected encoding ordinal");
                VectorFloat<?> vector = cursor.vector();
                if (vector == null) {
                    out.set(i, missing);
                    if (symmetricOffsets != null) symmetricOffsets[i] = missingOffset;
                    continue;
                }
                QuantizedVector dest = scratch[count];
                float residualNorm = prepareEncoding(vector, dest);
                normalizeResidual(vector, landmarks[dest.landmark & 0xFF], residualNorm, inputs[count]);
                residualNorms[count] = residualNorm;
                ordinals[count] = i;
                if (++count == capacity) {
                    encodeBatch(inputs, projected, residualNorms, ordinals, count, absNorm, scratch, out, symmetricOffsets);
                    count = 0;
                }
            }
            if (expected != end) throw new IllegalStateException("Incomplete encoding range");
        }
        if (count != 0) encodeBatch(inputs, projected, residualNorms, ordinals, count, absNorm, scratch, out, symmetricOffsets);
    }

    private void encodeBatch(float[][] inputs, float[][] projected, float[] residualNorms,
                             int[] ordinals, int count, double[] absNorm, QuantizedVector[] scratch,
                             ASHCodes out, float[] symmetricOffsets) {
        stiefelTransform.projectBatch(inputs, projected, count);
        for (int i = 0; i < count; i++) {
            QuantizedVector dest = scratch[i];
            // Quantizers OR bits and nibbles into place, so reused scratch starts from zero.
            Arrays.fill(dest.binaryVector, 0L);
            Arrays.fill(dest.extraBits, (byte) 0);
            float codeNorm = RandomBinaryQuantizer.quantizeProjected(projected[i], bitsPerDimension,
                    dest.binaryVector, dest.extraBits, absNorm);
            float correction = finishEncoding(dest, residualNorms[i], codeNorm);
            out.set(ordinals[i], dest);
            if (symmetricOffsets != null) symmetricOffsets[ordinals[i]] = correction;
        }
    }

    private static void normalizeResidual(VectorFloat<?> vector, VectorFloat<?> mean,
                                          float residualNorm, float[] normalized) {
        if (vector.length() != normalized.length || mean.length() != normalized.length)
            throw new IllegalArgumentException("ASH encoding vector dimension mismatch");
        float invNorm = residualNorm > 0f ? 1f / residualNorm : 0f;
        if (vector instanceof FloatArray
                && mean instanceof FloatArray) {
            // Resolve heap backing once, including MemorySegment views. The
            // simple array loop avoids per-component virtual/scope checks and
            // permits JVM SIMD without changing the FP32 operation order.
            float[] values = ((FloatArray) vector).array();
            float[] center = ((FloatArray) mean).array();
            for (int d = 0; d < normalized.length; d++)
                normalized[d] = (values[d] - center[d]) * invNorm;
        } else {
            for (int d = 0; d < normalized.length; d++)
                normalized[d] = (vector.get(d) - mean.get(d)) * invNorm;
        }
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
        float residualNorm = prepareEncoding(vector, dest);
        float codeNorm = quantizeVector(vector, residualNorm, dest.landmark, dest);
        finishEncoding(dest, residualNorm, codeNorm);
    }

    /** Sets landmark and raw offset; returns the residual norm needed by both paths. */
    private float prepareEncoding(VectorFloat<?> vector, QuantizedVector dest) {
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(dest, "dest");
        if (vector.length() != originalDimension)
            throw new IllegalArgumentException("ASH encoding vector dimension mismatch");
        if (dest.binaryVector == null || dest.binaryVector.length < QuantizedVector.signWordsForDims(quantizedDim, bitsPerDimension)
                || dest.extraBits == null || dest.extraBits.length < QuantizedVector.bodyBytesForDims(quantizedDim, bitsPerDimension))
            throw new IllegalArgumentException("ASH encoding destination too short");
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
        assert Float.isFinite(rawOffset) : "offset is not finite";

        return residualNorm;
    }

    /** Applies the existing code-dependent header adjustment and FP16 storage rounding. */
    /** Returns the symmetric correction using the same dot product and stored half headers. */
    private float finishEncoding(QuantizedVector dest, float residualNorm, float codeNorm) {
        final int c = dest.landmark & 0xFF;
        final float rawOffset = dest.offset;
        final float scale = codeNorm > 0f ? residualNorm / codeNorm : 0f;

        float finalOffset = rawOffset;
        float landmarkDot = 0f;
        if (usesFastScanProjectionCode(bitsPerDimension)) {
            // C++ projection-mode optimization:
            //   offset = <x, μ> - ||μ||² - scale * <Aμ, code>
            // so query-time scoring can use Aq instead of A(q - μ).
            landmarkDot = dotProjectionCode(
                    landmarkProj[c],
                    dest.extraBits,
                    quantizedDim,
                    bitsPerDimension);
            finalOffset -= scale * landmarkDot;
        }

        dest.scale = roundToFloat16(scale);
        dest.offset = roundToFloat16(finalOffset);
        return usesFastScanProjectionCode(bitsPerDimension)
                ? dest.offset + dest.scale * landmarkDot : dest.offset;
    }

    // ---------------------------------------------------------------------
    // Training of projection matrix
    // ---------------------------------------------------------------------

    /** Sampled, normalized residuals in the column-major layout consumed by LAPACK. */
    private static final class TrainingData {
        final double[] xCol;
        final int rows;
        final int columns;

        TrainingData(double[] xCol, int rows, int columns) {
            this.xCol = xCol;
            this.rows = rows;
            this.columns = columns;
        }

        TrainingData(double[][] rows) {
            this(serializeColumnMajor(rows), rows.length, rows[0].length);
        }
    }

    private static final class SingleLandmarkTraining {
        final VectorFloat<?> landmark;
        final TrainingData trainingData;

        SingleLandmarkTraining(VectorFloat<?> landmark, TrainingData trainingData) {
            this.landmark = landmark;
            this.trainingData = trainingData;
        }
    }

    private static int trainingSampleSize(int count, int dimensions) {
        return (int) Math.min((long) count, Math.max(1L, (long) dimensions * ITQ_TRAINING_FACTOR));
    }

    /**
     * Computes the C=1 landmark and ITQ inputs from the same deterministic
     * sample. Only sampled source vectors are read; normalized sample values
     * are retained until the mean is known. One reusable vector protects
     * borrowed source storage. Double sums follow sorted ordinal order, and
     * normalization retains the provider's FP32 dot/scale operations.
     */
    private static SingleLandmarkTraining prepareSingleLandmarkTraining(
            RandomAccessVectorValues source, int dimensions, Random rng, boolean learn) {
        final int count = source.size();
        if (count <= 0 || dimensions <= 0) {
            throw new IllegalArgumentException("ASH training requires positive size and dimension");
        }
        final int sampleSize = trainingSampleSize(count, dimensions);
        // Validate the flat-array representation before allocating ordinal or
        // normalization buffers.
        long sampleElements = (long) sampleSize * dimensions;
        if (learn && sampleElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("ASH training sample exceeds the Java array size limit: "
                    + sampleSize + " rows x " + dimensions + " dimensions");
        }
        // The existing PCA driver also requires a full D-by-D right basis,
        // including null-space directions when the training matrix is wide.
        if (learn && (long) dimensions * dimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("ASH PCA basis exceeds the Java array size limit: "
                    + dimensions + " x " + dimensions);
        }
        // RANDOM projections keep their existing Gaussian RNG sequence. ITQ
        // consumes exactly the same reservoir-sampling draws as before.
        final Random sampleRandom = learn ? rng : new Random(TRAINING_SEED);
        final int[] ordinals = reservoirSampleOrdinals(count, sampleSize, sampleRandom);
        Arrays.sort(ordinals);
        final float[] sample = learn ? new float[(int) sampleElements] : null;
        final double[] accumulator = new double[dimensions];
        final VectorFloat<?> normalized = vectorTypeSupport.createFloatVector(dimensions);
        final float[] values = normalized instanceof FloatArray ? ((FloatArray) normalized).array() : null;
        for (int row = 0; row < sampleSize; row++) {
            int ordinal = ordinals[row];
            VectorFloat<?> input = source.getVector(ordinal);
            if (input == null) {
                throw new NullPointerException("Null ASH training vector at ordinal " + ordinal);
            }
            if (input.length() != dimensions) {
                throw new IllegalArgumentException("ASH training vector at ordinal " + ordinal
                        + " has dimension " + input.length() + "; expected " + dimensions);
            }
            normalized.copyFrom(input, 0, 0, dimensions);
            // Dimensions are already checked. Call the provider directly so invalid
            // norms reach argument validation even when VectorUtil assertions are enabled.
            float norm = (float) Math.sqrt(vectorUtilSupport.dotProduct(normalized, normalized));
            if (!Float.isFinite(norm)) {
                throw new IllegalArgumentException("ASH training vector at ordinal " + ordinal
                        + " has a non-finite FP32 L2 norm");
            }
            if (norm > 1e-10f) {
                VectorUtil.scale(normalized, 1.0f / norm);
            }
            if (values != null) {
                for (int d = 0; d < dimensions; d++) accumulator[d] += values[d];
            } else {
                for (int d = 0; d < dimensions; d++) accumulator[d] += normalized.get(d);
            }
            if (learn) {
                int offset = row * dimensions;
                if (values != null) {
                    System.arraycopy(values, 0, sample, offset, dimensions);
                } else {
                    for (int d = 0; d < dimensions; d++) sample[offset + d] = normalized.get(d);
                }
            }
        }
        VectorFloat<?> mean = vectorTypeSupport.createFloatVector(dimensions);
        double invCount = 1.0 / sampleSize;
        for (int d = 0; d < dimensions; d++) mean.set(d, (float) (accumulator[d] * invCount));
        if (!learn) return new SingleLandmarkTraining(mean, null);
        return new SingleLandmarkTraining(mean, centerTrainingSample(sample, sampleSize, dimensions, mean));
    }

    /**
     * Centers/normalizes contiguous sample rows in FP64, then writes small tiles
     * into the final column-major matrix. No full-size double row matrix or
     * strided, element-at-a-time whole-matrix conversion is needed. Tile width
     * bounds scratch only; it does not change arithmetic or sample boundaries.
     */
    private static TrainingData centerTrainingSample(float[] sample, int rows, int dimensions,
                                                     VectorFloat<?> mean) {
        final int tileRows = Math.min(TRAINING_TRANSPOSE_TILE_ROWS, rows);
        double[] tile = new double[Math.multiplyExact(tileRows, dimensions)];
        double[] columnMajor = new double[Math.multiplyExact(rows, dimensions)];
        float[] center = new float[dimensions];
        for (int d = 0; d < dimensions; d++) center[d] = mean.get(d);
        for (int first = 0; first < rows; first += tileRows) {
            int rowsInTile = Math.min(tileRows, rows - first);
            for (int row = 0; row < rowsInTile; row++) {
                int sampleOffset = (first + row) * dimensions;
                int tileOffset = row * dimensions;
                double norm2 = 0.0;
                for (int d = 0; d < dimensions; d++) {
                    double value = (double) sample[sampleOffset + d] - (double) center[d];
                    tile[tileOffset + d] = value;
                    norm2 += value * value;
                }
                double inverse = 1.0 / Math.max(Math.sqrt(norm2), 1e-20);
                for (int d = 0; d < dimensions; d++) tile[tileOffset + d] *= inverse;
            }
            for (int d = 0; d < dimensions; d++) {
                int columnOffset = d * rows + first;
                for (int row = 0; row < rowsInTile; row++) {
                    columnMajor[columnOffset + row] = tile[row * dimensions + d];
                }
            }
        }
        return new TrainingData(columnMajor, rows, dimensions);
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

        final int nTrain = trainingSampleSize(N, originalDim);

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
     *  - PCA basis P from checked FP64 covariance PCA, with direct SVD fallback
     *  - Xld = X @ P
     *  - iterative Procrustes:
     *      R = polar(M)
     *      Xtr = Xld @ R
     *      Xenc = encodeTrainingDirections(Xtr, bitsPerDimension)
     *      M = Xld^T @ Xenc
     *  - W = P @ R
     */
    private static StiefelTransform runItqTrainer(
            TrainingData trainingData, // [N x D] normalized residuals, column-major
            int latentDim,            // projected dimensions, d
            int bitsPerDimension,
            Random rng,
            int nTrainingIterations
    ) {
        logProgress("\t[stage] Starting runITQTrainer...");

        validateBitsPerDimension(bitsPerDimension);

        final int N = trainingData.rows;
        final int D = trainingData.columns;
        final int d = latentDim;

        if (d <= 0 || d > D) {
            throw new IllegalArgumentException("Invalid latentDim=" + d + " for D=" + D);
        }

        // X is [N x D], flat column-major for BLAS/LAPACK.
        double[] xCol = trainingData.xCol;

        logProgress("\t[stage] Starting native PCA...");
        double[] pCol = computePcaBasis(xCol, N, D, d);
        logProgress("\t[stage] Completed native PCA...");

        // Xld = X @ P, shape [N x d].
        double[] xldCol = nativeMultiply(xCol, N, D, pCol, d);

        // Initial random update matrix M, shape [d x d].
        double[] mCol = gaussianMatrix(d, d, rng);

        // Each projected vector is contiguous. Quantization replaces it in place:
        // the loss uses the update matrix, so unquantized rows need not be retained.
        double[] projectedRows = new double[Math.multiplyExact(N, d)];

        logProgress("\t[stage] ITQ training iterations started...");
        long startTime = System.nanoTime();
        // Repeated polar steps need FP32 accuracy; the accumulated update,
        // projection products and loss retain FP64. Final W uses a FP64 polar
        // factor so the stored transform keeps its orthogonality contract.
        ASHFloatOrthogonalizer orthogonalizer = new ASHFloatOrthogonalizer(d, d);
        double[] rCol = new double[Math.multiplyExact(d, d)];
        TrainingDirectionEncoder encoder = new TrainingDirectionEncoder(N, d, bitsPerDimension);

        double[] recentLosses = new double[3];
        for (int epoch = 0; epoch < nTrainingIterations; epoch++) {
            orthogonalizer.orthogonalize(mCol, rCol);

            projectTrainingRows(xldCol, N, d, rCol, projectedRows);
            encoder.encodeInPlace(projectedRows);

            // ITQ update: M = Xld^T @ Xenc. Reuse the initial Gaussian buffer.
            updateTrainingMatrix(xldCol, N, d, projectedRows, mCol);

            // C++-style cheap loss: trace(R^T M) / N.
            ProjectionTrainingLoss loss = computeTrainingLossFromUpdateMatrix(rCol, mCol, N);
            printTrainingLoss(epoch, loss);
            if (epoch >= 8 && recentLosses[(epoch - 3) % 3] - loss.loss <= 0.01 * loss.loss) {
                logProgress("ITQ early stop at epoch " + epoch);
                break;
            }
            recentLosses[epoch % 3] = loss.loss;
        }

        // Recompute from the final update, in FP64, rather than retaining the
        // previous epoch's rotation or widening a FP32 polar factor.
        new ASHOrthogonalizer(d, d).orthogonalize(mCol, rCol);

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

    /**
     * Produces contiguous projected vectors without a separate transpose pass.
     * Xld is [rows x dims] column-major; output rows are its product with R,
     * stored as [dims x rows] column-major: output = R^T * Xld^T.
     */
    private static void projectTrainingRows(double[] xldCol, int rows, int dims,
                                            double[] rCol, double[] outRows) {
        checkTrainingProduct(xldCol, rows, dims, rCol, outRows);
        if (outRows == xldCol || outRows == rCol) {
            throw new IllegalArgumentException("ITQ projection output must not alias an input");
        }
        BLAS.getInstance().dgemm("T", "T", dims, rows, dims, 1.0,
                rCol, dims, xldCol, rows, 0.0, outRows, dims);
    }

    /**
     * Recomputes M = Xld^T * Xenc into an existing square matrix. Encoded rows
     * are viewed as [dims x rows] column-major, hence the second transpose.
     */
    private static void updateTrainingMatrix(double[] xldCol, int rows, int dims,
                                             double[] encodedRows, double[] mCol) {
        checkTrainingProduct(xldCol, rows, dims, mCol, encodedRows);
        if (mCol == xldCol || mCol == encodedRows) {
            throw new IllegalArgumentException("ITQ update output must not alias an input");
        }
        BLAS.getInstance().dgemm("T", "T", dims, dims, rows, 1.0,
                xldCol, rows, encodedRows, dims, 0.0, mCol, dims);
    }

    private static void checkTrainingProduct(double[] xldCol, int rows, int dims,
                                              double[] square, double[] rowMajor) {
        Objects.requireNonNull(xldCol, "xldCol");
        Objects.requireNonNull(square, "square");
        Objects.requireNonNull(rowMajor, "rowMajor");
        if (rows <= 0 || dims <= 0 || (long) rows * dims > Integer.MAX_VALUE
                || (long) dims * dims > Integer.MAX_VALUE
                || xldCol.length != rows * dims || rowMajor.length != rows * dims
                || square.length != dims * dims) {
            throw new IllegalArgumentException("Invalid ITQ matrix dimensions");
        }
    }

    /**
     * Computes all right singular vectors with the divide-and-conquer LAPACK driver.
     * Input X is [rows x cols] in column-major order; the returned VT is [cols x cols].
     *
     * <p>For tall matrices, overwrite the scratch input with the unused left singular
     * vectors instead of allocating another rows-by-cols array. Wide matrices need
     * the full VT, including its null-space basis, because the requested projection
     * dimension can exceed the number of training rows.</p>
     *
     * <p>Check the singular-value energy before accepting the basis. A native backend
     * can return success with an inaccurate decomposition; its status alone does not
     * establish that the PCA input was decomposed correctly.</p>
     */
    private static double[] computeNativeVt(double[] xCol, int rows, int cols) {
        if (rows <= 0 || cols <= 0 || xCol.length != Math.multiplyExact(rows, cols)) {
            throw new IllegalArgumentException("Invalid PCA matrix dimensions");
        }
        double inputEnergy = 0.0;
        for (double value : xCol) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("PCA training input contains a non-finite value");
            }
            inputEnergy += value * value;
        }

        int k = Math.min(rows, cols);
        boolean tall = rows >= cols;
        String job = tall ? "O" : "A";
        int ldu = tall ? 1 : rows;
        double[] u = new double[tall ? 1 : Math.multiplyExact(rows, rows)];
        double[] s = new double[k];
        double[] vt = new double[Math.multiplyExact(cols, cols)];
        int[] iwork = new int[Math.multiplyExact(8, k)];
        intW info = new intW(0);
        double[] workQuery = new double[1];
        // LAPACK's workspace query does not modify input matrix values.
        LAPACK.getInstance().dgesdd(job, rows, cols, xCol, rows, s, u, ldu,
                vt, cols, workQuery, -1, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("PCA SVD workspace query failed with info=" + info.val);
        }
        if (!Double.isFinite(workQuery[0]) || workQuery[0] < 1 || workQuery[0] > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid PCA SVD workspace size: " + workQuery[0]);
        }
        double[] work = new double[(int) Math.ceil(workQuery[0])];
        double[] a = xCol.clone(); // The actual decomposition overwrites its input.
        LAPACK.getInstance().dgesdd(job, rows, cols, a, rows, s, u, ldu,
                vt, cols, work, work.length, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("PCA SVD failed with info=" + info.val);
        }

        double singularEnergy = 0.0;
        for (double value : s) {
            if (!Double.isFinite(value) || value < 0.0) {
                throw new IllegalStateException("PCA SVD returned an invalid singular value");
            }
            singularEnergy += value * value;
        }
        // Training residuals are normalized; allow accumulation roundoff while rejecting
        // materially incorrect decompositions, including ones reported as successful.
        double tolerance = 1e-8 * Math.max(1.0, inputEnergy);
        if (!Double.isFinite(singularEnergy) || Math.abs(singularEnergy - inputEnergy) > tolerance) {
            throw new IllegalStateException("PCA SVD failed its energy check: input=" + inputEnergy
                    + ", singular values=" + singularEnergy);
        }
        return vt;
    }

    /** Prefer checked covariance PCA; preserve direct SVD for uncertain subspaces. */
    private static double[] computePcaBasis(double[] xCol, int rows, int dims, int latent) {
        double[] basis = ASHCovariancePCA.projection(xCol, rows, dims, latent);
        if (basis != null) {
            logProgress("\t[stage] PCA used FP64 covariance eigendecomposition");
            return basis;
        }
        logProgress("\t[stage] PCA used guarded direct SVD");
        return extractPcaBasis(computeNativeVt(xCol, rows, dims), dims, latent);
    }

    /** Single-use convenience path for random projection initialization. */
    private static double[] orthogonalize(double[] mCol, int rows, int cols) {
        ASHOrthogonalizer workspace = new ASHOrthogonalizer(rows, cols);
        double[] result = new double[Math.multiplyExact(rows, cols)];
        workspace.orthogonalize(mCol, result);
        return result;
    }

    /**
     * Quantizes contiguous training vectors in place, retaining the existing
     * FP64 norms and FP32 scaling-factor search. Scratch belongs to logical
     * workers and is reused across epochs, regardless of which pool threads
     * execute them. One encoder belongs to one trainer and is not thread-safe.
     */
    private static final class TrainingDirectionEncoder {
        private final int rows;
        private final int dims;
        private final int elements;
        private final int bits;
        private final DirectionScratch[] workers;

        TrainingDirectionEncoder(int rows, int dims, int bits) {
            validateBitsPerDimension(bits);
            if (rows <= 0 || dims <= 0 || (long) rows * dims > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Invalid ITQ direction dimensions");
            }
            this.rows = rows;
            this.dims = dims;
            this.elements = rows * dims;
            this.bits = bits;
            int count = bits == 1 ? 0
                    : Math.min(rows, PhysicalCoreExecutor.getPhysicalCoreCount());
            workers = new DirectionScratch[count];
            for (int i = 0; i < count; i++) workers[i] = new DirectionScratch(dims);
        }

        void encodeInPlace(double[] vectors) {
            Objects.requireNonNull(vectors, "vectors");
            if (vectors.length != elements) {
                throw new IllegalArgumentException("Invalid ITQ direction buffer length");
            }
            if (bits == 1) {
                double inverse = 1.0 / Math.sqrt(dims);
                for (int i = 0; i < elements; i++) {
                    vectors[i] = vectors[i] >= 0.0 ? inverse : -inverse;
                }
                return;
            }
            int maxCode = (1 << (bits - 1)) - 1;
            if (workers.length == 1) {
                encodeRows(vectors, workers[0], 0, rows, maxCode);
            } else {
                // Share the library's bounded physical-core pool. A trainer's
                // logical workers own their scratch and disjoint output rows.
                PhysicalCoreExecutor.instance().execute(() ->
                        IntStream.range(0, workers.length).parallel().forEach(worker -> {
                            int first = (int) ((long) worker * rows / workers.length);
                            int end = (int) ((long) (worker + 1) * rows / workers.length);
                            encodeRows(vectors, workers[worker], first, end, maxCode);
                        }));
            }
        }

        private void encodeRows(double[] vectors, DirectionScratch scratch, int first, int end, int maxCode) {
            double[] absNorm = scratch.absNorm;
            double[] mag = scratch.mag;
            for (int r = first; r < end; r++) {
                int offset = r * dims;
                double norm2 = 0.0;
                for (int j = 0; j < dims; j++) {
                    double value = vectors[offset + j];
                    norm2 += value * value;
                }
                double inverse = 1.0 / Math.sqrt(Math.max(norm2, 1e-20));
                for (int j = 0; j < dims; j++) {
                    absNorm[j] = Math.abs(vectors[offset + j]) * inverse;
                }
                double scale = computeOptimalScalingFactor(absNorm, dims, bits);
                double encodedNorm2 = 0.0;
                for (int j = 0; j < dims; j++) {
                    int q = Math.min((int) (scale * absNorm[j] + K_EPS), maxCode);
                    double magnitude = q + 0.5;
                    mag[j] = magnitude;
                    encodedNorm2 += magnitude * magnitude;
                }
                double encodedInverse = 1.0 / Math.sqrt(Math.max(encodedNorm2, 1e-20));
                for (int j = 0; j < dims; j++) {
                    double signed = vectors[offset + j] < 0.0 ? -mag[j] : mag[j];
                    vectors[offset + j] = signed * encodedInverse;
                }
            }
        }
    }

    private static final class DirectionScratch {
        final double[] absNorm;
        final double[] mag;

        DirectionScratch(int dims) {
            absNorm = new double[dims];
            mag = new double[dims];
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

    static void writeFloat16(IndexWriter out, float value) throws IOException {
        int bits = floatToHalfBits(value);

        // Explicit big-endian byte order, matching Java DataOutput-style primitives.
        out.writeByte((byte) ((bits >>> 8) & 0xFF));
        out.writeByte((byte) (bits & 0xFF));
    }

    static float readFloat16(RandomAccessReader in) throws IOException {
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

    /**
     * Retained model storage, including provider-owned native weights and lazy query weights.
     * Like PQ's compressor accounting, this excludes worker scratch and training temporaries.
     * Immutable Java backing is measured once; querying this estimate does not scan matrices.
     */
    @Override
    public long ramBytesUsed() {
        long bytes = heapBytes + stiefelTransform.projection.ramBytesUsed();
        var queryProjection = stiefelTransform.queryProjection;
        if (queryProjection != null) {
            bytes += stiefelTransform.queryProjectionHeapBytes;
            if (queryProjection != stiefelTransform.projection) bytes += queryProjection.ramBytesUsed();
        }
        return bytes;
    }

    static long estimateArray2DBytes(double[][] a) {
        if (a == null) return 0L;
        long bytes = RamUsageEstimator.shallowSizeOf(a);
        for (double[] row : a) bytes = Math.addExact(bytes, RamUsageEstimator.shallowSizeOf(row));
        return bytes;
    }

    static long estimateArray2DBytes(float[][] a) {
        if (a == null) return 0L;
        long bytes = RamUsageEstimator.shallowSizeOf(a);
        for (float[] row : a) bytes = Math.addExact(bytes, RamUsageEstimator.shallowSizeOf(row));
        return bytes;
    }

    private static long estimateMatrixBytes(RealMatrix matrix) {
        long bytes = RamUsageEstimator.shallowSizeOf(matrix);
        if (matrix instanceof Array2DRowRealMatrix) {
            return bytes + estimateArray2DBytes(((Array2DRowRealMatrix) matrix).getDataRef());
        }
        int rows = matrix.getRowDimension(), cols = matrix.getColumnDimension();
        if (matrix instanceof org.apache.commons.math3.linear.BlockRealMatrix) {
            int width = org.apache.commons.math3.linear.BlockRealMatrix.BLOCK_SIZE;
            long blocks = ((rows - 1L) / width + 1) * ((cols - 1L) / width + 1);
            bytes += RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                    + blocks * RamUsageEstimator.NUM_BYTES_OBJECT_REF);
            // Tail blocks have their own alignment padding; aggregating their
            // payloads before alignment undercounts some 16-byte-aligned layouts.
            for (long firstRow = 0; firstRow < rows; firstRow += width) {
                long blockRows = Math.min(width, rows - firstRow);
                for (long firstColumn = 0; firstColumn < cols; firstColumn += width) {
                    long blockColumns = Math.min(width, cols - firstColumn);
                    bytes = Math.addExact(bytes, RamUsageEstimator.alignObjectSize(
                            RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                                    + blockRows * blockColumns * Double.BYTES));
                }
            }
            return bytes;
        }
        // Other RealMatrix implementations retain the existing dense estimate.
        // Avoid getData(), which can allocate a full matrix copy just to measure it.
        return bytes + RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                + (long) rows * RamUsageEstimator.NUM_BYTES_OBJECT_REF)
                + rows * RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                + (long) cols * Double.BYTES);
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
