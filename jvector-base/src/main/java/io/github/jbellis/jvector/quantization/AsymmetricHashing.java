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
import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.CallerBatchProcessor;
import io.github.jbellis.jvector.util.BlasThreads;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
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
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * Asymmetric Hashing (ASH) for float vectors.
 * Encodes each vector into a fixed-length code using a learned or random
 * orthonormal projection. The 1-bit case stores only sign bits; multibit ASH
 * stores either generic sign/extra bits or the C++-style fast-scan projection code for 2/4-bit ASH.
 */
@Experimental
public class AsymmetricHashing implements VectorCompressor<AsymmetricHashing.QuantizedVector>, Accountable {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(AsymmetricHashing.class);
    public static final int ITQ = 1, RANDOM = 2;
    private static final int MAGIC = 0x75EC4015;
    /** The first on-disk graph format version that can store ASH (version 7, which added FUSED_ASH). */
    public static final int MIN_FORMAT_VERSION = 7;

    // ---------------------------------------------------------------------
    // Training configuration (defaults used for all published results)
    // ---------------------------------------------------------------------

    /** Shared by sample selection and the existing random projection initialization. */
    static final long TRAINING_SEED = 103;

    /** ITQ iterations. */
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

    /**
     * The similarity this model was trained for. It is fixed when the model is created and saved with it: the
     * stored headers hold that similarity's terms, so its codes are scored only by it. See
     * {@link #similarityFunction()}.
     */
    private final VectorSimilarityFunction similarityFunction;

    /**
     * For EUCLIDEAN, a power of two that scales vectors to about unit length before a code's terms are stored, so the
     * half-precision headers hold squared distances without overflow; 1 otherwise. Scaling by a power of two only
     * shifts exponents, so it adds no rounding. Squared distances are divided by its square when scored.
     */
    final float unit;

    /** How this model's kernel estimates become its similarity. */
    private final ASHScoreScale scores;

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

    // The largest length of any vector this model has encoded or holds codes for, or was saved with. Not part of the
    // model's identity.
    private final io.github.jbellis.jvector.util.MaxVectorLength maxEncoded = new io.github.jbellis.jvector.util.MaxVectorLength();

    // Half-precision headers estimate a squared length to within about 0.1%; codes recorded from their headers are
    // counted this much longer, so the recorded length stays an upper bound.
    private static final float HEADER_NORM_MARGIN = 1f + 1f / 128f;

    void recordEncodedSquaredNorm(float squaredNorm) {
        maxEncoded.recordSquared(squaredNorm);
    }

    /**
     * The largest length of any vector this model has encoded, including vectors encoded before it was saved and
     * loaded, and of any vector whose codes {@link ASHVectors} has wrapped with it (estimated from the codes, with a
     * margin for their half-precision headers). A vector's length is recorded before its encoding returns. With a
     * query's length, it bounds the dot products ASH estimates for that query, which adaptive stopping needs to read
     * ASH scores as distances. NaN until the model encodes or holds a vector, when dot-product searches stop by the
     * standard rule.
     */
    public float maxEncodedNorm() {
        return maxEncoded.get();
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
                              VectorSimilarityFunction similarityFunction,
                              float unit,
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

        // Both optimizers quantize the projected body the same way; they differ only in the
        // learned transform.
        if (optimizer != RANDOM && optimizer != ITQ) {
            throw new IllegalArgumentException("Unknown optimizer " + optimizer);
        }
        this.optimizer = optimizer;
        this.similarityFunction = supportedSimilarity(similarityFunction);
        if (!(unit > 0f) || Math.scalb(1f, Math.getExponent(unit)) != unit
                || (similarityFunction != VectorSimilarityFunction.EUCLIDEAN && unit != 1f)) {
            throw new IllegalArgumentException("Invalid ASH unit " + unit + " for " + similarityFunction);
        }
        this.unit = unit;
        this.scores = similarityFunction == VectorSimilarityFunction.EUCLIDEAN
                ? ASHScoreScale.euclidean(unit) : ASHScoreScale.DOT_PRODUCT;
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
                // The similarity is a shared enum constant and its unit a field; a EUCLIDEAN model owns its scale.
                + (scores == ASHScoreScale.DOT_PRODUCT ? 0 : RamUsageEstimator.shallowSizeOf(scores));
        for (VectorFloat<?> landmark : landmarks) bytes = Math.addExact(bytes, landmark.ramBytesUsed());
        this.heapBytes = bytes;
    }

    /**
     * The similarity this model was trained for, fixed when it was created and saved with it. Its codes, and every
     * scorer of them, use only this similarity; a scorer requested for another one is rejected.
     */
    public VectorSimilarityFunction similarityFunction() {
        return similarityFunction;
    }

    /**
     * Throws unless {@code requested} is this model's similarity: the stored headers hold that similarity's terms, so
     * scores for another one would be wrong.
     */
    void checkSimilarity(VectorSimilarityFunction requested) {
        if (requested != similarityFunction) {
            throw new IllegalArgumentException("This ASH model was trained for " + similarityFunction
                    + " and scores only by it; requested " + requested);
        }
    }

    /** How this model's kernel estimates become its similarity; every scorer of its codes converts through it. */
    ASHScoreScale scores() {
        return scores;
    }

    /**
     * Copies {@code query} into {@code input}, the FP32 vector the projection reads, in this model's form: for
     * EUCLIDEAN scaled by the unit. Allocates nothing.
     *
     * @param inputView a vector viewing {@code input}, or null; for similarities whose scoring vector is the
     *        transformed input
     * @return the vector the landmark terms are scored from: {@code query}
     */
    VectorFloat<?> loadQuery(VectorFloat<?> query, float[] input, VectorFloat<?> inputView) {
        if (query instanceof FloatArray) {
            System.arraycopy(((FloatArray) query).array(), 0, input, 0, input.length);
        } else {
            for (int i = 0; i < input.length; i++) input[i] = query.get(i);
        }
        if (unit != 1f) {
            for (int i = 0; i < input.length; i++) input[i] *= unit;
        }
        return query;
    }

    /** Whether a query's FP32 projection input differs from the caller's query: EUCLIDEAN scaling. */
    boolean transformsQueryInput() {
        return unit != 1f;
    }


    /**
     * The query term every kernel adds for codes of landmark {@code c}: {@code <q, mu_c>}, or for EUCLIDEAN
     * {@code -||q - mu_c||^2 / 2} in model units.
     */
    float queryLandmarkTerm(VectorFloat<?> query, int c) {
        if (similarityFunction != VectorSimilarityFunction.EUCLIDEAN) return VectorUtil.dotProduct(query, landmarks[c]);
        return -0.5f * (unit * unit) * VectorUtil.squareL2Distance(query, landmarks[c]);
    }

    /**
     * The landmark term of symmetric (code-to-code) scoring for codes of landmarks {@code a} and {@code b}:
     * {@code <mu_a, mu_b>}, or for EUCLIDEAN {@code -||mu_a - mu_b||^2 / 2} in model units.
     */
    float symmetricLandmarkTerm(int a, int b) {
        if (similarityFunction != VectorSimilarityFunction.EUCLIDEAN) return VectorUtil.dotProduct(landmarks[a], landmarks[b]);
        return -0.5f * (unit * unit) * VectorUtil.squareL2Distance(landmarks[a], landmarks[b]);
    }

    private static VectorSimilarityFunction supportedSimilarity(VectorSimilarityFunction similarity) {
        Objects.requireNonNull(similarity, "similarityFunction");
        if (similarity == VectorSimilarityFunction.COSINE) {
            throw new IllegalArgumentException("ASH does not support COSINE similarity");
        }
        return similarity;
    }

    /**
     * As {@link #initialize(RandomAccessVectorValues, VectorSimilarityFunction, int, int, int, int)} with the default
     * 2 bits per projected dimension.
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               VectorSimilarityFunction similarityFunction,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount) throws IOException {
        return initialize(ravv, similarityFunction, optimizer, encodedBits, landmarkCount, DEFAULT_BITS_PER_DIMENSION);
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
     * <p>The similarity is required: it is fixed here and saved with the model, and the model's codes, and every
     * scorer of them, use only it. ASH does not require unit-length vectors for any similarity.</p>
     *
     * @param ravv the vectors to quantize
     * @param similarityFunction DOT_PRODUCT or EUCLIDEAN
     * @param optimizer the optimizer to use
     * @param encodedBits the number of bits used to encode vector, including the header
     * @param landmarkCount number of landmarks
     * @param bitsPerDimension number of stored bits per projected dimension
     * @throws IllegalArgumentException if the configuration is invalid, the source
     *         is empty, or a sampled vector has an invalid dimension or non-finite norm
     * @throws NullPointerException if the source or a sampled vector is null
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               VectorSimilarityFunction similarityFunction,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount,
                                               int bitsPerDimension) throws IOException {
        supportedSimilarity(similarityFunction);
        try (var blas = BlasThreads.singleUnlessLimited()) {
            return train(ravv, similarityFunction, optimizer, encodedBits, landmarkCount, bitsPerDimension);
        }
    }

    /**
     * Initialize ASH index-wide parameters, letting native BLAS and LAPACK use as many threads as
     * {@code executor} has. Without an executor, training uses one BLAS thread (see
     * {@link BlasThreads}); other training work uses the library's physical-core pool either way.
     *
     * @param executor the pool whose parallelism sets the BLAS thread count
     * @see #initialize(RandomAccessVectorValues, VectorSimilarityFunction, int, int, int, int)
     */
    public static AsymmetricHashing initialize(RandomAccessVectorValues ravv,
                                               VectorSimilarityFunction similarityFunction,
                                               int optimizer,
                                               int encodedBits,
                                               int landmarkCount,
                                               int bitsPerDimension,
                                               ForkJoinPool executor) throws IOException {
        supportedSimilarity(similarityFunction);
        Objects.requireNonNull(executor, "executor");
        try (var blas = BlasThreads.limit(Math.max(1, executor.getParallelism()))) {
            return train(ravv, similarityFunction, optimizer, encodedBits, landmarkCount, bitsPerDimension);
        }
    }

    /**
     * The EUCLIDEAN unit for vectors with mean squared length {@code meanSquaredNorm}: the power of two that scales
     * their root-mean-square length into [1, 2). 1 when there is no length to scale.
     */
    static float euclideanUnit(double meanSquaredNorm) {
        float rms = (float) Math.sqrt(meanSquaredNorm);
        if (!(rms >= Float.MIN_NORMAL) || !Float.isFinite(rms)) return 1f;
        return Math.scalb(1f, -Math.getExponent(rms));
    }

    private static AsymmetricHashing train(RandomAccessVectorValues ravv,
                                           VectorSimilarityFunction similarityFunction,
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

        LOG.info("ASH training: {} landmark(s), {} projected dimensions, {} bit(s) per dimension",
                landmarkCount, quantizedDim, bitsPerDimension);

        // C=1 uses the same sample for its mean and learned projection.
        // Sample selection preserves ITQ's existing RNG sequence.
        final Random rng = new Random(TRAINING_SEED);
        VectorFloat<?>[] landmarks;
        ASHTraining.TrainingData trainingData = null;
        double meanSquaredNorm;
        if (landmarkCount == 1) {
            ASHTraining.SingleLandmarkTraining prepared = ASHTraining.prepareSingleLandmarkTraining(
                    ravvCopy, originalDim, rng, optimizer == ITQ);
            landmarks = new VectorFloat<?>[]{prepared.landmark};
            trainingData = prepared.trainingData;
            meanSquaredNorm = prepared.meanSquaredNorm;
        } else {
            // Multi-landmark clustering still needs its materialized input. Like
            // encoding, k-means and ITQ use the raw vectors, not unit-length copies.
            VectorFloat<?>[] points = new VectorFloat<?>[ravvCopy.size()];
            for (int i = 0; i < points.length; i++) {
                points[i] = ravvCopy.getVector(i).copy();
            }
            double squaredNorms = 0;
            for (VectorFloat<?> point : points) squaredNorms += VectorUtil.dotProduct(point, point);
            meanSquaredNorm = squaredNorms / points.length;
            long kmStart = System.nanoTime();
            MiniBatchKMeansClusterer km = new MiniBatchKMeansClusterer(points, landmarkCount, 1024);
            VectorFloat<?> packed = km.cluster(/*unweightedIterations*/ 20, /*anisotropicIterations*/ 0);
            long kmEnd = System.nanoTime();
            if (LOG.isInfoEnabled()) {
                LOG.info("ASH landmark k-means ({} landmarks, {} vectors, {} dimensions) finished in {} s",
                        landmarkCount, points.length, originalDim,
                        String.format(java.util.Locale.ROOT, "%.1f", (kmEnd - kmStart) / 1e9));
            }
            landmarks = new VectorFloat<?>[landmarkCount];
            for (int c = 0; c < landmarkCount; c++) {
                VectorFloat<?> mu = vectorTypeSupport.createFloatVector(originalDim);
                mu.copyFrom(packed, c * originalDim, 0, originalDim);
                landmarks[c] = mu;
            }
            if (optimizer == ITQ) trainingData = ASHTraining.prepareTrainingData(points, landmarks, originalDim, rng);
        }

        StiefelTransform stiefelTransform = optimizer == RANDOM
                ? ASHTraining.runWithoutTraining(originalDim, quantizedDim, rng)
                : ASHTraining.runItqTrainer(trainingData, quantizedDim, bitsPerDimension, rng, TRAINING_ITERS);

        LandmarkProjections lp =
                computeLandmarkProjections(
                        landmarks,
                        stiefelTransform,
                        quantizedDim,
                        originalDim
                );

        float unit = similarityFunction == VectorSimilarityFunction.EUCLIDEAN ? euclideanUnit(meanSquaredNorm) : 1f;
        return new AsymmetricHashing(
                originalDim,
                encodedBits,
                quantizedDim,
                bitsPerDimension,
                optimizer,
                similarityFunction,
                unit,
                landmarks,
                stiefelTransform,
                lp.proj,
                lp.sum
        );
    }

    static void validateBitsPerDimension(int bitsPerDimension) {
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
    private static void checkFormatVersion(int version) {
        if (version < MIN_FORMAT_VERSION || version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("ASH requires on-disk format version " + MIN_FORMAT_VERSION
                    + " to " + OnDiskGraphIndex.CURRENT_VERSION + ", got " + version);
        }
    }

    /** Stable on-disk codes for the similarity, independent of the enum's declaration order. */
    private static int similarityCode(VectorSimilarityFunction similarity) {
        switch (similarity) {
            case DOT_PRODUCT: return 1;
            case EUCLIDEAN: return 2;
            case COSINE: return 3;
            default: throw new IllegalArgumentException("No ASH code for " + similarity);
        }
    }

    private static VectorSimilarityFunction similarityFromCode(int code) throws IOException {
        switch (code) {
            case 1: return VectorSimilarityFunction.DOT_PRODUCT;
            case 2: return VectorSimilarityFunction.EUCLIDEAN;
            case 3: return VectorSimilarityFunction.COSINE;
            default: throw new IOException("Invalid ASH similarity code " + code);
        }
    }

    public static AsymmetricHashing load(RandomAccessReader in) throws IOException {

        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new IllegalArgumentException(
                    String.format("Invalid ASH magic: 0x%08X", magic)
            );
        }

        checkFormatVersion(in.readInt());

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
        VectorSimilarityFunction similarityFunction = similarityFromCode(in.readInt());
        float unit = in.readFloat();

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

        float maxEncodedNorm = in.readFloat();
        var ash = new AsymmetricHashing(
                originalDimension,
                encodedBits,
                quantizedDim,
                bitsPerDimension,
                optimizer,
                similarityFunction,
                unit,
                landmarks,
                stiefelTransform,
                lp.proj,
                lp.sum
        );
        ash.maxEncoded.restore(maxEncodedNorm);
        return ash;
    }

    /**
     * Writes the ASH model: the ASH magic number ({@code 0x75EC4015}), the on-disk graph format
     * version, the dimensions, bit widths and optimizer, the similarity and its unit, the landmarks, and the
     * Stiefel transform.
     * {@link #load} checks the magic number and version before reading the rest.
     *
     * @param out the IndexWriter into which to write the object
     * @param version the on-disk graph format version, from {@link #MIN_FORMAT_VERSION} to
     *        {@link OnDiskGraphIndex#CURRENT_VERSION}
     * @throws IllegalArgumentException if the version cannot store ASH
     * @throws IOException if there is a problem writing to out.
     */
    public void write(IndexWriter out, int version) throws IOException {
        checkFormatVersion(version);

        out.writeInt(MAGIC);
        out.writeInt(version);

        out.writeInt(originalDimension);
        out.writeInt(encodedBits);
        out.writeInt(quantizedDim);
        out.writeInt(bitsPerDimension);
        out.writeInt(optimizer);
        out.writeInt(similarityCode(similarityFunction));
        out.writeFloat(unit);

        // Multi-landmark serialization
        out.writeInt(landmarkCount);

        for (int c = 0; c < landmarkCount; c++) {
            out.writeInt(landmarks[c].length());
            vectorTypeSupport.writeFloatVector(out, landmarks[c]);
        }

        // Stiefel transform serialization:
        // Serialize A = W^T in float precision as a [d][D] row-major matrix.
        writeStiefelTransform(out, stiefelTransform);
        out.writeFloat(maxEncodedNorm());
    }

    // ---------------------------------------------------------------------
    // Index-level quantization entry point
    // ---------------------------------------------------------------------

    float quantizeVector(VectorFloat<?> vector,
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
                dest
        );
    }

    // ---------------------------------------------------------------------
    // Projection-code layout helpers
    // ---------------------------------------------------------------------

    static boolean usesFastScanProjectionCode(int bitsPerDimension) {
        return bitsPerDimension == 2 || bitsPerDimension == 4;
    }

    /**
     * Whether codes are offset-binary fields: dimension {@code j} is the unsigned {@code b}-bit field at bits
     * {@code [j*b, (j+1)*b)} of the little-endian body, with value {@code field - (2^b - 1) / 2}. Every width
     * except 1 bit (sign words) and the 2/4-bit sign-magnitude projection codes uses this layout.
     */
    static boolean usesOffsetBinaryCode(int bitsPerDimension) {
        return bitsPerDimension != 1 && !usesFastScanProjectionCode(bitsPerDimension);
    }

    /** Field {@code j} of an offset-binary code whose body starts at {@code codeOffset}. */
    static int offsetBinaryField(byte[] code, int codeOffset, int j, int bitsPerDimension) {
        int bit = j * bitsPerDimension, at = codeOffset + (bit >>> 3), shift = bit & 7;
        int window = code[at] & 0xFF;
        if (shift + bitsPerDimension > 8) window |= (code[at + 1] & 0xFF) << 8;
        if (shift + bitsPerDimension > 16) window |= (code[at + 2] & 0xFF) << 16;
        return (window >>> shift) & ((1 << bitsPerDimension) - 1);
    }

    /** ORs {@code field} into dimension {@code j} of a zeroed offset-binary code. */
    static void setOffsetBinaryField(byte[] code, int j, int bitsPerDimension, int field) {
        int bit = j * bitsPerDimension;
        for (int t = 0; t < bitsPerDimension; t++, bit++) {
            if ((field & (1 << t)) != 0) code[bit >>> 3] |= (byte) (1 << (bit & 7));
        }
    }

    /** The value of an offset-binary field: an odd multiple of one half, centered on zero. */
    static float decodeOffsetBinary(int field, int bitsPerDimension) {
        return field - ((1 << bitsPerDimension) - 1) * 0.5f;
    }

    /** {@code <v, code>} for an offset-binary code starting at {@code codeOffset}. */
    static float dotOffsetBinaryCode(float[] v, byte[] code, int codeOffset, int quantizedDim, int bitsPerDimension) {
        float sum = 0f;
        for (int j = 0; j < quantizedDim; j++) {
            sum += v[j] * decodeOffsetBinary(offsetBinaryField(code, codeOffset, j, bitsPerDimension), bitsPerDimension);
        }
        return sum;
    }

    /**
     * {@code <A mu_c, code>}, the landmark term that every stored offset includes as
     * {@code -scale * <A mu_c, code>}, so that asymmetric scoring needs only {@code Aq}.
     */
    float landmarkCodeDot(int landmark, long[] signs, int signOffset, byte[] body, int bodyOffset) {
        float[] projected = landmarkProj[landmark];
        if (usesFastScanProjectionCode(bitsPerDimension)) {
            return dotProjectionCode(projected, body, bodyOffset, quantizedDim, bitsPerDimension);
        }
        if (bitsPerDimension != 1) return dotOffsetBinaryCode(projected, body, bodyOffset, quantizedDim, bitsPerDimension);
        // Signs s = 2b - 1, so <A mu, s> = 2 * (sum of A mu over set bits) - sum(A mu).
        float positive = 0f;
        int words = QuantizedVector.wordsForDims(quantizedDim);
        for (int w = 0, base = 0; w < words; w++, base += 64) {
            long word = signs[signOffset + w];
            while (word != 0L) {
                int j = base + Long.numberOfTrailingZeros(word);
                if (j < quantizedDim) positive += projected[j];
                word &= word - 1;
            }
        }
        return 2f * positive - landmarkProjSum[landmark];
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
        return centeredOffset(vector.scale, vector.offset, vector.landmark, vector.binaryVector, 0, vector.extraBits, 0);
    }

    /** The squared length of a code's decoded components: the norm its scale is divided by. */
    float codeSquaredNorm(QuantizedVector vector) {
        if (bitsPerDimension == 1) return quantizedDim;
        boolean projection = usesFastScanProjectionCode(bitsPerDimension);
        double sum = 0;
        for (int j = 0; j < quantizedDim; j++) {
            float v = projection ? projectionComponent(vector.extraBits, 0, j, bitsPerDimension)
                    : decodeOffsetBinary(offsetBinaryField(vector.extraBits, 0, j, bitsPerDimension), bitsPerDimension);
            sum += v * v;
        }
        return (float) sum;
    }

    /**
     * The squared length of the vector {@code vector} encodes, from its stored terms:
     * {@code ||x||^2 = ||x - mu||^2 + 2 (<x, mu> - ||mu||^2) + ||mu||^2}, with {@code ||x - mu|| = scale * ||code||}.
     * Exact up to the half-precision rounding of the header.
     */
    float encodedSquaredNorm(QuantizedVector vector) {
        int c = vector.landmark & 255;
        return vector.scale * vector.scale * codeSquaredNorm(vector) + 2f * centeredOffset(vector) + landmarkNormSq[c];
    }

    /**
     * Counts vectors encoded by another instance of this model in {@link #maxEncodedNorm}, from their codes, with a
     * margin for the half-precision headers.
     */
    void recordEncodedCodes(QuantizedVector[] vectors) {
        // EUCLIDEAN headers do not hold <x, mu>, and its scores need no dot-product bound.
        if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) return;
        for (var vector : vectors) {
            if (vector != null) recordEncodedSquaredNorm(encodedSquaredNorm(vector) * HEADER_NORM_MARGIN);
        }
    }

    /**
     * As {@link #centeredOffset(QuantizedVector)} for a header with sign words at {@code signOffset}
     * and a body at {@code bodyOffset}; only the array the bit width stores is read.
     */
    float centeredOffset(float scale, float offset, byte landmark, long[] signs, int signOffset,
                         byte[] body, int bodyOffset) {
        int c = landmark & 255;
        if (c >= landmarkCount) throw new IllegalArgumentException("Invalid landmark " + c);
        return offset + scale * (unit * landmarkCodeDot(c, signs, signOffset, body, bodyOffset));
    }

    // ---------------------------------------------------------------------
    // Quantization strategies
    // ---------------------------------------------------------------------

    /** Quantizes the projected residual body into sign words and extra magnitude bits. */
    private static final class ProjectedBinaryQuantizer {
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

        static float quantizeBody(VectorFloat<?> vector,
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

            // Offset-binary: magnitude index q is field 2^(b-1) + q when positive, 2^(b-1) - 1 - q otherwise.
            final int half = 1 << exBits;
            double codeNorm2 = 0.0;
            for (int j = 0; j < quantizedDim; j++) {
                int q = Math.min((int) (t * absNorm[j] + K_EPS), maxCode);
                double mag = q + 0.5;
                codeNorm2 += mag * mag;
                setOffsetBinaryField(outExtraBits, j, bitsPerDimension, proj[j] >= 0.0f ? half + q : half - 1 - q);
            }

            return (float) Math.sqrt(Math.max(codeNorm2, 1e-20));
        }

        /**
         * Fast-scan projection codes use C++-compatible 4-bit projection groups.
         * Core ASH supports bitsPerDimension in [1,9], but this layout is only
         * defined for the nibble-friendly multibit cases used by FusedASH.
         */
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
        public long[] binaryVector; // 1-bit codes only: sign bits; bit=1 means positive code value
        public byte[] extraBits; // multibit codes only: packed fields (see usesOffsetBinaryCode)

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

        /** Sign words: only 1-bit codes store them. */
        public static int signWordsForDims(int quantizedDim, int bitsPerDimension) {
            return bitsPerDimension == 1 ? wordsForDims(quantizedDim) : 0;
        }

        /** Body bytes: the packed fields of every multibit code, none for 1-bit codes. */
        public static int bodyBytesForDims(int quantizedDim, int bitsPerDimension) {
            if (bitsPerDimension == 1) return 0;
            if (usesFastScanProjectionCode(bitsPerDimension)) {
                return projectionCodeBytesForDims(quantizedDim, bitsPerDimension);
            }
            return Math.toIntExact(((long) quantizedDim * bitsPerDimension + 7) >>> 3);
        }

        public static QuantizedVector createEmpty(int quantizedDim) {
            return createEmpty(quantizedDim, DEFAULT_BITS_PER_DIMENSION);
        }

        /**
         * An all-zero code. It is also the code of a missing vector (such as a deleted one after a model change), so
         * its scale and offset are 0 and it scores at the landmark rather than as NaN.
         */
        public static QuantizedVector createEmpty(int quantizedDim, int bitsPerDimension) {
            return new QuantizedVector(
                    0f,
                    0f,
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

            return ProjectedBinaryQuantizer.quantizeBody(
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

            byte landmark = in.readByte();

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

            dest.landmark = in.readByte();

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

        static boolean getBit(long[] words, int bitIndex) {
            return getBit(words, 0, bitIndex);
        }

        /** Bit {@code bitIndex} of the sign words starting at {@code wordOffset}. */
        static boolean getBit(long[] words, int wordOffset, int bitIndex) {
            return ((words[wordOffset + (bitIndex >>> 6)] >>> (bitIndex & 63)) & 1L) != 0L;
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
     * Sets how this model projects queries: the operand precision and the largest group of concurrent
     * synchronous queries projected together in one matrix multiply (1 disables combining). Call it
     * before the first query, for example right after loading an index; afterwards it throws
     * {@link IllegalStateException}. Without it, the {@code jvector.ash.queryProjection} (default
     * {@code auto}) and {@code jvector.ash.queryBatchSize} (default 8) system properties apply.
     * Base-vector encoding always uses FP32.
     */
    public void setQueryOptions(ASHQueryProjection precision, int maxBatchSize) {
        stiefelTransform.configureQueries(precision, maxBatchSize);
    }

    /** The query projection precision this model uses or will use. */
    public ASHQueryProjection getQueryProjection() {
        return stiefelTransform.configuredQueryPrecision();
    }

    /** The largest group of concurrent synchronous queries this model projects together. */
    public int getQueryBatchSize() {
        return stiefelTransform.queryBatchSize();
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
        private static final String QUERY_BATCH_SIZE_PROPERTY = "jvector.ash.queryBatchSize";
        // Published together with queryProjection. Training/encoding never
        // initializes query-only dispatch or its retained storage.
        private CallerBatchProcessor<float[], float[]> queryBatches;
        // Query settings; fixed once the query projection is prepared.
        private int queryBatchSize;
        private ASHQueryProjection queryPrecision; // null: the system property
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
        long[] queryProjectionBatchHistogram() {
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
                        ready = projection.forQueryScoring(queryPrecision != null
                                ? queryPrecision : ASHQueryProjection.fromSystemProperty());
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

        /**
         * Sets query projection precision and the largest group of concurrent synchronous queries projected
         * together. Must be called before this model projects its first query.
         */
        synchronized void configureQueries(ASHQueryProjection precision, int maxBatchSize) {
            Objects.requireNonNull(precision, "precision");
            if (maxBatchSize < 1 || maxBatchSize == Integer.MAX_VALUE)
                throw new IllegalArgumentException("maxBatchSize must be between 1 and " + (Integer.MAX_VALUE - 1));
            if (queryProjection != null)
                throw new IllegalStateException("ASH query options must be set before the first query");
            queryPrecision = precision;
            queryBatchSize = maxBatchSize;
        }

        ASHQueryProjection configuredQueryPrecision() {
            return queryPrecision != null ? queryPrecision : ASHQueryProjection.fromSystemProperty();
        }

        int queryBatchSize() {
            return queryBatchSize;
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
            String batchSize = System.getProperty(QUERY_BATCH_SIZE_PROPERTY, "8");
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

            // Explicitly materialize A = Wᵀ into row-major arrays. Models persist A in FP32, so AData holds
            // those FP32 values: a trained model and its reloaded copy then derive identical landmark terms.
            this.AData = new double[quantizedDim][originalDim];
            for (int i = 0; i < quantizedDim; i++) {
                final double[] row = this.AData[i];
                for (int j = 0; j < originalDim; j++) {
                    // A[i][j] = W[j][i]
                    row[j] = (float) W.getEntry(j, i);
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
        final float[] symmetricOffsets = new float[n];
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

    /**
     * A builder that encodes vectors into an immutable {@link ASHVectors} of up to {@code capacity} codes as they
     * arrive, in batches from any number of threads; see {@link ASHVectors.Builder}.
     */
    public ASHVectors.Builder vectorsBuilder(int capacity) {
        return new ASHVectors.Builder(this, capacity);
    }

    /**
     * Encodes {@code vectors[0..count)} into {@code out} at {@code ordinals[0..count)}, as {@link #encodeAll}
     * encodes a range. Safe to call from several threads at once for distinct ordinals.
     */
    void encodeInto(VectorFloat<?>[] vectors, int[] ordinals, int count, ASHCodes out, float[] symmetricOffsets) {
        int capacity = Math.min(ENCODING_BATCH_SIZE, count);
        if (capacity == 0) return;
        float[][] inputs = new float[capacity][originalDimension];
        float[][] projected = new float[capacity][quantizedDim];
        float[] residualNorms = new float[capacity];
        int[] slots = new int[capacity];
        double[] absNorm = new double[quantizedDim];
        QuantizedVector[] scratch = new QuantizedVector[capacity];
        for (int k = 0; k < capacity; k++) scratch[k] = QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
        int pending = 0;
        for (int v = 0; v < count; v++) {
            VectorFloat<?> vector = Objects.requireNonNull(vectors[v], "vector");
            if (vector.length() != originalDimension) throw new IllegalArgumentException("ASH encoding vector dimension mismatch");
            QuantizedVector dest = scratch[pending];
            float residualNorm = prepareEncoding(vector, dest);
            normalizeResidual(vector, landmarks[dest.landmark & 0xFF], residualNorm, inputs[pending]);
            residualNorms[pending] = residualNorm;
            slots[pending] = ordinals[v];
            if (++pending == capacity) {
                encodeBatch(inputs, projected, residualNorms, slots, pending, absNorm, scratch, out, symmetricOffsets);
                pending = 0;
            }
        }
        if (pending != 0) encodeBatch(inputs, projected, residualNorms, slots, pending, absNorm, scratch, out, symmetricOffsets);
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
            float codeNorm = ProjectedBinaryQuantizer.quantizeProjected(projected[i], bitsPerDimension,
                    dest.binaryVector, dest.extraBits, absNorm);
            float correction = finishEncoding(dest, residualNorms[i], codeNorm);
            out.set(ordinals[i], dest);
            if (symmetricOffsets != null) symmetricOffsets[ordinals[i]] = correction;
        }
    }

    /**
     * Working set for encoding vectors in batches, as {@link #encodeAll} does, one batch at a time. It holds no
     * model, so one scratch serves every model of the same dimensions. Used by one thread at a time.
     */
    static final class EncodingBatch {
        private final float[][] inputs;
        private final float[][] projected;
        private final float[] residualNorms;
        private final double[] absNorm;
        private final QuantizedVector[] codes;
        private final float[] centeredOffsets;
        private int count;

        EncodingBatch(int originalDimension, int quantizedDim, int bitsPerDimension) {
            inputs = new float[ENCODING_BATCH_SIZE][originalDimension];
            projected = new float[ENCODING_BATCH_SIZE][quantizedDim];
            residualNorms = new float[ENCODING_BATCH_SIZE];
            absNorm = new double[quantizedDim];
            codes = new QuantizedVector[ENCODING_BATCH_SIZE];
            for (int i = 0; i < codes.length; i++) codes[i] = QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
            centeredOffsets = new float[ENCODING_BATCH_SIZE];
        }

        int size() {
            return count;
        }

        int capacity() {
            return codes.length;
        }

        boolean isFull() {
            return count == codes.length;
        }

        void clear() {
            count = 0;
        }

        /** The code of the {@code i}-th vector of the last encoded batch. */
        QuantizedVector code(int i) {
            return codes[i];
        }

        /** The centered offset of the {@code i}-th code, as {@link #encodeToWithCenteredOffset} returns it. */
        float centeredOffset(int i) {
            return centeredOffsets[i];
        }
    }

    /** A batch scratch sized for this model. */
    EncodingBatch newEncodingBatch() {
        return new EncodingBatch(originalDimension, quantizedDim, bitsPerDimension);
    }

    /** Adds {@code vector} to a batch that is not full, reading it now: the caller may reuse it afterwards. */
    void addToBatch(EncodingBatch batch, VectorFloat<?> vector) {
        if (batch.isFull()) throw new IllegalStateException("Encoding batch is full");
        int i = batch.count;
        QuantizedVector dest = batch.codes[i];
        float residualNorm = prepareEncoding(vector, dest);
        normalizeResidual(vector, landmarks[dest.landmark & 0xFF], residualNorm, batch.inputs[i]);
        batch.residualNorms[i] = residualNorm;
        batch.count = i + 1;
    }

    /** Encodes the batch's vectors with this model, as {@link #encodeAll} does; the batch keeps its size. */
    void encodeBatch(EncodingBatch batch) {
        int count = batch.count;
        if (count == 0) return;
        stiefelTransform.projectBatch(batch.inputs, batch.projected, count);
        for (int i = 0; i < count; i++) {
            QuantizedVector dest = batch.codes[i];
            Arrays.fill(dest.binaryVector, 0L);
            Arrays.fill(dest.extraBits, (byte) 0);
            float codeNorm = ProjectedBinaryQuantizer.quantizeProjected(batch.projected[i], bitsPerDimension,
                    dest.binaryVector, dest.extraBits, batch.absNorm);
            batch.centeredOffsets[i] = finishEncoding(dest, batch.residualNorms[i], codeNorm);
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
        encodeToWithCenteredOffset(vector, dest);
    }

    /** As {@link #encodeTo}, returning the code's centered offset for symmetric scoring. */
    float encodeToWithCenteredOffset(VectorFloat<?> vector, QuantizedVector dest) {
        float residualNorm = prepareEncoding(vector, dest);
        float codeNorm = quantizeVector(vector, residualNorm, dest.landmark, dest);
        return finishEncoding(dest, residualNorm, codeNorm);
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

        // DOT_PRODUCT: <x, mu> - ||mu||^2. EUCLIDEAN: -||x - mu||^2 / 2 in units of the model's unit; with the
        // query term -||q - mu||^2 / 2, the same kernels then estimate -||q - x||^2 / 2 (paper, Appendix A).
        final float rawOffset = similarityFunction == VectorSimilarityFunction.EUCLIDEAN
                ? -0.5f * (unit * unit) * sqDist
                : dotXMu - landmarkNormSq[c];
        // ||x||^2 = ||x - mu||^2 + 2 <x, mu> - ||mu||^2, from the terms already computed.
        recordEncodedSquaredNorm(sqDist + 2f * dotXMu - landmarkNormSq[c]);

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
        // Lengths are in units of the model's unit, 1 except for EUCLIDEAN; powers of two scale exactly.
        final float scale = (codeNorm > 0f ? residualNorm / codeNorm : 0f) * unit;

        // Projection-mode offset: offset = <x, μ> - ||μ||² - scale * <Aμ, code> (EUCLIDEAN: -||x - μ||² / 2
        // - scale * <Aμ, code>), so query-time scoring can use Aq instead of A(q - μ). A 1-bit code is its ±1 signs.
        final float landmarkDot = unit * landmarkCodeDot(c, dest.binaryVector, 0, dest.extraBits, 0);
        dest.scale = roundToFloat16(scale);
        dest.offset = roundToFloat16(rawOffset - scale * landmarkDot);
        return dest.offset + dest.scale * landmarkDot;
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

    static final double K_EPS = 1e-5;

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
    static double computeOptimalScalingFactor(
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
        size += Integer.BYTES; // similarity
        size += Float.BYTES; // unit
        size += Integer.BYTES; // landmarkCount
        for (int c = 0; c < landmarkCount; c++) {
            size += Integer.BYTES; // landmark dimension
            size += Float.BYTES * landmarks[c].length();
        }
        size += Integer.BYTES; // stiefel rows
        size += Integer.BYTES; // stiefel cols
        size += Float.BYTES * quantizedDim * originalDimension;
        size += Float.BYTES; // largest encoded vector length
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
        int hi = in.readByte() & 0xFF;
        int lo = in.readByte() & 0xFF;
        return halfBitsToFloat((hi << 8) | lo);
    }

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

    /**
     * The part of {@link #ramBytesUsed} held in native memory, outside the Java heap: native projections of the model
     * for encoding and for queries, which a native backend allocates when it first needs them.
     */
    public long nativeBytesUsed() {
        long bytes = stiefelTransform.projection.nativeBytesUsed();
        var queryProjection = stiefelTransform.queryProjection;
        if (queryProjection != null && queryProjection != stiefelTransform.projection)
            bytes += queryProjection.nativeBytesUsed();
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
                && similarityFunction == that.similarityFunction
                && unit == that.unit
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
        result = 31 * result + similarityCode(similarityFunction);
        result = 31 * result + Float.hashCode(unit);
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
            default:
                optimizerName = "UNKNOWN(" + optimizer + ")";
        }

        return String.format(
                "AsymmetricHashing[origDim=%d, encodedBits=%d, bodyBits=%d, quantizedDim=%d, bitsPerDimension=%d, optimizer=%s, similarity=%s, landmarks=%d]",
                originalDimension,
                encodedBits,
                bodyBits,
                quantizedDim,
                bitsPerDimension,
                optimizerName,
                similarityFunction,
                landmarkCount
        );
    }
}
