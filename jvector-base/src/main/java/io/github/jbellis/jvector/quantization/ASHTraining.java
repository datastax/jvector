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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.BlasThreads;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import dev.ludovic.netlib.blas.BLAS;
import dev.ludovic.netlib.lapack.LAPACK;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.netlib.util.intW;

/** Learns and initializes ASH projections: training-sample preparation, ITQ, and the random transform. */
final class ASHTraining {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(AsymmetricHashing.class);
    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport vectorUtilSupport =
            VectorizationProvider.getInstance().getVectorUtilSupport();

    /**
     * training_size = min(N, D * training_factor).
     */
    static final int ITQ_TRAINING_FACTOR = 20;

    /** Bounds transpose scratch while keeping writes to the training matrix contiguous. */
    static final int TRAINING_TRANSPOSE_TILE_ROWS = 32;

    private ASHTraining() {}

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
     * @return a {@link AsymmetricHashing.StiefelTransform} containing W ∈ R^{D×d} with orthonormal columns
     * @throws IllegalArgumentException if {@code d <= 0 or d > D}
     * @throws RuntimeException if the native SVD fails
     */
    static AsymmetricHashing.StiefelTransform runWithoutTraining(int D, int d, Random rng) {
        try (var blas = BlasThreads.singleUnlessLimited()) {
            return randomTransform(D, d, rng);
        }
    }

    static AsymmetricHashing.StiefelTransform randomTransform(int D, int d, Random rng) {
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
        return new AsymmetricHashing.StiefelTransform(W);
    }

    // ---------------------------------------------------------------------
    // Training of projection matrix
    // ---------------------------------------------------------------------

    /** Sampled, normalized residuals in the column-major layout consumed by LAPACK. */
    static final class TrainingData {
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

    static final class SingleLandmarkTraining {
        final VectorFloat<?> landmark;
        final TrainingData trainingData;
        /** The sample's mean squared vector length. */
        final double meanSquaredNorm;

        SingleLandmarkTraining(VectorFloat<?> landmark, TrainingData trainingData, double meanSquaredNorm) {
            this.landmark = landmark;
            this.trainingData = trainingData;
            this.meanSquaredNorm = meanSquaredNorm;
        }
    }

    static int trainingSampleSize(int count, int dimensions) {
        return (int) Math.min((long) count, Math.max(1L, (long) dimensions * ITQ_TRAINING_FACTOR));
    }

    /**
     * Computes the C=1 landmark and ITQ inputs from the same deterministic
     * sample. Only sampled source vectors are read; raw sample values are
     * retained until the mean is known. One reusable vector protects borrowed
     * source storage, and double sums follow sorted ordinal order.
     *
     * <p>Like encoding, training works on the raw vectors: the landmark is the
     * mean of the sampled vectors, and ITQ learns from (x - mu) / ||x - mu||.
     * Vectors are not scaled to unit length first.
     */
    static SingleLandmarkTraining prepareSingleLandmarkTraining(
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
        final Random sampleRandom = learn ? rng : new Random(AsymmetricHashing.TRAINING_SEED);
        final int[] ordinals = reservoirSampleOrdinals(count, sampleSize, sampleRandom);
        Arrays.sort(ordinals);
        final float[] sample = learn ? new float[(int) sampleElements] : null;
        final double[] accumulator = new double[dimensions];
        double squaredNorms = 0;
        final VectorFloat<?> rowValues = vectorTypeSupport.createFloatVector(dimensions);
        final float[] values = rowValues instanceof FloatArray ? ((FloatArray) rowValues).array() : null;
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
            rowValues.copyFrom(input, 0, 0, dimensions);
            // Dimensions are already checked. Call the provider directly so invalid
            // norms reach argument validation even when VectorUtil assertions are enabled.
            // Encoding computes FP32 norms too, so a vector whose norm is not finite
            // cannot be encoded and is rejected here.
            float norm = (float) Math.sqrt(vectorUtilSupport.dotProduct(rowValues, rowValues));
            if (!Float.isFinite(norm)) {
                throw new IllegalArgumentException("ASH training vector at ordinal " + ordinal
                        + " has a non-finite FP32 L2 norm");
            }
            squaredNorms += (double) norm * norm;
            if (values != null) {
                for (int d = 0; d < dimensions; d++) accumulator[d] += values[d];
            } else {
                for (int d = 0; d < dimensions; d++) accumulator[d] += rowValues.get(d);
            }
            if (learn) {
                int offset = row * dimensions;
                if (values != null) {
                    System.arraycopy(values, 0, sample, offset, dimensions);
                } else {
                    for (int d = 0; d < dimensions; d++) sample[offset + d] = rowValues.get(d);
                }
            }
        }
        VectorFloat<?> mean = vectorTypeSupport.createFloatVector(dimensions);
        double invCount = 1.0 / sampleSize;
        for (int d = 0; d < dimensions; d++) mean.set(d, (float) (accumulator[d] * invCount));
        double meanSquaredNorm = squaredNorms * invCount;
        if (!learn) return new SingleLandmarkTraining(mean, null, meanSquaredNorm);
        return new SingleLandmarkTraining(mean, centerTrainingSample(sample, sampleSize, dimensions, mean),
                meanSquaredNorm);
    }

    /**
     * Centers/normalizes contiguous sample rows in FP64, then writes small tiles
     * into the final column-major matrix. No full-size double row matrix or
     * strided, element-at-a-time whole-matrix conversion is needed. Tile width
     * bounds scratch only; it does not change arithmetic or sample boundaries.
     */
    static TrainingData centerTrainingSample(float[] sample, int rows, int dimensions,
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
    static TrainingData prepareTrainingData(
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
    static int[] reservoirSampleOrdinals(int N, int k, Random rng) {
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
    static AsymmetricHashing.StiefelTransform runItqTrainer(
            TrainingData trainingData, // [N x D] normalized residuals, column-major
            int latentDim,            // projected dimensions, d
            int bitsPerDimension,
            Random rng,
            int nTrainingIterations
    ) {
        LOG.info("ASH ITQ training started");

        AsymmetricHashing.validateBitsPerDimension(bitsPerDimension);

        final int N = trainingData.rows;
        final int D = trainingData.columns;
        final int d = latentDim;

        if (d <= 0 || d > D) {
            throw new IllegalArgumentException("Invalid latentDim=" + d + " for D=" + D);
        }

        // X is [N x D], flat column-major for BLAS/LAPACK.
        double[] xCol = trainingData.xCol;

        LOG.debug("ASH ITQ: PCA started");
        double[] pCol = computePcaBasis(xCol, N, D, d);
        LOG.debug("ASH ITQ: PCA finished");

        // Xld = X @ P, shape [N x d].
        double[] xldCol = nativeMultiply(xCol, N, D, pCol, d);

        // Initial random update matrix M, shape [d x d].
        double[] mCol = gaussianMatrix(d, d, rng);

        // Each projected vector is contiguous. Quantization replaces it in place:
        // the loss uses the update matrix, so unquantized rows need not be retained.
        double[] projectedRows = new double[Math.multiplyExact(N, d)];

        LOG.debug("ASH ITQ: training iterations started");
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
                LOG.info("ASH ITQ training converged at epoch {}", epoch);
                break;
            }
            recentLosses[epoch % 3] = loss.loss;
        }

        // Recompute from the final update, in FP64, rather than retaining the
        // previous epoch's rotation or widening a FP32 polar factor.
        new ASHOrthogonalizer(d, d).orthogonalize(mCol, rCol);

        double loopTime = (System.nanoTime() - startTime) / 1e9;
        if (LOG.isInfoEnabled()) {
            LOG.info("ASH ITQ training finished in {} s", String.format(java.util.Locale.ROOT, "%.1f", loopTime));
        }

        // W = P @ R, shape [D x d].
        double[] wCol = nativeMultiply(pCol, D, d, rCol, d);

        RealMatrix W = new Array2DRowRealMatrix(deserializeColumnMajor(wCol, D, d), false);
        return new AsymmetricHashing.StiefelTransform(W);
    }

    static final class ProjectionTrainingLoss {
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
    static ProjectionTrainingLoss computeTrainingLossFromUpdateMatrix(
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

    static void printTrainingLoss(int epoch, ProjectionTrainingLoss loss) {
        if (LOG.isDebugEnabled()) {
            LOG.debug(String.format(java.util.Locale.ROOT,
                    "ASH ITQ: epoch %2d inner-product loss %.5f, loss %.5f", epoch, loss.ip, loss.loss));
        }
    }

    /**
     * Samples a Gaussian matrix in flat column-major storage.
     *
     * Values are assigned in row-major logical order.
     */
    static double[] gaussianMatrix(int rows, int cols, Random rng) {
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
    static double[] serializeColumnMajor(double[][] matrix) {
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
    static double[][] deserializeColumnMajor(double[] data, int rows, int cols) {
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
    static double[] extractPcaBasis(double[] vtCol, int dim, int d) {
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
    static double[] nativeMultiply(double[] aCol, int m, int k, double[] bCol, int n) {
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
    static void projectTrainingRows(double[] xldCol, int rows, int dims,
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
    static void updateTrainingMatrix(double[] xldCol, int rows, int dims,
                                             double[] encodedRows, double[] mCol) {
        checkTrainingProduct(xldCol, rows, dims, mCol, encodedRows);
        if (mCol == xldCol || mCol == encodedRows) {
            throw new IllegalArgumentException("ITQ update output must not alias an input");
        }
        BLAS.getInstance().dgemm("T", "T", dims, dims, rows, 1.0,
                xldCol, rows, encodedRows, dims, 0.0, mCol, dims);
    }

    static void checkTrainingProduct(double[] xldCol, int rows, int dims,
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
    static double[] computeNativeVt(double[] xCol, int rows, int cols) {
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
    static double[] computePcaBasis(double[] xCol, int rows, int dims, int latent) {
        double[] basis = ASHCovariancePCA.projection(xCol, rows, dims, latent);
        if (basis != null) {
            LOG.debug("ASH ITQ: PCA used FP64 covariance eigendecomposition");
            return basis;
        }
        LOG.debug("ASH ITQ: PCA used guarded direct SVD");
        return extractPcaBasis(computeNativeVt(xCol, rows, dims), dims, latent);
    }

    /** Single-use convenience path for random projection initialization. */
    static double[] orthogonalize(double[] mCol, int rows, int cols) {
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
    static final class TrainingDirectionEncoder {
        private final int rows;
        private final int dims;
        private final int elements;
        private final int bits;
        private final DirectionScratch[] workers;

        TrainingDirectionEncoder(int rows, int dims, int bits) {
            AsymmetricHashing.validateBitsPerDimension(bits);
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
                double scale = AsymmetricHashing.computeOptimalScalingFactor(absNorm, dims, bits);
                double encodedNorm2 = 0.0;
                for (int j = 0; j < dims; j++) {
                    int q = Math.min((int) (scale * absNorm[j] + AsymmetricHashing.K_EPS), maxCode);
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

    static final class DirectionScratch {
        final double[] absNorm;
        final double[] mag;

        DirectionScratch(int dims) {
            absNorm = new double[dims];
            mag = new double[dims];
        }
    }
}
