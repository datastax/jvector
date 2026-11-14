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

import dev.ludovic.netlib.blas.BLAS;
import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

import io.github.jbellis.jvector.util.BlasThreads;

import java.util.Objects;

/**
 * FP64 PCA from a symmetric rank-k product. Avoids the unused tall left-singular
 * matrix. An uncertain retained subspace or a failed numerical check returns
 * null so the trainer can use its guarded direct SVD instead. No source is mutated.
 */
final class ASHCovariancePCA {
    private ASHCovariancePCA() {}

    /**
     * Returns the leading {@code latent} PCA directions, largest eigenvalue
     * first, as a column-major {@code [dims, latent]} matrix. The input is
     * column-major {@code [rows, dims]} and remains unchanged.
     *
     * <p>Returns {@code null} when direct SVD is needed: underdetermined input,
     * unresolved retained directions or cutoff, nonconvergence, or a failed
     * numerical check. Invalid input and native argument/workspace errors are
     * reported rather than hidden by the fallback.</p>
     */
    static double[] projection(double[] x, int rows, int dims, int latent) {
        try (var blas = BlasThreads.singleUnlessLimited()) {
            return computeProjection(x, rows, dims, latent);
        }
    }

    private static double[] computeProjection(double[] x, int rows, int dims, int latent) {
        Objects.requireNonNull(x, "x");
        if (rows <= 0 || dims <= 0 || latent <= 0 || latent > dims
                || (long) rows * dims > Integer.MAX_VALUE || (long) dims * dims > Integer.MAX_VALUE
                || x.length != rows * dims) {
            throw new IllegalArgumentException("Invalid PCA matrix dimensions");
        }
        double energy = 0;
        for (double value : x) {
            if (!Double.isFinite(value)) throw new IllegalArgumentException("Non-finite PCA input");
            energy += value * value;
        }
        if (!Double.isFinite(energy)) throw new IllegalArgumentException("PCA input energy overflows FP64");
        // Preserve direct SVD's full null-space basis for underdetermined inputs.
        if (rows < dims || energy == 0) return null;
        BLAS blas = BLAS.getInstance();
        double[] covariance = new double[dims * dims];
        blas.dsyrk("U", "T", dims, rows, 1.0, x, rows, 0.0, covariance, dims);
        double trace = 0;
        for (int col = 0; col < dims; col++) {
            for (int row = 0; row <= col; row++) {
                if (!Double.isFinite(covariance[row + col * dims])) return null;
            }
            trace += covariance[col + col * dims];
        }
        double energyTolerance = 1e-8 * Math.max(1.0, energy);
        if (!Double.isFinite(trace) || Math.abs(trace - energy) > energyTolerance) return null;

        // Retain just the small covariance for independent eigenpair checks.
        // Workspace queries do not need another input copy.
        double[] eigenvectors = covariance.clone();
        double[] values = new double[dims], query = new double[1];
        int[] integerQuery = new int[1];
        intW info = new intW(0);
        LAPACK lapack = LAPACK.getInstance();
        lapack.dsyevd("V", "U", dims, eigenvectors, dims, values, query, -1, integerQuery, -1, info);
        if (info.val != 0) throw new IllegalStateException("PCA eigen workspace query failed: " + info.val);
        if (!Double.isFinite(query[0]) || query[0] < 1 || query[0] > Integer.MAX_VALUE || integerQuery[0] < 1) {
            throw new IllegalStateException("Invalid PCA eigen workspace size");
        }
        double[] work = new double[(int) Math.ceil(query[0])];
        int[] integerWork = new int[integerQuery[0]];
        info.val = 0;
        lapack.dsyevd("V", "U", dims, eigenvectors, dims, values, work, work.length,
                integerWork, integerWork.length, info);
        if (info.val < 0) throw new IllegalStateException("Invalid native PCA eigen argument: " + info.val);
        if (info.val > 0) return null;
        double sum = 0;
        double previous = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (!Double.isFinite(value)) return null;
            if (value < previous) return null;
            previous = value;
            sum += value;
        }
        if (Math.abs(sum - energy) > energyTolerance) return null;
        double largest = values[dims - 1];
        // Covariance squares conditioning. Reject a retained tail or boundary
        // gap at the scale of accumulated FP64 error, including rank deficiency.
        double uncertainty = 128.0 * dims * Math.ulp(1.0) * Math.max(1.0, largest);
        int cutoff = dims - latent;
        if (!(largest > 0) || values[0] < -uncertainty || values[cutoff] <= uncertainty
                || (cutoff > 0 && values[cutoff] - values[cutoff - 1] <= uncertainty)) {
            return null;
        }

        double[] vector = new double[dims], product = new double[dims];
        int[] probes = {dims - 1, dims - 1 - latent / 2, cutoff};
        for (int p = 0; p < probes.length; p++) {
            int column = probes[p];
            if ((p > 0 && column == probes[0]) || (p > 1 && column == probes[1])) continue;
            System.arraycopy(eigenvectors, column * dims, vector, 0, dims);
            blas.dsymv("U", dims, 1.0, covariance, dims, vector, 1, 0.0, product, 1);
            double residual = 0;
            for (int i = 0; i < dims; i++) {
                double difference = product[i] - values[column] * vector[i];
                residual += difference * difference;
            }
            if (!Double.isFinite(residual) || Math.sqrt(residual) > 1e-8 * Math.max(1.0, largest)) return null;
        }
        double[] basis = new double[dims * latent];
        for (int col = 0; col < latent; col++) {
            int source = (dims - 1 - col) * dims;
            double norm = 0;
            for (int row = 0; row < dims; row++) {
                double value = eigenvectors[source + row];
                if (!Double.isFinite(value)) return null;
                basis[col * dims + row] = value;
                norm += value * value;
            }
            if (Math.abs(norm - 1.0) > 1e-8) return null;
        }
        return basis;
    }
}
