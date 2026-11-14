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
 * Reusable FP64 thin-SVD workspace for the polar factor U * VT.
 *
 * <p>One instance belongs to one trainer. It is not thread-safe and retains no
 * caller buffers. Input is copied only for the destructive decomposition; the
 * workspace query is performed once, using internal scratch.</p>
 */
final class ASHOrthogonalizer {
    private final int rows;
    private final int cols;
    private final int rank;
    private final int elements;
    private final double[] input;
    private final double[] u;
    private final double[] vt;
    private final double[] singularValues;
    private final int[] iwork;
    private final double[] work;
    private final intW info = new intW(0);
    private final LAPACK lapack = LAPACK.getInstance();
    private final BLAS blas = BLAS.getInstance();

    ASHOrthogonalizer(int rows, int cols) {
        if (rows <= 0 || cols <= 0 || (long) rows * cols > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Invalid ASH orthogonalization dimensions");
        }
        this.rows = rows;
        this.cols = cols;
        rank = Math.min(rows, cols);
        elements = rows * cols;
        input = new double[elements];
        u = new double[Math.multiplyExact(rows, rank)];
        vt = new double[Math.multiplyExact(rank, cols)];
        singularValues = new double[rank];
        iwork = new int[Math.multiplyExact(8, rank)];
        double[] query = new double[1];
        lapack.dgesdd("S", rows, cols, input, rows, singularValues, u, rows,
                vt, rank, query, -1, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("ASH SVD workspace query failed with info=" + info.val);
        }
        if (!Double.isFinite(query[0]) || query[0] < 1 || query[0] > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid ASH SVD workspace size: " + query[0]);
        }
        work = new double[(int) Math.ceil(query[0])];
    }

    /**
     * Writes the polar factor in column-major order. Buffers must have the
     * constructor's shape and must be distinct. The source is never modified,
     * including on failure; every successful call overwrites the whole output.
     */
    void orthogonalize(double[] source, double[] destination) {
        try (var blas = BlasThreads.singleUnlessLimited()) {
            orthogonalizeInScope(source, destination);
        }
    }

    private void orthogonalizeInScope(double[] source, double[] destination) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(destination, "destination");
        if (source.length != elements || destination.length != elements || source == destination) {
            throw new IllegalArgumentException("Invalid ASH orthogonalization buffers");
        }
        for (int i = 0; i < elements; i++) {
            double value = source[i];
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("ASH orthogonalization input must be finite");
            }
            input[i] = value;
        }
        info.val = 0;
        lapack.dgesdd("S", rows, cols, input, rows, singularValues, u, rows,
                vt, rank, work, work.length, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("ASH SVD failed with info=" + info.val);
        }
        for (double value : singularValues) {
            if (!Double.isFinite(value) || value < 0) {
                throw new IllegalStateException("ASH SVD returned an invalid singular value");
            }
        }
        blas.dgemm("N", "N", rows, cols, rank, 1.0, u, rows,
                vt, rank, 0.0, destination, rows);
    }
}
