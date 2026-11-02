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

import java.util.Objects;

/**
 * Reusable FP32 thin-SVD workspace for the polar factor U * VT.
 * FP64 caller buffers keep the rest of the trainer independent of this choice.
 *
 * <p>One instance belongs to one trainer. It is not thread-safe and retains no
 * caller buffers. Input is copied only for the destructive decomposition; the
 * workspace query is performed once, using internal scratch.</p>
 */
final class ASHFloatOrthogonalizer {
    private final int rows;
    private final int cols;
    private final int rank;
    private final int elements;
    private final float[] input;
    private final float[] u;
    private final float[] vt;
    private final float[] singularValues;
    private final int[] iwork;
    private final float[] work;
    private final intW info = new intW(0);
    private final LAPACK lapack = LAPACK.getInstance();
    private final BLAS blas = BLAS.getInstance();

    ASHFloatOrthogonalizer(int rows, int cols) {
        if (rows <= 0 || cols <= 0 || (long) rows * cols > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Invalid ASH orthogonalization dimensions");
        }
        this.rows = rows;
        this.cols = cols;
        rank = Math.min(rows, cols);
        elements = rows * cols;
        input = new float[elements];
        u = new float[Math.multiplyExact(rows, rank)];
        vt = new float[Math.multiplyExact(rank, cols)];
        singularValues = new float[rank];
        iwork = new int[Math.multiplyExact(8, rank)];
        float[] query = new float[1];
        lapack.sgesdd("S", rows, cols, input, rows, singularValues, u, rows,
                vt, rank, query, -1, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("FP32 ASH SVD workspace query failed with info=" + info.val);
        }
        if (!Float.isFinite(query[0]) || query[0] < 1 || (double) query[0] > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid FP32 ASH SVD workspace size: " + query[0]);
        }
        // A FP32 query may round a large integer workspace size downward.
        // One representable step rounds it conservatively upward.
        double required = Math.nextUp(query[0]);
        if (required > Integer.MAX_VALUE) {
            throw new IllegalStateException("FP32 ASH SVD workspace exceeds the Java array limit");
        }
        work = new float[(int) Math.ceil(required)];
    }

    /**
     * Writes the polar factor in column-major order. Buffers must have the
     * constructor's shape and must be distinct. The source is never modified,
     * including on failure; every successful call overwrites the whole output.
     */
    void orthogonalize(double[] source, double[] destination) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(destination, "destination");
        if (source.length != elements || destination.length != elements || source == destination) {
            throw new IllegalArgumentException("Invalid ASH orthogonalization buffers");
        }
        for (int i = 0; i < elements; i++) {
            float value = (float) source[i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("ASH orthogonalization input must be finite and representable in FP32");
            }
            input[i] = value;
        }
        info.val = 0;
        lapack.sgesdd("S", rows, cols, input, rows, singularValues, u, rows,
                vt, rank, work, work.length, iwork, info);
        if (info.val != 0) {
            throw new IllegalStateException("FP32 ASH SVD failed with info=" + info.val);
        }
        for (float value : singularValues) {
            if (!Float.isFinite(value) || value < 0) {
                throw new IllegalStateException("FP32 ASH SVD returned an invalid singular value");
            }
        }
        // The destructive SVD input is dead and can hold the FP32 polar factor.
        blas.sgemm("N", "N", rows, cols, rank, 1.0f, u, rows,
                vt, rank, 0.0f, input, rows);
        for (int i = 0; i < elements; i++) {
            if (!Float.isFinite(input[i])) {
                throw new IllegalStateException("FP32 ASH polar factor contains a non-finite value");
            }
            destination[i] = input[i];
        }
    }
}
