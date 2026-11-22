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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.*;

/** FP32 polar factors against an independent FP64 reference and degenerate shapes. */
public class TestASHFloatOrthogonalizer {
    @Test
    public void reusedWorkspaceMatchesIndependentPolarFactor() {
        Random random = new Random(77);
        for (int[] shape : new int[][] {{1,1}, {5,3}, {3,5}, {7,7}}) {
            int rows = shape[0], cols = shape[1];
            ASHFloatOrthogonalizer workspace = new ASHFloatOrthogonalizer(rows, cols);
            double[] output = new double[rows * cols];
            for (int repeat = 0; repeat < 12; repeat++) {
                double[] input = new double[output.length];
                double[][] rounded = new double[rows][cols];
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                    input[r + c * rows] = random.nextGaussian();
                    rounded[r][c] = (float) input[r + c * rows];
                }
                double[] copy = input.clone();
                var svd = new SingularValueDecomposition(new Array2DRowRealMatrix(rounded, false));
                var expected = svd.getU().multiply(svd.getVT());
                Arrays.fill(output, Double.NaN);
                workspace.orthogonalize(input, output);
                assertArrayEquals(copy, input, 0);
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                    assertEquals(expected.getEntry(r, c), output[r + c * rows], 2e-5);
                }
                assertPartialIsometry(output, rows, cols);
            }
        }
    }

    @Test
    public void zeroAndRankDeficientInputsKeepAnOrthonormalCompletion() {
        for (int[] shape : new int[][] {{1,1}, {5,3}, {3,5}, {7,7}}) {
            int rows = shape[0], cols = shape[1];
            ASHFloatOrthogonalizer workspace = new ASHFloatOrthogonalizer(rows, cols);
            double[] input = new double[rows * cols], output = new double[input.length];
            workspace.orthogonalize(input, output);
            assertPartialIsometry(output, rows, cols);
            for (int r = 0; r < rows; r++) input[r] = r + 1;
            workspace.orthogonalize(input, output);
            assertPartialIsometry(output, rows, cols);
        }
    }

    @Test
    public void invalidInputsAreRejectedAndTheWorkspaceCanBeReusedAfterFailure() {
        assertThrows(IllegalArgumentException.class, () -> new ASHFloatOrthogonalizer(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new ASHFloatOrthogonalizer(1, -1));
        assertThrows(IllegalArgumentException.class, () -> new ASHFloatOrthogonalizer(50000, 50000));
        ASHFloatOrthogonalizer workspace = new ASHFloatOrthogonalizer(2, 2);
        double[] input = {1,0,0,1}, output = new double[4];
        assertThrows(NullPointerException.class, () -> workspace.orthogonalize(null, output));
        assertThrows(NullPointerException.class, () -> workspace.orthogonalize(input, null));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(new double[3], output));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(input, new double[3]));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(input, input));
        for (double invalid : new double[] {Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.MAX_VALUE}) {
            assertThrows(IllegalArgumentException.class,
                    () -> workspace.orthogonalize(new double[] {1,invalid,0,1}, output));
            workspace.orthogonalize(input, output);
            assertArrayEquals(input, output, 1e-6);
        }
    }

    private static void assertPartialIsometry(double[] matrix, int rows, int cols) {
        int rank = Math.min(rows, cols);
        for (int i = 0; i < rank; i++) for (int j = 0; j < rank; j++) {
            double dot = 0;
            for (int k = 0; k < Math.max(rows, cols); k++) {
                dot += rows >= cols ? matrix[k + i * rows] * matrix[k + j * rows]
                        : matrix[i + k * rows] * matrix[j + k * rows];
            }
            assertEquals(i == j ? 1 : 0, dot, 2e-5);
        }
    }
}
