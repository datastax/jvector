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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

/** Numeric layout, in-place quantization, workspace reuse and native buffer contracts. */
public class TestASHTrainingWorkspace {
    private static final double K_EPS = 1e-5;

    @Test
    public void transposedProductsMatchIndependentScalarProducts() throws Exception {
        Random random = new Random(41);
        for (int[] shape : new int[][] {{1, 1}, {1, 5}, {3, 7}, {19, 5}, {33, 8}}) {
            int rows = shape[0], dims = shape[1];
            double[] x = randomArray(rows * dims, random);
            double[] xCopy = x.clone();
            double[] rotation = randomArray(dims * dims, random);
            double[] rotationCopy = rotation.clone();
            double[] projected = new double[x.length];
            double[] update = new double[rotation.length];
            for (int repeat = 0; repeat < 3; repeat++) {
                Arrays.fill(projected, Double.NaN);
                product("projectTrainingRows", x, rows, dims, rotation, projected);
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < dims; col++) {
                        double expected = 0;
                        for (int k = 0; k < dims; k++) expected += x[row + k * rows] * rotation[k + col * dims];
                        assertEquals(expected, projected[row * dims + col], 1e-12);
                    }
                }
                Arrays.fill(update, Double.NaN);
                product("updateTrainingMatrix", x, rows, dims, projected, update);
                for (int row = 0; row < dims; row++) {
                    for (int col = 0; col < dims; col++) {
                        double expected = 0;
                        for (int k = 0; k < rows; k++) expected += x[k + row * rows] * projected[k * dims + col];
                        assertEquals(expected, update[row + col * dims], 1e-11);
                    }
                }
                assertArrayEquals(xCopy, x, 0);
                assertArrayEquals(rotationCopy, rotation, 0);
            }
        }
    }

    @Test
    public void contiguousQuantizationMatchesPreviousColumnMajorAlgorithmBitForBit() throws Exception {
        Random random = new Random(103);
        for (int bits = 1; bits <= 9; bits++) {
            for (int[] shape : new int[][] {{1, 1}, {3, 5}, {17, 9}, {35, 32}}) {
                int rows = shape[0], dims = shape[1];
                Object encoder = encoder(rows, dims, bits);
                for (int repeat = 0; repeat < 3; repeat++) {
                    double[] contiguous = randomArray(rows * dims, random);
                    // Zero/tiny rows and signed zero exercise normalization and sign boundaries.
                    for (int j = 0; j < dims; j++) contiguous[j] = j % 2 == 0 ? 0.0 : -0.0;
                    if (rows > 1) for (int j = 0; j < dims; j++) contiguous[dims + j] *= 1e-30;
                    double[] columnMajor = new double[contiguous.length];
                    for (int r = 0; r < rows; r++) for (int j = 0; j < dims; j++) columnMajor[r + j * rows] = contiguous[r * dims + j];
                    double[] expected = new double[contiguous.length];
                    referenceColumnMajor(columnMajor, rows, dims, bits, expected);
                    encode(encoder, contiguous);
                    for (int r = 0; r < rows; r++) for (int j = 0; j < dims; j++) {
                        assertEquals("bits=" + bits + " row=" + r + " dim=" + j,
                                Double.doubleToLongBits(expected[r + j * rows]), Double.doubleToLongBits(contiguous[r * dims + j]));
                    }
                }
            }
        }
    }

    @Test
    public void reusedPolarWorkspaceMatchesIndependentSvd() {
        Random random = new Random(77);
        for (int[] shape : new int[][] {{1, 1}, {5, 3}, {3, 5}, {4, 4}}) {
            int rows = shape[0], cols = shape[1];
            ASHOrthogonalizer workspace = new ASHOrthogonalizer(rows, cols);
            double[] destination = new double[rows * cols];
            for (int repeat = 0; repeat < 4; repeat++) {
                double[] input = randomArray(rows * cols, random);
                double[] copy = input.clone();
                double[][] matrix = new double[rows][cols];
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) matrix[r][c] = input[r + c * rows];
                var svd = new SingularValueDecomposition(new Array2DRowRealMatrix(matrix, false));
                var expected = svd.getU().multiply(svd.getVT());
                Arrays.fill(destination, Double.NaN);
                workspace.orthogonalize(input, destination);
                assertArrayEquals(copy, input, 0);
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) assertEquals(expected.getEntry(r, c), destination[r + c * rows], 1e-11);
                assertPartialIsometry(destination, rows, cols);
            }
            // The polar completion is not unique for rank-deficient inputs.
            double[] deficient = new double[rows * cols];
            for (int r = 0; r < rows; r++) deficient[r] = r + 1;
            workspace.orthogonalize(deficient, destination);
            assertPartialIsometry(destination, rows, cols);
            workspace.orthogonalize(new double[rows * cols], destination);
            assertPartialIsometry(destination, rows, cols);
        }
    }

    @Test
    public void polarWorkspaceRejectsInvalidBuffersAndRecoversAfterFailure() {
        assertThrows(IllegalArgumentException.class, () -> new ASHOrthogonalizer(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new ASHOrthogonalizer(1, -1));
        assertThrows(IllegalArgumentException.class, () -> new ASHOrthogonalizer(50000, 50000));
        ASHOrthogonalizer workspace = new ASHOrthogonalizer(2, 2);
        double[] input = {1, 0, 0, 1}, output = new double[4];
        assertThrows(NullPointerException.class, () -> workspace.orthogonalize(null, output));
        assertThrows(NullPointerException.class, () -> workspace.orthogonalize(input, null));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(new double[3], output));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(input, new double[3]));
        assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(input, input));
        for (double invalid : new double[] {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}) {
            double[] bad = {1, invalid, 0, 1};
            assertThrows(IllegalArgumentException.class, () -> workspace.orthogonalize(bad, output));
            workspace.orthogonalize(input, output);
            assertArrayEquals(input, output, 1e-12);
        }
    }

    @Test
    public void matrixAndEncoderContractsRejectInvalidShapesAndAliasing() throws Exception {
        double[] x = new double[6], square = new double[4];
        assertThrows(IllegalArgumentException.class, () -> product("projectTrainingRows", x, 0, 2, square, x));
        assertThrows(IllegalArgumentException.class, () -> product("projectTrainingRows", x, 3, 2, square, new double[5]));
        assertThrows(IllegalArgumentException.class, () -> product("projectTrainingRows", x, 3, 2, square, x));
        assertThrows(IllegalArgumentException.class, () -> product("updateTrainingMatrix", x, 3, 2, x, x));
        assertThrows(IllegalArgumentException.class, () -> product("updateTrainingMatrix", square, 2, 2, square, square));
        assertThrows(NullPointerException.class, () -> product("projectTrainingRows", null, 3, 2, square, x));
        assertThrows(IllegalArgumentException.class, () -> encoder(0, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> encoder(1, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> encoder(1, 2, 10));
        assertThrows(IllegalArgumentException.class, () -> encoder(50000, 50000, 2));
        Object encoder = encoder(3, 2, 2);
        assertThrows(NullPointerException.class, () -> encode(encoder, null));
        assertThrows(IllegalArgumentException.class, () -> encode(encoder, new double[5]));
    }

    @Test
    public void pcaWorkspaceQueryAndDecompositionLeaveAllInputUntouched() throws Exception {
        Random random = new Random(82);
        for (int[] shape : new int[][] {{7, 3}, {2, 5}, {1, 1}}) {
            double[] input = randomArray(shape[0] * shape[1], random);
            double[] copy = input.clone();
            call(method("computeNativeVt", double[].class, int.class, int.class), null, input, shape[0], shape[1]);
            assertArrayEquals(copy, input, 0);
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
            assertEquals(i == j ? 1 : 0, dot, 1e-11);
        }
    }

    private static double[] randomArray(int count, Random random) {
        double[] values = new double[count];
        for (int i = 0; i < count; i++) values[i] = random.nextGaussian();
        return values;
    }

    private static Method method(String name, Class<?>... types) throws NoSuchMethodException {
        Method method = AsymmetricHashing.class.getDeclaredMethod(name, types);
        method.setAccessible(true);
        return method;
    }

    private static Object call(Method method, Object target, Object... args) throws Exception {
        try { return method.invoke(target, args); }
        catch (InvocationTargetException failure) {
            if (failure.getCause() instanceof Exception) throw (Exception) failure.getCause();
            throw (Error) failure.getCause();
        }
    }

    private static void product(String name, double[] x, int rows, int dims, double[] other, double[] output) throws Exception {
        call(method(name, double[].class, int.class, int.class, double[].class, double[].class), null, x, rows, dims, other, output);
    }

    private static Object encoder(int rows, int dims, int bits) throws Exception {
        Class<?> type = Class.forName(AsymmetricHashing.class.getName() + "$TrainingDirectionEncoder");
        Constructor<?> constructor = type.getDeclaredConstructor(int.class, int.class, int.class);
        constructor.setAccessible(true);
        try { return constructor.newInstance(rows, dims, bits); }
        catch (InvocationTargetException failure) {
            if (failure.getCause() instanceof Exception) throw (Exception) failure.getCause();
            throw (Error) failure.getCause();
        }
    }

    private static void encode(Object encoder, double[] input) throws Exception {
        Method method = encoder.getClass().getDeclaredMethod("encodeInPlace", double[].class);
        method.setAccessible(true);
        call(method, encoder, (Object) input);
    }

    private static double scale(double[] values, int dims, int bits) {
        try { return (double) call(method("computeOptimalScalingFactor", double[].class, int.class, int.class), null, values, dims, bits); }
        catch (Exception failure) { throw new AssertionError(failure); }
    }

    // Retain the former column-major loop as the numerical regression oracle.
    private static void referenceColumnMajor(
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

        // Each task owns a contiguous range of training rows and its scratch arrays.
        // computeOptimalScalingFactor already keeps its event buffers thread-local.
        final int workers = Math.min(rows, Math.min(16, Runtime.getRuntime().availableProcessors()));
        IntStream.range(0, workers).parallel().forEach(worker -> {
            double[] absNorm = new double[dims];
            double[] mag = new double[dims];
            int start = worker * rows / workers;
            int end = (worker + 1) * rows / workers;

            for (int r = start; r < end; r++) {
                double norm2 = 0.0;
                for (int j = 0; j < dims; j++) {
                    double v = xCol[r + j * rows];
                    norm2 += v * v;
                }

                double invNorm = 1.0 / Math.sqrt(Math.max(norm2, 1e-20));
                for (int j = 0; j < dims; j++) {
                    absNorm[j] = Math.abs(xCol[r + j * rows]) * invNorm;
                }

                double t = scale(absNorm, dims, bitsPerDimension);
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
        });
    }

}
