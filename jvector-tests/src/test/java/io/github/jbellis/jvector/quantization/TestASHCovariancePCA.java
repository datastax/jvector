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
import java.lang.reflect.InvocationTargetException;
import java.util.Random;
import static org.junit.Assert.*;

/** Covariance PCA against independent SVD subspaces and direct-SVD fallback cases. */
public class TestASHCovariancePCA {
    @Test public void principalAxesAndInputArePreserved() {
        double[] x = {3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0};
        double[] copy = x.clone(), basis = ASHCovariancePCA.projection(x, 5, 3, 2);
        assertNotNull(basis);
        assertOrthonormal(basis, 3, 2);
        assertEquals(1.0, Math.abs(basis[0]), 1e-12);
        assertEquals(1.0, Math.abs(basis[4]), 1e-12);
        assertArrayEquals(copy, x, 0);
    }

    @Test public void retainedSubspaceMatchesIndependentDirectSvd() {
        Random random = new Random(103);
        for (int[] shape : new int[][] {{13, 5, 2}, {31, 7, 4}, {21, 3, 1}}) {
            int rows = shape[0], dims = shape[1], latent = shape[2];
            double[] x = new double[rows * dims];
            double[][] matrix = new double[rows][dims];
            for (int r = 0; r < rows; r++) for (int c = 0; c < dims; c++)
                x[r + c * rows] = matrix[r][c] = random.nextGaussian();
            double[] copy = x.clone(), basis = ASHCovariancePCA.projection(x, rows, dims, latent);
            assertNotNull(basis);
            var v = new SingularValueDecomposition(new Array2DRowRealMatrix(matrix, false)).getV();
            assertOrthonormal(basis, dims, latent);
            // Compare projectors, independent of eigenvector signs or rotations.
            for (int r = 0; r < dims; r++) for (int c = 0; c < dims; c++) {
                double actual = 0, expected = 0;
                for (int j = 0; j < latent; j++) {
                    actual += basis[r + j * dims] * basis[c + j * dims];
                    expected += v.getEntry(r, j) * v.getEntry(c, j);
                }
                assertEquals(expected, actual, 1e-10);
            }
            assertArrayEquals(copy, x, 0);
        }
    }

    @Test public void uncertainRankAndBoundaryUseDirectSvd() throws Exception {
        double[][] inputs = {new double[15], new double[] {3,0,0, 0,0,0, 0,0,0},
                new double[] {3,0,0, 0,1,0, 0,0,1}};
        for (double[] x : inputs) {
            int rows = x.length / 3;
            assertNull(ASHCovariancePCA.projection(x, rows, 3, 2));
            assertOrthonormal(basis(x, rows, 3, 2), 3, 2);
        }
    }

    @Test public void wideInputsRetainNullSpaceCompletion() throws Exception {
        double[] x = {3, 0, 4, 0, 0, 0, 0, 0};
        double[] copy = x.clone();
        assertNull(ASHCovariancePCA.projection(x, 2, 4, 3));
        assertOrthonormal(basis(x, 2, 4, 3), 4, 3);
        assertArrayEquals(copy, x, 0);
    }

    @Test public void illConditionedRetainedTailUsesDirectSvd() throws Exception {
        double[] x = {1,0,0, 0,1e-10,0, 0,0,1e-15};
        assertNull(ASHCovariancePCA.projection(x, 3, 3, 2));
        assertOrthonormal(basis(x, 3, 3, 2), 3, 2);
    }

    @Test public void repeatedEigenvaluesWithinRetainedSubspaceAreAllowed() {
        double[] x = {2,0,0, 0,2,0, 0,0,1};
        for (int latent : new int[] {2, 3}) {
            double[] basis = ASHCovariancePCA.projection(x, 3, 3, latent);
            assertNotNull(basis);
            assertOrthonormal(basis, 3, latent);
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                double product = 0;
                for (int j = 0; j < latent; j++) product += basis[r + j * 3] * basis[c + j * 3];
                assertEquals(r == c && r < latent ? 1 : 0, product, 1e-12);
            }
        }
    }

    @Test public void discardedNullDirectionsDoNotForceFallback() {
        double[] x = {3,0,0, 0,2,0, 0,0,0};
        double[] basis = ASHCovariancePCA.projection(x, 3, 3, 2);
        assertNotNull(basis);
        assertOrthonormal(basis, 3, 2);
        assertEquals(1, Math.abs(basis[0]), 1e-12);
        assertEquals(1, Math.abs(basis[4]), 1e-12);
    }

    @Test public void oneDimensionalInputHasACompleteBasis() {
        double[] x = {0, 2, -3}, copy = x.clone();
        double[] basis = ASHCovariancePCA.projection(x, 3, 1, 1);
        assertNotNull(basis);
        assertOrthonormal(basis, 1, 1);
        assertArrayEquals(copy, x, 0);
    }

    @Test public void invalidShapesAndNonFiniteInputAreRejected() {
        assertThrows(NullPointerException.class, () -> ASHCovariancePCA.projection(null, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[1], 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[1], 1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[1], 1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[2], 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[0], 50000, 50000, 1));
        for (double invalid : new double[] {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.MAX_VALUE})
            assertThrows(IllegalArgumentException.class, () -> ASHCovariancePCA.projection(new double[] {invalid}, 1, 1, 1));
    }

    private static double[] basis(double[] x, int rows, int dims, int latent) throws Exception {
        var method = AsymmetricHashing.class.getDeclaredMethod("computePcaBasis", double[].class, int.class, int.class, int.class);
        method.setAccessible(true);
        try { return (double[]) method.invoke(null, x, rows, dims, latent); }
        catch (InvocationTargetException failure) {
            if (failure.getCause() instanceof Exception) throw (Exception) failure.getCause();
            throw (Error) failure.getCause();
        }
    }
    private static void assertOrthonormal(double[] basis, int dims, int latent) {
        for (int i = 0; i < latent; i++) for (int j = 0; j < latent; j++) {
            double dot = 0;
            for (int k = 0; k < dims; k++) dot += basis[k + i * dims] * basis[k + j * dims];
            assertEquals(i == j ? 1 : 0, dot, 1e-11);
        }
    }
}
