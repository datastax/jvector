/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.quantization;

import org.junit.Test;

import java.lang.reflect.InvocationTargetException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** Checks the PCA basis contract independently of subsequent ITQ rotations. */
public class TestASHPcaTraining {
    @Test
    public void tallMatrixPreservesPrincipalAxes() throws Exception {
        // Column-major diagonal matrix with two additional zero rows.
        double[] x = {3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0};
        double[] vt = pca(x, 5, 3);
        assertOrthonormal(vt, 3);
        for (int axis = 0; axis < 3; axis++) {
            assertEquals(1.0, Math.abs(vt[axis + axis * 3]), 1e-12);
        }
        assertEquals(3.0, x[0], 0.0); // LAPACK must only overwrite the scratch copy.
    }

    @Test
    public void wideRankDeficientMatrixIncludesNullSpace() throws Exception {
        double[] vt = pca(new double[] {3, 0, 4, 0, 0, 0, 0, 0}, 2, 4);
        assertEquals(16, vt.length);
        assertOrthonormal(vt, 4);
        assertEquals(0.6, Math.abs(vt[0]), 1e-12);
        assertEquals(0.8, Math.abs(vt[4]), 1e-12);
        assertEquals(0.0, vt[8], 1e-12);
        assertEquals(0.0, vt[12], 1e-12);
    }

    @Test
    public void nonFiniteInputIsRejectedBeforeLapack() {
        assertThrows(IllegalArgumentException.class,
                () -> pca(new double[] {1, Double.NaN, 0, 1}, 2, 2));
    }

    private static double[] pca(double[] x, int rows, int cols) throws ReflectiveOperationException {
        var method = AsymmetricHashing.class.getDeclaredMethod("computeNativeVt", double[].class, int.class, int.class);
        method.setAccessible(true);
        try {
            return (double[]) method.invoke(null, x, rows, cols);
        } catch (InvocationTargetException error) {
            if (error.getCause() instanceof RuntimeException) throw (RuntimeException) error.getCause();
            throw error;
        }
    }

    private static void assertOrthonormal(double[] vt, int dimension) {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double dot = 0.0;
                for (int c = 0; c < dimension; c++) dot += vt[i + c * dimension] * vt[j + c * dimension];
                assertEquals(i == j ? 1.0 : 0.0, dot, 1e-12);
            }
        }
    }
}
