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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import static java.lang.Math.abs;

/**
 * Matrix object where each row is a VectorFloat; this makes multiplication of a matrix by a vector
 * a series of efficient dot products.
 */
public class Matrix {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    VectorFloat<?>[] data;

    public Matrix(int m, int n) {
        this(m, n, true);
    }

    public Matrix(int m, int n, boolean allocateZeroed) {
        data = new VectorFloat[m];
        if (allocateZeroed) {
            for (int i = 0; i < m; i++) {
                data[i] = vts.createFloatVector(n);
            }
        }
    }

    public float get(int i, int j) {
        return data[i].get(j);
    }

    public void set(int i, int j, float value) {
        data[i].set(j, value);
    }

    public boolean isIsomorphicWith(Matrix other) {
        return data.length == other.data.length && data[0].length() == other.data[0].length();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (VectorFloat<?> row : data) {
            sb.append(row.toString());
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Inverts a square matrix using gaussian elimination.
     * @return The inverse of the matrix.
     */
    public Matrix invert() {
        if (data.length == 0 || data.length != data[0].length()) {
            throw new IllegalArgumentException("matrix must be square");
        }

        int N = data.length;

        // Initialize augmented matrix (original matrix on the left, identity matrix on the right)
        var augmented = new Matrix(N, 2 * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                augmented.set(i, j, get(i, j));
                augmented.set(i, j + N, (i == j) ? 1 : 0);
            }
        }

        // Perform Gaussian elimination with pivoting
        for (int i = 0; i < N; i++) {
            // Pivot: Find the row with the largest absolute value in column i to promote numerical stability
            int maxRow = i;
            for (int k = i + 1; k < N; k++) {
                if (abs(augmented.get(k, i)) > abs(augmented.get(maxRow, i))) {
                    maxRow = k;
                }
            }

            // Swap the current row with the maxRow
            var temp = augmented.data[i];
            augmented.data[i] = augmented.data[maxRow];
            augmented.data[maxRow] = temp;

            // Scale pivot row
            VectorUtil.scale(augmented.data[i], 1 / augmented.get(i, i));

            // Eliminate below and above pivot
            for (int k = 0; k < N; k++) {
                if (k != i) {
                    float factor = augmented.get(k, i);
                    for (int j = 0; j < 2 * N; j++) {
                        augmented.addTo(k, j, -factor * augmented.get(i, j));
                    }
                }
            }
        }

        // Extract inverse matrix
        var inverse = new Matrix(N, N);
        for (int i = 0; i < N; i++) {
            inverse.data[i].copyFrom(augmented.data[i], N, 0, N);
        }

        return inverse;
    }

    public void addTo(int i, int j, float delta) {
        data[i].set(j, data[i].get(j) + delta);
    }

    public void addInPlace(Matrix other) {
        if (!this.isIsomorphicWith(other)) {
            throw new IllegalArgumentException("matrix dimensions differ for " + this + "!=" + other);
        }

        for (int i = 0; i < this.data.length; i++) {
            VectorUtil.addInPlace(this.data[i], other.data[i]);
        }
    }

    public VectorFloat<?> multiply(VectorFloat<?> v) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Cannot multiply empty matrix");
        }
        if (v.length() == 0) {
            throw new IllegalArgumentException("Cannot multiply empty vector");
        }

        var result = vts.createFloatVector(data.length);
        for (int i = 0; i < data.length; i++) {
            result.set(i, VectorUtil.dotProduct(data[i], v));
        }
        return result;
    }

    public static Matrix outerProduct(VectorFloat<?> a, VectorFloat<?> b) {
        var result = new Matrix(a.length(), b.length(), false);

        for (int i = 0; i < a.length(); i++) {
            var rowI = b.copy();
            VectorUtil.scale(rowI, a.get(i));
            result.data[i] = rowI;
        }

        return result;
    }

    public void scale(float multiplier) {
        for (var row : data) {
            VectorUtil.scale(row, multiplier);
        }
    }

    public boolean equals(Object obj) {
        if (!(obj instanceof Matrix)) {
            return false;
        }

        var other = (Matrix) obj;
        if (data.length != other.data.length) {
            return false;
        }
        for (int i = 0; i < data.length; i++) {
            if (!data[i].equals(other.data[i])) {
                return false;
            }
        }
        return true;
    }

    public static Matrix from(float[][] values) {
        var result = new Matrix(values.length, values[0].length, false);
        for (int i = 0; i < values.length; i++) {
            result.data[i] = vts.createFloatVector(values[i]);
        }
        return result;
    }
}
