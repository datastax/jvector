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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class TestAshRunWithoutTraining {

    public static void main(String[] args) {
        int originalDim = 1024;
        int quantizedDim = 128;

        Random rng = new Random(42);

        // Method under test
        AsymmetricHashing.StiefelTransform stiefelTransform =
                AsymmetricHashing.runWithoutTraining(originalDim, quantizedDim, rng);

        // Shapes
        System.out.println("A shape = " +
                stiefelTransform.rows + " x " + stiefelTransform.cols);
        System.out.println("W shape = " +
                stiefelTransform.W.getRowDimension() + " x " +
                stiefelTransform.W.getColumnDimension());

        // Reconstruct A as a RealMatrix *for testing only*
        RealMatrix A =
                MatrixUtils.createRealMatrix(stiefelTransform.AData);

        RealMatrix W = stiefelTransform.W;

        // Sanity check: a few entries of A
        System.out.println("First few entries of A:");
        for (int i = 0; i < Math.min(3, A.getRowDimension()); i++) {
            for (int j = 0; j < Math.min(3, A.getColumnDimension()); j++) {
                System.out.printf("%.4f ", A.getEntry(i, j));
            }
            System.out.println();
        }

        // Check orthogonality: WᵀW ≈ I
        RealMatrix I = W.transpose().multiply(W);
        System.out.printf("I(0,0)=%.5f, I(0,1)=%.5f%n",
                I.getEntry(0, 0), I.getEntry(0, 1));
    }
}
