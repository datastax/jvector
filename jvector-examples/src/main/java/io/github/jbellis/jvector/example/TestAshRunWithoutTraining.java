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
