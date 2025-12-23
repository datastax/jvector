package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import org.apache.commons.math3.linear.*;
import java.util.Random;

public class TestAshRunWithoutTraining {

    public static void main(String[] args) {
        int d = 1024;     // input dimension
        int nDims = 128;  // target dimension

        // Dummy data matrix X only to scrape shape
        RealMatrix X = MatrixUtils.createRealMatrix(10, d);  // (10 × 1024)

        Random rng = new Random(42);

        // Method under test
        AsymmetricHashing.StiefelTransform stiefelTransform =
                AsymmetricHashing.runWithoutTraining(X, nDims, rng);

        // Inspect shapes and a few entries
        RealMatrix A = stiefelTransform.A;
        RealMatrix W = stiefelTransform.W;

        System.out.println("A shape = " + A.getRowDimension() + " x " + A.getColumnDimension());
        System.out.println("W shape = " + W.getRowDimension() + " x " + W.getColumnDimension());

        // Sanity check....
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
