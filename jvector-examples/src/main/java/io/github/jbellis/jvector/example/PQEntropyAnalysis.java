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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * Diagnostic tool to analyze Product Quantization centroid assignment entropy.
 *
 * <p>
 * Explains PQ scan-time behavior by measuring how uniformly vectors are
 * assigned to centroids in each subspace.
 * </p>
 *
 * <p>
 * High entropy ⇒ many centroids exercised ⇒ poor cache locality and slower ADC.
 * Low entropy ⇒ concentrated centroids ⇒ faster PQ scans.
 * </p>
 *
 * <p>
 * This is NOT a benchmark:
 * <ul>
 *   <li>No timing loops</li>
 *   <li>No scoring</li>
 *   <li>No SIMD assumptions</li>
 * </ul>
 * </p>
 */
public class PQEntropyAnalysis {

    public static void analyze(
            String label,
            String baseVectorsPath,
            int encodedBits
    ) throws IOException {

        System.out.println("--------------------------------------------------");
        System.out.println("PQ Entropy Analysis: " + label);

        // ------------------------------------------------------------
        // Load base vectors
        // ------------------------------------------------------------
        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(baseVectorsPath);
        int dimension = vectors.get(0).length();

        System.out.println("Vectors: " + vectors.size());
        System.out.println("Dimension: " + dimension);
        System.out.println("Target compression: " + encodedBits + " bits");

        // ------------------------------------------------------------
        // PQ parameters (match DistancesPQ)
        // ------------------------------------------------------------
        // ASH: encodedBits = d / 4  → bytes = d / 32
        // PQ: 1 byte per subspace   → M = d / 32
        final int M = Math.max(1, encodedBits / 8);
        final int K = 256;
        final boolean globallyCenter = false;
        final float anisotropicThreshold = 0.0f;

        System.out.println("PQ params:");
        System.out.println("  M (subspaces) = " + M);
        System.out.println("  K (clusters)  = " + K);

        // ------------------------------------------------------------
        // Build PQ
        // ------------------------------------------------------------
        RandomAccessVectorValues ravv =
                new ListRandomAccessVectorValues(vectors, dimension);

        ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();

        ProductQuantization pq = ProductQuantization.compute(
                ravv,
                M,
                K,
                globallyCenter,
                anisotropicThreshold,
                simdExecutor,
                ForkJoinPool.commonPool()
        );

        PQVectors pqVecs = pq.encodeAll(ravv, simdExecutor);

        // ------------------------------------------------------------
        // Count centroid assignments
        // ------------------------------------------------------------
        long[][] counts = new long[M][K];

        for (int i = 0; i < pqVecs.count(); i++) {
            var code = pqVecs.get(i);
            for (int m = 0; m < M; m++) {
                int c = Byte.toUnsignedInt(code.get(m));
                counts[m][c]++;
            }
        }

        // ------------------------------------------------------------
        // Compute entropy per subspace
        // ------------------------------------------------------------
        double totalEntropy = 0.0;

        for (int m = 0; m < M; m++) {
            long total = 0;
            for (int k = 0; k < K; k++) {
                total += counts[m][k];
            }

            double entropy = 0.0;
            for (int k = 0; k < K; k++) {
                long cnt = counts[m][k];
                if (cnt == 0) continue;
                double p = (double) cnt / total;
                entropy -= p * (Math.log(p) / Math.log(2));
            }

            totalEntropy += entropy;
        }

        double avgEntropy = totalEntropy / M;
        double effectiveCentroids = Math.pow(2.0, avgEntropy);

        // ------------------------------------------------------------
        // Report
        // ------------------------------------------------------------
        System.out.printf("Subspaces (M): %d%n", M);
        System.out.printf("Centroids per subspace (K): %d%n", K);
        System.out.printf("Average entropy per subspace: %.4f bits%n", avgEntropy);
        System.out.printf("Effective centroids used: %.1f%n", effectiveCentroids);
    }

    // ------------------------------------------------------------------
    // Dataset runners (aligned with DistancesPQ / DistancesASH)
    // ------------------------------------------------------------------

    public static void runCohere100k() throws IOException {
        analyze(
                "Cohere-100k",
                "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec",
                256
        );
    }

    public static void runAda002() throws IOException {
        analyze(
                "ada_002",
                "./fvec/ada-002/ada_002_100000_base_vectors.fvec",
                384
        );
    }

    public static void runOpenAI1536() throws IOException {
        analyze(
                "text-embedding-3-large-1536",
                "./fvec/openai-v3-large-1536-100k/text-embedding-3-large_1536_100000_base_vectors.fvec",
                384
        );
    }

    public static void runOpenAI3072() throws IOException {
        analyze(
                "text-embedding-3-large-3072",
                "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec",
                768
        );
    }

    public static void main(String[] args) throws IOException {
        runCohere100k();
        runAda002();
        runOpenAI1536();
        runOpenAI3072();
    }
}

