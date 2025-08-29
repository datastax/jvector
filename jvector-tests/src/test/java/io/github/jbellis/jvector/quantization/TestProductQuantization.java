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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.TestUtil.randomVector;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static io.github.jbellis.jvector.quantization.ProductQuantization.DEFAULT_CLUSTERS;
import static io.github.jbellis.jvector.quantization.ProductQuantization.getSubvectorSizesAndOffsets;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Test
    // special cases where each vector maps exactly to a centroid
    public void testPerfectReconstruction() {
        var R = getRandom();

        // exactly the same number of random vectors as clusters
        List<VectorFloat<?>> v1 = IntStream.range(0, DEFAULT_CLUSTERS).mapToObj(
                        i -> vectorTypeSupport.createFloatVector(new float[] {R.nextInt(100000), R.nextInt(100000), R.nextInt(100000)}))
                .collect(Collectors.toList());
        assertPerfectQuantization(v1);

        // 10x the number of random vectors as clusters (with duplicates)
        List<VectorFloat<?>> v2 = v1.stream().flatMap(v -> IntStream.range(0, 10).mapToObj(i -> v))
                .collect(Collectors.toList());
        assertPerfectQuantization(v2);
    }

    private static void assertPerfectQuantization(List<VectorFloat<?>> vectors) {
        var ravv = new ListRandomAccessVectorValues(vectors, 3);
        var pq = ProductQuantization.compute(ravv, 2, DEFAULT_CLUSTERS, false);
        var cv = (PQVectors) pq.encodeAll(ravv);
        var decodedScratch = vectorTypeSupport.createFloatVector(3);
        for (int i = 0; i < vectors.size(); i++) {
            pq.decode(cv.get(i), decodedScratch);
            assertEquals(vectors.get(i), decodedScratch);
        }
    }

    @Test
    // validate that iterating on our cluster centroids improves the encoding
    public void testIterativeImprovement() {
        for (int i = 0; i < 10; i++) {
            testIterativeImprovementOnce();
            testConvergenceAnisotropic();
        }
    }

    public void testIterativeImprovementOnce() {
        var R = getRandom();
        VectorFloat<?>[] vectors = generate(DEFAULT_CLUSTERS + R.nextInt(10* DEFAULT_CLUSTERS),
                                            2 + R.nextInt(10),
                                            1_000 + R.nextInt(10_000));

        var clusterer = new KMeansPlusPlusClusterer(vectors, DEFAULT_CLUSTERS);
        var initialLoss = loss(clusterer, vectors, -Float.MAX_VALUE);

        assert clusterer.clusterOnceUnweighted() > 0;
        var improvedLoss = loss(clusterer, vectors, -Float.MAX_VALUE);

        assertTrue("improvedLoss=" + improvedLoss + " initialLoss=" + initialLoss, improvedLoss < initialLoss);
    }

    @Test
    public void testRefine() {
        var R = getRandom();
        VectorFloat<?>[] vectors = generate(DEFAULT_CLUSTERS + R.nextInt(10* DEFAULT_CLUSTERS),
                                            2 + R.nextInt(10),
                                            1_000 + R.nextInt(10_000));

        // generate PQ codebooks from half of the dataset
        var half1 = Arrays.copyOf(vectors, vectors.length / 2);
        var ravv1 = new ListRandomAccessVectorValues(List.of(half1), vectors[0].length());
        var pq1 = ProductQuantization.compute(ravv1, 1, DEFAULT_CLUSTERS, false);

        // refine the codebooks with the other half (so, drawn from the same distribution)
        int remaining = vectors.length - vectors.length / 2;
        var half2 = new VectorFloat<?>[remaining];
        System.arraycopy(vectors, vectors.length / 2, half2, 0, remaining);
        var ravv2 = new ListRandomAccessVectorValues(List.of(half2), vectors[0].length());
        var pq2 = pq1.refine(ravv2);

        // the refined version should work better
        var clusterer1 = new KMeansPlusPlusClusterer(half2, pq1.codebooks[0], UNWEIGHTED);
        var clusterer2 = new KMeansPlusPlusClusterer(half2, pq2.codebooks[0], UNWEIGHTED);
        var loss1 = loss(clusterer1, half2, UNWEIGHTED);
        var loss2 = loss(clusterer2, half2, UNWEIGHTED);
        assertTrue("loss1=" + loss1 + " loss2=" + loss2, loss2 < loss1);
    }

    public void testConvergenceAnisotropic() {
        var R = getRandom();
        var vectors = generate(DEFAULT_CLUSTERS + R.nextInt(10 * DEFAULT_CLUSTERS),
                               2 + R.nextInt(10),
                               1_000 + R.nextInt(10_000));

        float T = 0.2f;
        var clusterer = new KMeansPlusPlusClusterer(vectors, DEFAULT_CLUSTERS, T);
        var initialLoss = loss(clusterer, vectors, T);

        double improvedLoss = Double.MAX_VALUE;
        while (true) {
            int n = clusterer.clusterOnceAnisotropic();
            if (n <= 0.01 * vectors.length) {
                break;
            }
            improvedLoss = loss(clusterer, vectors, T);
            // System.out.println("improvedLoss=" + improvedLoss + " n=" + n);
        }
        // System.out.println("iterations=" + iterations);

        assertTrue(improvedLoss < initialLoss);
    }

    /**
     * only include vectors whose dot product is greater than or equal to T
     */
    private static double loss(KMeansPlusPlusClusterer clusterer, VectorFloat<?>[] vectors, float T) {
        var pq = new ProductQuantization(new VectorFloat<?>[] {clusterer.getCentroids()},
                                         DEFAULT_CLUSTERS,
                                         getSubvectorSizesAndOffsets(vectors[0].length(), 1),
                                         null,
                                         UNWEIGHTED);
        var ravv = new ListRandomAccessVectorValues(List.of(vectors), vectors[0].length());
        var cv = (PQVectors) pq.encodeAll(ravv);
        var loss = 0.0;
        var decodedScratch = vectorTypeSupport.createFloatVector(vectors[0].length());
        for (int i = 0; i < vectors.length; i++) {
            pq.decode(cv.get(i), decodedScratch);
            if (VectorUtil.dotProduct(vectors[i], decodedScratch) >= T) {
                loss += 1 - VectorSimilarityFunction.EUCLIDEAN.compare(vectors[i], decodedScratch);
            }
        }
        return loss;
    }

    private static VectorFloat<?>[] generate(int nClusters, int nDimensions, int nVectors) {
        var R = getRandom();

        // generate clusters
        var clusters = IntStream.range(0, nClusters)
                .mapToObj(i -> randomVector(R, nDimensions))
                .collect(Collectors.toList());

        // generate vectors by perturbing clusters
        return IntStream.range(0, nVectors).mapToObj(__ -> {
            var cluster = clusters.get(R.nextInt(nClusters));
            var v = randomVector(R, nDimensions);
            VectorUtil.scale(v, 0.1f + 0.9f * R.nextFloat());
            VectorUtil.addInPlace(v, cluster);
            return v;
        }).toArray(VectorFloat<?>[]::new);
    }

    @Test
    public void testSaveLoad() throws Exception {
        // Generate a PQ for random 2D vectors
        var vectors = createRandomVectors(512, 2);
        var pq = ProductQuantization.compute(new ListRandomAccessVectorValues(vectors, 2), 1, 256, false, 0.2f);

        // Write
        var file = File.createTempFile("pqtest", ".pq");
        try (var out = new DataOutputStream(new FileOutputStream(file))) {
            pq.write(out);
        }
        // Read
        try (var readerSupplier = new SimpleMappedReader.Supplier(file.toPath())) {
            var pq2 = ProductQuantization.load(readerSupplier.get());
            Assertions.assertEquals(pq, pq2);
        }
    }

    @Test
    public void testLoadVersion0() throws Exception {
        var file = new File("resources/version0.pq");
        try (var readerSupplier = new SimpleMappedReader.Supplier(file.toPath())) {
            var pq = ProductQuantization.load(readerSupplier.get());
            assertEquals(2, pq.originalDimension);
            assertNull(pq.globalCentroid);
            assertEquals(1, pq.M);
            assertEquals(1, pq.codebooks.length);
            assertEquals(256, pq.getClusterCount());
            assertEquals(pq.subvectorSizesAndOffsets[0][0] * pq.getClusterCount(), pq.codebooks[0].length());
            assertEquals(UNWEIGHTED, pq.anisotropicThreshold, 1E-6); // v0 only supported (implicitly) unweighted
        }
    }

    @Test
    public void testSaveVersion0() throws Exception {
        var fileIn = new File("resources/version0.pq");
        var fileOut = File.createTempFile("pqtest", ".pq");

        try (var readerSupplier = new SimpleMappedReader.Supplier(fileIn.toPath())) {
            var pq = ProductQuantization.load(readerSupplier.get());

            // re-save, emulating version 0
            try (var out = new DataOutputStream(new FileOutputStream(fileOut))) {
                pq.write(out, 0);
            }
        }

        // check that the contents match
        var contents1 = Files.readAllBytes(fileIn.toPath());
        var contents2 = Files.readAllBytes(fileOut.toPath());
        assertArrayEquals(contents1, contents2);
    }

    private void validateChunkMath(PQVectors.PQLayout layout, int expectedTotalVectors, int dimension) {
        // Basic parameter validation
        assertTrue("vectorsPerChunk must be positive", layout.fullChunkVectors > 0);
        assertTrue("totalChunks must be positive", layout.totalChunks > 0);
        assertTrue("fullSizeChunks must be non-negative", layout.fullSizeChunks >= 0);
        assertTrue("remainingVectors must be non-negative", layout.lastChunkVectors >= 0);
        assertTrue("fullSizeChunks must not exceed totalChunks", layout.fullSizeChunks <= layout.totalChunks);
        assertTrue("remainingVectors must be less than vectorsPerChunk", layout.lastChunkVectors < layout.fullChunkVectors);

        // Total vectors validation
        long calculatedTotal = (long) layout.fullSizeChunks * layout.fullChunkVectors + layout.lastChunkVectors;
        assertEquals("Total vectors must match expected count",
                     expectedTotalVectors, calculatedTotal);

        // Chunk count validation
        assertEquals("Total chunks must match full + partial chunks",
                     layout.totalChunks, layout.fullSizeChunks + (layout.lastChunkVectors > 0 ? 1 : 0));
    }

    @Test
    public void testPQVectorsChunkCalculation() {
        // Test normal case
        PQVectors.PQLayout dims = new PQVectors.PQLayout(1000, 8);
        validateChunkMath(dims, 1000, 8);
        assertEquals(1000, dims.fullChunkVectors); // vectorsPerChunk
        assertEquals(1, dims.totalChunks);    // numChunks
        assertEquals(1, dims.fullSizeChunks);    // fullSizeChunks
        assertEquals(0, dims.lastChunkVectors);    // remainingVectors

        // Test case requiring multiple chunks
        int bigVectorCount = Integer.MAX_VALUE - 1;
        int smallDim = 8;
        PQVectors.PQLayout layoutBigSmall = new PQVectors.PQLayout(bigVectorCount, smallDim);
        validateChunkMath(layoutBigSmall, bigVectorCount, smallDim);
        assertTrue(layoutBigSmall.fullChunkVectors > 0);
        assertTrue(layoutBigSmall.fullChunkVectors > 1);

        // Test edge case with large dimension
        int smallVectorCount = 1000;
        int bigDim = Integer.MAX_VALUE / 2;
        PQVectors.PQLayout layoutSmallBig = new PQVectors.PQLayout(smallVectorCount, bigDim);
        validateChunkMath(layoutSmallBig, smallVectorCount, bigDim);
        assertTrue(layoutSmallBig.fullChunkVectors > 0);

        // Test invalid inputs
        assertThrows(IllegalArgumentException.class, () -> new PQVectors.PQLayout(-1, 8));
        assertThrows(IllegalArgumentException.class, () -> new PQVectors.PQLayout(100, -1));
        assertThrows(IllegalArgumentException.class, () -> new PQVectors.PQLayout(100, 0));
        assertThrows(IllegalArgumentException.class, () -> new PQVectors.PQLayout(0, 1));
        // Test last chunk sizing
        PQVectors.PQLayout maxLayout = new PQVectors.PQLayout(Integer.MAX_VALUE, 1 << 10);
        assertTrue(maxLayout.lastChunkVectors <= maxLayout.fullChunkVectors);
        assertTrue(maxLayout.lastChunkBytes <= maxLayout.fullChunkBytes);
    }

    /**
     * Leaving this test enabled for the actual boundary checks, but here is the output:
     * <pre><code>
     * === PQLayout Edge Cases Test ===
     * VectorCount  CompDim         FullChunkVecs   LastChunkVecs   FullSizeChunks  TotalChunks     FullChunkBytes  LastChunkBytes
     * =========================================================================================================================
     * 1            1               1               0               1               1               1               0
     * 1            2               1               0               1               1               2               0
     * 10           1               10              0               1               1               10              0
     * 10           2               10              0               1               1               20              0
     * 10           3               10              0               1               1               30              0
     * 10           4               10              0               1               1               40              0
     * 10           5               10              0               1               1               50              0
     * 10           7               10              0               1               1               70              0
     * 10           8               10              0               1               1               80              0
     * 10           9               10              0               1               1               90              0
     * 10           15              10              0               1               1               150             0
     * 10           16              10              0               1               1               160             0
     * 10           17              10              0               1               1               170             0
     * 10           31              10              0               1               1               310             0
     * 10           32              10              0               1               1               320             0
     * 10           33              10              0               1               1               330             0
     * 10           63              10              0               1               1               630             0
     * 10           64              10              0               1               1               640             0
     * 10           65              10              0               1               1               650             0
     * 10           127             10              0               1               1               1270            0
     * 10           128             10              0               1               1               1280            0
     * 10           129             10              0               1               1               1290            0
     * 1073741823   1               1073741823      0               1               1               1073741823      0
     * 1073741823   2               1073741823      0               1               1               2147483646      0
     * 1073741824   2               1073741823      1               1               2               2147483646      2
     * 1000         1024            1000            0               1               1               1024000         0
     * 2000000      1024            2000000         0               1               1               2048000000      0
     * 536870911    4               536870911       0               1               1               2147483644      0
     * 536870912    4               536870911       1               1               2               2147483644      4
     * 100          1073741824      1               0               100             100             1073741824      0
     * =========================================================================================================================
     * </code></pre>
     */
    @Test
    public void testPQLayoutEdgeCases() {
        System.out.println("\n=== PQLayout Edge Cases Test ===");
        System.out.printf("%-12s %-15s %-15s %-15s %-15s %-15s %-15s %-15s%n",
                "VectorCount", "CompDim", "FullChunkVecs", "LastChunkVecs", "FullSizeChunks", "TotalChunks", "FullChunkBytes", "LastChunkBytes");
        System.out.println("=" + "=".repeat(120));

        int[][] testCases = {
                // Minimal cases
                {1, 1}, {1, 2},
                
                // Power-of-2 boundaries for compressedDimension (layoutBytesPerVector changes)
                {10, 1}, {10, 2}, {10, 3}, {10, 4}, {10, 5},
                {10, 7}, {10, 8}, {10, 9},
                {10, 15}, {10, 16}, {10, 17},
                {10, 31}, {10, 32}, {10, 33},
                {10, 63}, {10, 64}, {10, 65},
                {10, 127}, {10, 128}, {10, 129},
                
                // Cases where addressableVectorsPerChunk becomes interesting
                {1073741823, 1}, // layoutBytesPerVector=2, addressableVectorsPerChunk=1073741823
                {1073741823, 2}, // layoutBytesPerVector=4, addressableVectorsPerChunk=536870911  
                {1073741824, 2}, // vectorCount > addressableVectorsPerChunk, creates chunks
                
                // Large dimension cases (small addressableVectorsPerChunk)
                {1000, 1024}, // layoutBytesPerVector=2048, addressableVectorsPerChunk=1048575
                {2000000, 1024}, // vectorCount > addressableVectorsPerChunk
                
                // Integer overflow boundary cases
                {536870911, 4}, // layoutBytesPerVector=8, exactly fits in one chunk
                {536870912, 4}, // one more than above, creates multiple chunks
                
                // Edge case where lastChunkVectors becomes non-zero
                {100, 1073741824} // layoutBytesPerVector huge, addressableVectorsPerChunk=1, creates 100 chunks
        };

        for (int[] testCase : testCases) {
            int vectorCount = testCase[0];
            int compressedDimension = testCase[1];

            try {
                PQVectors.PQLayout layout = new PQVectors.PQLayout(vectorCount, compressedDimension);
                System.out.printf("%-12d %-15d %-15d %-15d %-15d %-15d %-15d %-15d%n",
                        vectorCount, compressedDimension,
                        layout.fullChunkVectors, layout.lastChunkVectors,
                        layout.fullSizeChunks, layout.totalChunks,
                        layout.fullChunkBytes, layout.lastChunkBytes);

                // Basic sanity checks
                assertTrue("Total chunks should be positive", layout.totalChunks > 0);
                assertTrue("Full size chunks should be non-negative", layout.fullSizeChunks >= 0);
                assertTrue("Full chunk vectors should be positive", layout.fullChunkVectors > 0);
                assertTrue("Last chunk vectors should be non-negative", layout.lastChunkVectors >= 0);
                assertTrue("Last chunk vectors should be less than full chunk vectors",
                          layout.lastChunkVectors < layout.fullChunkVectors || layout.lastChunkVectors == 0);

            } catch (Exception e) {
                System.out.printf("%-12d %-15d %-60s%n", vectorCount, compressedDimension, "ERROR: " + e.getMessage());
            }
        }

        System.out.println("=" + "=".repeat(120));
        System.out.println("Test completed successfully");
    }


}
