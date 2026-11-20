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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import static org.junit.Assert.*;

public class TestByteVectorSimilarityFunction extends RandomizedTest {

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Fills {@code raw} with random bytes, guaranteeing it is not the zero vector (cosine is undefined for it).
     * RandomizedTest's generator is biased toward runs of 0, so all-zero draws do happen.
     */
    private void nextNonZeroBytes(byte[] raw) {
        getRandom().nextBytes(raw);
        boolean allZero = true;
        for (byte b : raw) {
            if (b != 0) { allZero = false; break; }
        }
        if (allZero) raw[0] = 1;
    }

    private ByteSequence<?> seq(byte... values) {
        return vts.createByteSequence(values);
    }

    // -----------------------------------------------------------------------
    // EUCLIDEAN
    // -----------------------------------------------------------------------

    @Test
    public void testEuclideanKnownValue() {
        // v1=[1,0], v2=[0,1]  squared-L2 = 1+1 = 2
        // expected = 1 / (1 + 2) = 1/3
        var v1 = seq((byte) 1, (byte) 0);
        var v2 = seq((byte) 0, (byte) 1);
        assertEquals(1.0f / 3.0f, ByteVectorSimilarityFunction.EUCLIDEAN.compare(v1, v2), 1e-6f);
    }

    @Test
    public void testEuclideanDistinguishesSmallDistancesInHighDimensions() {
        // At 1024 dimensions, squared distances 0, 1, 2 and 3 must not collapse to the same float score.
        int dim = 1024;
        var base = seq(new byte[dim]);
        float previous = Float.POSITIVE_INFINITY;
        for (int sq = 0; sq <= 3; sq++) {
            byte[] other = new byte[dim];
            for (int i = 0; i < sq; i++) {
                other[i] = 1; // each unit component adds 1 to the squared distance
            }
            float score = ByteVectorSimilarityFunction.EUCLIDEAN.compare(base, seq(other));
            assertEquals(1.0f / (1.0f + sq), score, 1e-6f);
            assertTrue("score for squared distance " + sq + " should be strictly lower than for " + (sq - 1),
                       score < previous);
            previous = score;
        }
    }

    @Test
    public void testEuclideanIdenticalVectors() {
        var v = seq((byte) 42, (byte) -7, (byte) 100);
        // squaredL2 = 0, so score = 1/(1+0) = 1.0
        assertEquals(1.0f, ByteVectorSimilarityFunction.EUCLIDEAN.compare(v, v), 1e-6f);
    }

    @Test
    public void testEuclideanResultInRange() {
        for (int trial = 0; trial < 50; trial++) {
            byte[] raw1 = new byte[32];
            byte[] raw2 = new byte[32];
            getRandom().nextBytes(raw1);
            getRandom().nextBytes(raw2);
            float score = ByteVectorSimilarityFunction.EUCLIDEAN.compare(seq(raw1), seq(raw2));
            assertTrue("EUCLIDEAN score out of (0,1]: " + score, score > 0f && score <= 1.0f);
        }
    }

    @Test
    public void testEuclideanSymmetry() {
        byte[] raw1 = new byte[16];
        byte[] raw2 = new byte[16];
        getRandom().nextBytes(raw1);
        getRandom().nextBytes(raw2);
        assertEquals(
                ByteVectorSimilarityFunction.EUCLIDEAN.compare(seq(raw1), seq(raw2)),
                ByteVectorSimilarityFunction.EUCLIDEAN.compare(seq(raw2), seq(raw1)),
                1e-6f);
    }

    // -----------------------------------------------------------------------
    // DOT_PRODUCT
    // -----------------------------------------------------------------------

    @Test
    public void testDotProductKnownValue() {
        // v1=[1,0], v2=[0,1]  dot = 0
        // expected = dot + 1 = 1
        var v1 = seq((byte) 1, (byte) 0);
        var v2 = seq((byte) 0, (byte) 1);
        assertEquals(1.0f, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2), 1e-6f);
        // negative dot = -2: 1 / (1 + 2)
        assertEquals(1f / 3f, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq((byte) 1, (byte) 1), seq((byte) -1, (byte) -1)), 1e-6f);
    }

    @Test
    public void testDotProductResultInRange() {
        for (int trial = 0; trial < 50; trial++) {
            byte[] raw1 = new byte[32];
            byte[] raw2 = new byte[32];
            getRandom().nextBytes(raw1);
            getRandom().nextBytes(raw2);
            float score = ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq(raw1), seq(raw2));
            assertTrue("DOT_PRODUCT score not positive: " + score, score > 0f);
        }
    }

    @Test
    public void testDotProductKeepsDistinctScoresAtHighDimension() {
        // At 1,024 dimensions, adjacent integer dot products must not collapse to the same float score.
        int dim = 1024;
        byte[] q = new byte[dim];
        java.util.Arrays.fill(q, (byte) 1);
        float prev = Float.NEGATIVE_INFINITY;
        for (int dot = -3; dot <= 3; dot++) {
            byte[] v = new byte[dim];
            v[0] = (byte) dot;
            float score = ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq(q), seq(v));
            assertTrue("scores must strictly increase with dot product: " + prev + " -> " + score, score > prev);
            prev = score;
        }
    }

    @Test
    public void testDotProductSymmetry() {
        byte[] raw1 = new byte[16];
        byte[] raw2 = new byte[16];
        getRandom().nextBytes(raw1);
        getRandom().nextBytes(raw2);
        assertEquals(
                ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq(raw1), seq(raw2)),
                ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq(raw2), seq(raw1)),
                1e-6f);
    }

    // -----------------------------------------------------------------------
    // COSINE
    // -----------------------------------------------------------------------

    @Test
    public void testCosineParallelVectors() {
        // v1 == v2 → cosine = 1.0, score = (1+1)/2 = 1.0
        var v = seq((byte) 3, (byte) 4);
        assertEquals(1.0f, ByteVectorSimilarityFunction.COSINE.compare(v, v), 1e-5f);
    }

    @Test
    public void testCosineOrthogonalVectors() {
        // [1,0] · [0,1] = 0, cosine = 0, score = (1+0)/2 = 0.5
        var v1 = seq((byte) 1, (byte) 0);
        var v2 = seq((byte) 0, (byte) 1);
        assertEquals(0.5f, ByteVectorSimilarityFunction.COSINE.compare(v1, v2), 1e-5f);
    }

    @Test
    public void testCosineResultInRange() {
        for (int trial = 0; trial < 50; trial++) {
            byte[] raw1 = new byte[32];
            byte[] raw2 = new byte[32];
            nextNonZeroBytes(raw1);
            nextNonZeroBytes(raw2);
            float score = ByteVectorSimilarityFunction.COSINE.compare(seq(raw1), seq(raw2));
            assertTrue("COSINE score out of [0,1]: " + score, score >= 0f && score <= 1.0f);
        }
    }

    @Test
    public void testCosineSymmetry() {
        byte[] raw1 = new byte[16];
        byte[] raw2 = new byte[16];
        nextNonZeroBytes(raw1);
        nextNonZeroBytes(raw2);
        assertEquals(
                ByteVectorSimilarityFunction.COSINE.compare(seq(raw1), seq(raw2)),
                ByteVectorSimilarityFunction.COSINE.compare(seq(raw2), seq(raw1)),
                1e-5f);
    }

    @Test
    public void testCosineDimensionMismatchThrows() {
        var v1 = seq((byte) 1, (byte) 2, (byte) 3);
        var v2 = seq((byte) 1, (byte) 2);
        assertThrows(IllegalArgumentException.class, () -> VectorUtil.cosine(v1, v2));
    }

    @Test
    public void testCosineZeroVectorIsRejectedByAssertion() {
        // Mirrors the float overload: zero vectors are unsupported and caught by an assertion, not defined.
        boolean assertionsEnabled = false;
        assert assertionsEnabled = true;
        org.junit.Assume.assumeTrue("requires -ea", assertionsEnabled);

        var zero = seq(new byte[8]);
        var nonZero = seq((byte) 1, (byte) 2, (byte) 3, (byte) 4, (byte) 5, (byte) 6, (byte) 7, (byte) 8);
        assertThrows(AssertionError.class, () -> VectorUtil.cosine(zero, zero));
        assertThrows(AssertionError.class, () -> VectorUtil.cosine(zero, nonZero));
        assertThrows(AssertionError.class, () -> VectorUtil.cosine(nonZero, zero));
    }

    // -----------------------------------------------------------------------
    // Boundary values
    // -----------------------------------------------------------------------

    @Test
    public void testAllMaxValues() {
        // All -128 vectors — EUCLIDEAN identity = 1, DOT_PRODUCT = n*128^2 + 1, COSINE = 1
        byte[] rawMin = new byte[8];
        java.util.Arrays.fill(rawMin, (byte) -128);
        var vMin = seq(rawMin);
        assertEquals(1.0f, ByteVectorSimilarityFunction.EUCLIDEAN.compare(vMin, vMin), 1e-5f);
        assertEquals(8 * 128f * 128f + 1f, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(vMin, vMin), 1e-3f);
        assertEquals(1.0f, ByteVectorSimilarityFunction.COSINE.compare(vMin, vMin), 1e-5f);

        // All 127 vectors — EUCLIDEAN identity = 1, COSINE = 1
        byte[] rawMax = new byte[8];
        java.util.Arrays.fill(rawMax, (byte) 127);
        var vMax = seq(rawMax);
        assertEquals(1.0f, ByteVectorSimilarityFunction.EUCLIDEAN.compare(vMax, vMax), 1e-5f);
        float expectedDot = 8 * 127f * 127f + 1f;
        assertEquals(expectedDot, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(vMax, vMax), 1e-3f);
        assertEquals(1.0f, ByteVectorSimilarityFunction.COSINE.compare(vMax, vMax), 1e-5f);
    }

    // -----------------------------------------------------------------------
    // Default implementation accumulation
    // -----------------------------------------------------------------------

    @Test
    public void testDefaultImplementationAccumulatesExactlyInHighDimensions() {
        // Float accumulation stops being exact above 2^24; these sums exceed it, so a float accumulator
        // would round differently from the integer-accumulating Panama and native implementations.
        var impl = new DefaultVectorUtilSupport();
        int dim = 4096;
        byte[][][] cases = new byte[3][2][dim];
        java.util.Arrays.fill(cases[0][0], (byte) 127);
        java.util.Arrays.fill(cases[0][1], (byte) 127);
        java.util.Arrays.fill(cases[1][0], (byte) 127);
        java.util.Arrays.fill(cases[1][1], (byte) -128);
        getRandom().nextBytes(cases[2][0]);
        getRandom().nextBytes(cases[2][1]);

        for (byte[][] c : cases) {
            long dot = 0, normA = 0, normB = 0, sq = 0;
            for (int i = 0; i < dim; i++) {
                int ai = c[0][i], bi = c[1][i];
                dot += ai * bi;
                normA += ai * ai;
                normB += bi * bi;
                sq += (ai - bi) * (ai - bi);
            }
            var a = seq(c[0]);
            var b = seq(c[1]);
            assertEquals((float) dot, impl.dotProduct(a, b), 0f);
            assertEquals((float) sq, impl.squareDistance(a, b), 0f);
            assertEquals((float) (dot / Math.sqrt((double) normA * normB)), impl.cosine(a, b), 0f);
        }
    }

    // -----------------------------------------------------------------------
    // Sliced inputs
    // -----------------------------------------------------------------------

    @Test
    public void testSlicedInputsMatchScalarReference() {
        // Lengths cover: shorter than any SIMD step, exact multiples, and lengths with a scalar tail.
        for (int length : new int[] {5, 8, 16, 37, 64, 100}) {
            for (int[] offsets : new int[][] {{0, 0}, {3, 0}, {0, 5}, {7, 11}}) {
                byte[] raw1 = new byte[offsets[0] + length + 9];
                byte[] raw2 = new byte[offsets[1] + length + 9];
                getRandom().nextBytes(raw1);
                getRandom().nextBytes(raw2);
                var a = seq(raw1).slice(offsets[0], length);
                var b = seq(raw2).slice(offsets[1], length);

                long dot = 0, normA = 0, normB = 0, sq = 0;
                for (int i = 0; i < length; i++) {
                    int ai = raw1[offsets[0] + i], bi = raw2[offsets[1] + i];
                    dot += ai * bi;
                    normA += ai * ai;
                    normB += bi * bi;
                    sq += (ai - bi) * (ai - bi);
                }
                String msg = "length=" + length + " offsets=" + offsets[0] + "," + offsets[1];
                assertEquals(msg, (float) dot, VectorUtil.dotProduct(a, b), 0f);
                assertEquals(msg, (float) sq, VectorUtil.squareL2Distance(a, b), 0f);
                // RandomizedTest's nextBytes is biased toward 0/-1 runs, so a short slice can be all zeros;
                // cosine is undefined for zero vectors (VectorUtil.cosine asserts on them), so skip it then.
                if (normA != 0 && normB != 0) {
                    assertEquals(msg, (float) (dot / Math.sqrt((double) normA * normB)), VectorUtil.cosine(a, b), 1e-5f);
                }
            }
        }
    }
}
