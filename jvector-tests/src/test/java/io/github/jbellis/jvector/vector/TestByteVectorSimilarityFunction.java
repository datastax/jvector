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

    private ByteSequence<?> seq(byte... values) {
        return vts.createByteSequence(values);
    }

    // -----------------------------------------------------------------------
    // EUCLIDEAN
    // -----------------------------------------------------------------------

    @Test
    public void testEuclideanKnownValue() {
        // v1=[1,0], v2=[0,1]  squared-L2 = 1+1 = 2
        // maxSquaredDist = 2 * 255^2 = 130050
        // expected = 1 / (1 + 2/130050)
        var v1 = seq((byte) 1, (byte) 0);
        var v2 = seq((byte) 0, (byte) 1);
        float squaredL2 = 2.0f;
        float max = 2 * 255.0f * 255.0f;
        float expected = 1.0f / (1.0f + squaredL2 / max);
        assertEquals(expected, ByteVectorSimilarityFunction.EUCLIDEAN.compare(v1, v2), 1e-6f);
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
        // maxMag = 2 * 127^2 = 32258
        // expected = (1 + 0/32258) / 2 = 0.5
        var v1 = seq((byte) 1, (byte) 0);
        var v2 = seq((byte) 0, (byte) 1);
        assertEquals(0.5f, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2), 1e-6f);
    }

    @Test
    public void testDotProductResultInRange() {
        for (int trial = 0; trial < 50; trial++) {
            byte[] raw1 = new byte[32];
            byte[] raw2 = new byte[32];
            getRandom().nextBytes(raw1);
            getRandom().nextBytes(raw2);
            float score = ByteVectorSimilarityFunction.DOT_PRODUCT.compare(seq(raw1), seq(raw2));
            assertTrue("DOT_PRODUCT score out of [0,1]: " + score, score >= 0f && score <= 1.0f);
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
            getRandom().nextBytes(raw1);
            getRandom().nextBytes(raw2);
            float score = ByteVectorSimilarityFunction.COSINE.compare(seq(raw1), seq(raw2));
            assertTrue("COSINE score out of [0,1]: " + score, score >= 0f && score <= 1.0f);
        }
    }

    @Test
    public void testCosineSymmetry() {
        byte[] raw1 = new byte[16];
        byte[] raw2 = new byte[16];
        getRandom().nextBytes(raw1);
        getRandom().nextBytes(raw2);
        assertEquals(
                ByteVectorSimilarityFunction.COSINE.compare(seq(raw1), seq(raw2)),
                ByteVectorSimilarityFunction.COSINE.compare(seq(raw2), seq(raw1)),
                1e-5f);
    }

    // -----------------------------------------------------------------------
    // Boundary values
    // -----------------------------------------------------------------------

    @Test
    public void testAllMaxValues() {
        // All 127 vectors — EUCLIDEAN identity = 1, DOT_PRODUCT = 1, COSINE = 1
        byte[] raw = new byte[8];
        java.util.Arrays.fill(raw, (byte) 127);
        var v = seq(raw);
        assertEquals(1.0f, ByteVectorSimilarityFunction.EUCLIDEAN.compare(v, v), 1e-5f);
        assertEquals(1.0f, ByteVectorSimilarityFunction.DOT_PRODUCT.compare(v, v), 1e-5f);
        assertEquals(1.0f, ByteVectorSimilarityFunction.COSINE.compare(v, v), 1e-5f);
    }
}
