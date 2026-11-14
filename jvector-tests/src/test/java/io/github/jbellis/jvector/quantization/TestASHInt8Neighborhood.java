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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** Offset-binary field-block neighborhoods scored with an INT8 query, in Java and natively. */
public class TestASHInt8Neighborhood {
    private static final int[] WIDTHS = {3, 5, 6, 7, 8, 9};

    private static AsymmetricHashing.QuantizedVector randomCode(Random random, int dimension, int bits) throws Exception {
        var constructor = AsymmetricHashing.QuantizedVector.class.getDeclaredConstructor(
                float.class, float.class, byte.class, long[].class, byte[].class);
        constructor.setAccessible(true);
        byte[] code = new byte[FusedASHLayout.canonicalCodeBytes(dimension, bits)];
        for (int j = 0; j < dimension; j++) {
            AsymmetricHashing.setOffsetBinaryField(code, j, bits, random.nextInt(1 << bits));
        }
        return constructor.newInstance(0.5f + random.nextFloat(), random.nextFloat() - 0.5f, (byte) 0,
                new long[0], code);
    }

    @Test
    public void javaInt8MatchesExactDotsAndNativeMatchesJava() throws Exception {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        Random random = new Random(20261111);
        for (int bits : WIDTHS) {
            for (int dimension : new int[]{1, 7, 8, 9, 63, 64, 65, 192, 383}) {
                for (int blockSize : new int[]{8, 16, 32}) {
                    int capacity = 2 * blockSize + 3;
                    int blocks = FusedASHLayout.blocksPerNode(capacity, blockSize);
                    byte[] packed = new byte[FusedASHLayout.featureSize(capacity, dimension, bits, blockSize)];
                    var vectors = new AsymmetricHashing.QuantizedVector[blocks * blockSize];
                    for (int i = 0; i < vectors.length; i++) {
                        vectors[i] = randomCode(random, dimension, bits);
                        int offset = FusedASHLayout.blockOffset(i / blockSize, dimension, bits, blockSize);
                        FusedASHLayout.packQuantizedVector(packed, offset, i % blockSize, vectors[i], dimension, bits, blockSize);
                    }
                    float[] q = new float[dimension];
                    double maxAbs = 0;
                    for (int j = 0; j < dimension; j++) {
                        q[j] = (float) random.nextGaussian();
                        maxAbs = Math.max(maxAbs, Math.abs(q[j]));
                    }
                    var query = new ASHInt8Query(dimension, bits);
                    query.prepare(q);
                    float landmarkDot = random.nextFloat() - 0.5f;
                    float[] java = new float[capacity + 2];
                    Arrays.fill(java, -7f);
                    double center = ((1 << bits) - 1) / 2.0;
                    for (int i = 0; i < capacity; i++) {
                        int offset = FusedASHLayout.blockOffset(i / blockSize, dimension, bits, blockSize);
                        int lane = i % blockSize;
                        byte[] code = new byte[FusedASHLayout.canonicalCodeBytes(dimension, bits)];
                        FusedASHLayout.readLaneCode(packed, offset, lane, dimension, bits, blockSize, code);
                        assertArrayEquals("lane code bits=" + bits + " d=" + dimension, vectors[i].extraBits, code);
                        double exact = 0, valueSum = 0;
                        for (int j = 0; j < dimension; j++) {
                            double value = AsymmetricHashing.offsetBinaryField(code, 0, j, bits) - center;
                            exact += q[j] * value;
                            valueSum += Math.abs(value);
                        }
                        float dot = query.laneDot(packed, offset, lane, blockSize);
                        // Each rounded component is within half a step, maxAbs / 254, of the query.
                        assertEquals("bits=" + bits + " d=" + dimension, exact, dot,
                                maxAbs / 254 * valueSum * 1.001 + 1e-5 * Math.max(1, Math.abs(exact)));
                        // The packed header holds FP16 scale and offset.
                        java[i] = ASHScorer.toSimilarity(
                                FusedASHLayout.readScale(packed, offset, lane, dimension, bits, blockSize) * dot
                                + landmarkDot + FusedASHLayout.readOffset(packed, offset, lane, dimension, bits, blockSize));
                    }
                    if (!backend.supportsAshInt8NeighborhoodScoring()) continue;
                    float[] nativeOut = new float[capacity + 2];
                    Arrays.fill(nativeOut, -7f);
                    backend.ashScoreNeighborhoodInt8(packed, bits, query.units, blockSize, capacity,
                            query.digits, query.invScale, query.center, landmarkDot, nativeOut);
                    for (int i = 0; i < capacity; i++) {
                        // Exact integer sums; the FP32 finish may round differently (no FMA below AVX2).
                        assertEquals("native bits=" + bits + " d=" + dimension + " bs=" + blockSize + " lane=" + i,
                                java[i], nativeOut[i], 1e-5f * Math.max(1f, Math.abs(java[i])));
                    }
                    assertEquals("Lanes past the capacity stay untouched", -7f, nativeOut[capacity], 0f);
                }
            }
        }
    }

    @Test
    public void nativeRejectsNonzeroLandmarksAndBadShapes() throws Exception {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        if (!backend.supportsAshInt8NeighborhoodScoring()) return;
        Random random = new Random(5);
        int dimension = 24, bits = 5, blockSize = 8;
        byte[] packed = new byte[FusedASHLayout.featureSize(blockSize, dimension, bits, blockSize)];
        for (int lane = 0; lane < blockSize; lane++) {
            var v = randomCode(random, dimension, bits);
            if (lane == 3) v.landmark = 1;
            FusedASHLayout.packQuantizedVector(packed, 0, lane, v, dimension, bits, blockSize);
        }
        var query = new ASHInt8Query(dimension, bits);
        query.prepare(new float[]{1, -2, 3, 0.5f, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1});
        float[] out = new float[blockSize];
        assertThrows(IllegalStateException.class, () -> backend.ashScoreNeighborhoodInt8(packed, bits, query.units,
                blockSize, blockSize, query.digits, query.invScale, query.center, 0f, out));
        assertThrows(IllegalArgumentException.class, () -> backend.ashScoreNeighborhoodInt8(packed, 4, query.units,
                blockSize, blockSize, query.digits, query.invScale, query.center, 0f, out));
        assertThrows(IndexOutOfBoundsException.class, () -> backend.ashScoreNeighborhoodInt8(packed, bits,
                query.units + 1, blockSize, blockSize, query.digits, query.invScale, query.center, 0f, out));
    }
}
