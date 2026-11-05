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

import java.util.Random;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TestNVQScoring {
    private final PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();

    @Test
    void queryShuffleMatchesFastLanesPermutationAndLeavesTailUnchanged() {
        int lanes = FloatVector.SPECIES_PREFERRED.length(), block = lanes * 4;
        for (int length : new int[]{0, 1, block - 1, block, block + 1, block * 2 + 7, 768, 1536, 3072}) {
            float[] input = new float[length], expected = new float[length];
            for (int i = 0; i < length; i++) input[i] = expected[i] = i;
            for (int start = 0; start + block <= length; start += block)
                for (int part = 0; part < 4; part++)
                    for (int lane = 0; lane < lanes; lane++)
                        expected[start + part * lanes + lane] = input[start + 4 * lane + part];
            var vector = new ArrayVectorFloat(input.clone());
            support.nvqShuffleQueryInPlace8bit(vector);
            assertArrayEquals(expected, vector.get(), "length=" + length);
        }
        // The transpose helper must also respect a nonzero start and its surrounding data.
        float[] input = new float[31], expected = new float[31];
        for (int i = 0; i < input.length; i++) input[i] = expected[i] = i;
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 6; col++) expected[3 + row * 6 + col] = input[3 + col * 4 + row];
        var vector = new ArrayVectorFloat(input.clone());
        support.transpose(vector, 3, 27, 4);
        assertArrayEquals(expected, vector.get());
    }

    @Test
    void allScorersMatchIndependentScalarDequantization() {
        var random = new Random(681);
        int block = FloatVector.SPECIES_PREFERRED.length() * 4;
        for (int length : new int[]{0, 1, 7, 8, 17, block - 1, block, block + 1, 129, 768, 1536, 3072}) {
            float[] query = new float[length], centroid = new float[length];
            byte[] codes = new byte[length];
            for (int i = 0; i < length; i++) {
                query[i] = random.nextFloat() * 2 - 1;
                centroid[i] = random.nextFloat() - 0.5f;
                codes[i] = (byte) i; // Includes every unsigned byte on large inputs.
            }
            for (float alpha : new float[]{0.1f, 1f, 5f}) {
                for (float midpoint : new float[]{-0.25f, 0f, 0.5f}) {
                    var shuffledQuery = new ArrayVectorFloat(query.clone());
                    var shuffledCentroid = new ArrayVectorFloat(centroid.clone());
                    support.nvqShuffleQueryInPlace8bit(shuffledQuery);
                    support.nvqShuffleQueryInPlace8bit(shuffledCentroid);
                    double dot = 0, absoluteDot = 0, l2 = 0, cosineDot = 0, absoluteCosineDot = 0, mag = 0;
                    for (int i = 0; i < length; i++) {
                        float decoded = decode(codes[i] & 255, alpha, midpoint, -1f, 1f);
                        float withCentroid = decoded + centroid[i];
                        dot += (double) query[i] * decoded;
                        absoluteDot += Math.abs((double) query[i] * decoded);
                        float diff = query[i] - decoded;
                        l2 += (double) diff * diff;
                        cosineDot += (double) query[i] * withCentroid;
                        absoluteCosineDot += Math.abs((double) query[i] * withCentroid);
                        mag += (double) withCentroid * withCentroid;
                    }
                    var encoded = new ArrayByteSequence(codes);
                    assertEquals(dot, support.nvqDotProduct8bit(shuffledQuery, encoded, alpha, midpoint, -1, 1),
                            Math.max(1e-6, absoluteDot * 2e-6));
                    assertEquals(l2, support.nvqSquareL2Distance8bit(shuffledQuery, encoded, alpha, midpoint, -1, 1),
                            Math.max(1e-6, l2 * 2e-6));
                    float[] cosine = support.nvqCosine8bit(shuffledQuery, encoded, alpha, midpoint, -1, 1, shuffledCentroid);
                    assertEquals(2, cosine.length);
                    assertEquals(cosineDot, cosine[0], Math.max(1e-6, absoluteCosineDot * 2e-6));
                    assertEquals(mag, cosine[1], Math.max(1e-6, mag * 2e-6));
                    assertArrayEquals(codes, encoded.get());
                }
            }
        }
        assertThrows(IllegalArgumentException.class, () -> support.nvqCosine8bit(new ArrayVectorFloat(1),
                new ArrayByteSequence(1), 1, 0, -1, 1, new ArrayVectorFloat(2)));
    }

    @Test
    void uniformLossMatchesScalarQuantizationAcrossBitDepths() {
        var random = new Random(221);
        for (int length : new int[]{0, 1, 7, 8, 17, 33, 129, 768, 1536, 3072}) {
            for (float[] range : new float[][]{{-1, 1}, {-2, 3}}) {
                float min = range[0], max = range[1], delta = max - min;
                float[] values = new float[length];
                for (int i = 0; i < length; i++) values[i] = min + random.nextFloat() * delta;
                for (int bits : new int[]{1, 2, 4, 8}) {
                    double expected = 0;
                    int levels = (1 << bits) - 1;
                    for (float value : values) {
                        int code = Math.round((value - min) / delta * levels);
                        double decoded = min + (double) code * delta / levels;
                        expected += (value - decoded) * (value - decoded);
                    }
                    assertEquals(expected, support.nvqUniformLoss(new ArrayVectorFloat(values), min, max, bits),
                            Math.max(1e-6, expected * 2e-6));
                }
            }
        }
    }

    // Scalar implementation of NVQ's documented approximate logistic/logit transform.
    // It does not call the kernel helpers, so scoring and lane ordering have separate oracles.
    private static float logistic(float value, float alpha, float midpoint) {
        float x = Math.fma(value, alpha, -alpha * midpoint);
        int exponent = (int) Math.floor(x + 1);
        int bits = Float.floatToIntBits(Math.fma(x - exponent, 0.5f, 1));
        float exponential = Float.intBitsToFloat(bits + (exponent << 23));
        return exponential / (exponential + 1);
    }

    private static float decode(int code, float alpha, float midpoint, float min, float max) {
        float scaledAlpha = alpha / (max - min), scaledMidpoint = midpoint * (max - min);
        float bias = logistic(min, scaledAlpha, scaledMidpoint);
        float scale = (logistic(max, scaledAlpha, scaledMidpoint) - bias) / 255;
        float x = Math.fma(scale, code, bias);
        int bits = Float.floatToIntBits(x / (1 - x));
        float exponent = ((bits & 0x7f800000) >>> 23) - 128;
        float mantissa = Float.intBitsToFloat((bits & 0x007fffff) + 0x3f800000);
        return Math.fma(mantissa + exponent, 1 / scaledAlpha, scaledMidpoint);
    }
}
