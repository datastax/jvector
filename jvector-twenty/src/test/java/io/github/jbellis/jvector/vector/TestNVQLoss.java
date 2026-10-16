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
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestNVQLoss {
    @Test
    void vectorLossMatchesIndependentScalarCoordinates() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(991);
        for (int dimensions : new int[] {7, 8, 9, 15, 16, 17, 33, 127, 129, 768, 1024, 1536}) {
            float[] values = new float[dimensions];
            for (int i = 0; i < dimensions; i++) values[i] = random.nextFloat() * 2 - 1;
            var vector = new ArrayVectorFloat(values);
            var coordinate = new ArrayVectorFloat(new float[1]);
            for (float alpha : new float[] {0.1f, 0.5f, 1f, 2f, 5f}) {
                for (float midpoint : new float[] {-0.25f, 0f, 0.25f, 0.5f}) {
                    for (int bits : new int[] {1, 2, 4, 8}) {
                        double expected = 0;
                        // A single-coordinate input takes the scalar tail path. It uses
                        // Math.round rather than the vector conversion sequence under test.
                        for (float value : values) {
                            coordinate.set(0, value);
                            expected += support.nvqLoss(coordinate, alpha, midpoint, -1f, 1f, bits);
                        }
                        float actual = support.nvqLoss(vector, alpha, midpoint, -1f, 1f, bits);
                        assertEquals(expected, actual, Math.max(1e-6, Math.abs(expected) * 2e-6),
                                "dimensions=" + dimensions + ", alpha=" + alpha
                                        + ", midpoint=" + midpoint + ", bits=" + bits);
                    }
                }
            }
        }
    }
    @Test
    void encodingAndHelpersMatchScalarCoordinates() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(717);
        var species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
        for (int dimensions : new int[] {7, 16, 17, 33, 129, 768, 1536}) {
            float[] values = new float[dimensions];
            for (int i = 0; i < dimensions; i++) values[i] = random.nextFloat() * 2 - 1;
            var encoded = new ArrayByteSequence(new byte[dimensions]);
            var single = new ArrayByteSequence(new byte[1]);
            for (float alpha : new float[] {0.1f, 1f, 5f}) {
                for (float midpoint : new float[] {-0.25f, 0f, 0.5f}) {
                    support.nvqQuantize8bit(new ArrayVectorFloat(values), alpha, midpoint, -1, 1, encoded);
                    for (int j = 0; j < dimensions; j++) {
                        support.nvqQuantize8bit(new ArrayVectorFloat(new float[] {values[j]}),
                                alpha, midpoint, -1, 1, single);
                        assertEquals(single.get(0), encoded.get(j));
                    }
                    for (int j = 0; j < species.loopBound(dimensions); j += species.length()) {
                        var input = jdk.incubator.vector.FloatVector.fromArray(species, values, j);
                        var logistic = support.logisticNQT(input, alpha, midpoint);
                        var inverse = support.logitNQT(logistic, 1 / alpha, midpoint);
                        for (int lane = 0; lane < species.length(); lane++) {
                            assertEquals(support.logisticNQT(values[j + lane], alpha, midpoint), logistic.lane(lane));
                            assertEquals(support.logitNQT(logistic.lane(lane), 1 / alpha, midpoint), inverse.lane(lane));
                        }
                    }
                }
            }
        }
    }
}
