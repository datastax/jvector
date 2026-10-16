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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TestNVQDecoding {
    @Test
    void logisticHelperRetainsEveryInputSpecies() {
        var support = new PanamaVectorUtilSupport();
        for (var species : List.of(FloatVector.SPECIES_64, FloatVector.SPECIES_128,
                FloatVector.SPECIES_256, FloatVector.SPECIES_512, FloatVector.SPECIES_MAX)) {
            float[] values = new float[species.length()];
            for (int i = 0; i < values.length; i++) values[i] = (i - 7) / 8f;
            var vector = FloatVector.fromArray(species, values, 0);
            var actual = support.logisticNQT(vector, 1.3f, 0.25f);
            assertEquals(species, actual.species());
            // Frozen original helper: no reductions, so every lane must retain its exact bits.
            var temp = vector.fma(1.3f, -1.3f * 0.25f);
            var positive = temp.test(VectorOperators.IS_NEGATIVE).not();
            var exponent = temp.add(1, positive).convert(VectorOperators.F2I, 0).reinterpretAsInts();
            var integral = exponent.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
            var mantissa = temp.sub(integral).fma(0.5f, 1).reinterpretAsInts();
            temp = mantissa.add(exponent.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();
            var expected = temp.div(temp.add(1)).toArray();
            var lanes = actual.toArray();
            for (int i = 0; i < lanes.length; i++)
                assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(lanes[i]));
        }
    }

    @Test
    void encodedByteShapeUsesTheSmallestFixedWidthThatFits() {
        int[] lanes = {2, 4, 8, 12, 16, 32, 48, 64};
        int[] bits = {64, 64, 64, 128, 128, 256, 512, 512};
        for (int i = 0; i < lanes.length; i++) {
            var species = PanamaVectorUtilSupport.nvqByteSpecies(lanes[i]);
            assertEquals(bits[i], species.vectorBitSize());
            assertTrue(species.length() >= lanes[i]);
        }
        for (var species : List.of(FloatVector.SPECIES_64, FloatVector.SPECIES_128,
                FloatVector.SPECIES_256, FloatVector.SPECIES_512, FloatVector.SPECIES_MAX)) {
            assertTrue(PanamaVectorUtilSupport.nvqByteSpecies(species.length()).length() >= species.length());
        }
        // The MAX fallback is needed only on platforms with byte shapes wider than 512 bits.
        if (ByteVector.SPECIES_MAX.length() > 64) {
            assertEquals(ByteVector.SPECIES_MAX, PanamaVectorUtilSupport.nvqByteSpecies(65));
        }
    }

    @Test
    void concreteByteConversionsRetainEveryOutputSpecies() {
        int[] values = new int[IntVector.SPECIES_PREFERRED.length()];
        for (var species : List.of(ByteVector.SPECIES_64, ByteVector.SPECIES_128,
                ByteVector.SPECIES_256, ByteVector.SPECIES_512, ByteVector.SPECIES_MAX)) {
            int count = Math.min(values.length, species.length());
            byte[] output = new byte[count + 2];
            for (int code = 0; code < 256; code++) {
                for (int lane = 0; lane < values.length; lane++) values[lane] = (code + lane) & 0xff;
                var integers = IntVector.fromArray(IntVector.SPECIES_PREFERRED, values, 0);
                var bytes = (ByteVector) species.vectorType().cast(
                        integers.convertShape(VectorOperators.I2B, species, 0));
                assertEquals(species, bytes.species());
                Arrays.fill(output, (byte) 0x5a);
                bytes.intoArray(output, 1, species.indexInRange(0, count));
                for (int lane = 0; lane < count; lane++) assertEquals((byte) values[lane], output[lane + 1]);
                assertEquals((byte) 0x5a, output[0]);
                assertEquals((byte) 0x5a, output[count + 1]);
            }
        }
    }

    @Test
    void fastLanesDecodeMatchesScalarBitsForEveryByteAndPart() {
        var support = new PanamaVectorUtilSupport();
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        byte[] bytes = new byte[ByteVector.SPECIES_PREFERRED.length()];
        for (int code = 0; code < 256; code++) {
            // Across code iterations, every coordinate exercises every unsigned byte.
            for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) (code + i);
            var packed = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, bytes, 0);
            for (float inverseAlpha : new float[] {0.2f, 0.5f, 1f, 2f, 10f}) {
                for (float midpoint : new float[] {-0.25f, 0f, 0.25f}) {
                    for (int part = 0; part < 4; part++) {
                        float[] actual = support.nvqDequantize8bit(
                                packed, inverseAlpha, midpoint, 0.003f, 0.05f, part).toArray();
                        for (int lane = 0; lane < lanes; lane++) {
                            float value = Math.fma(Byte.toUnsignedInt(bytes[4 * lane + part]), 0.003f, 0.05f);
                            float z = value / (1f - value);
                            int bits = Float.floatToIntBits(z);
                            float exponent = ((bits & 0x7f800000) >>> 23) - 128;
                            float mantissa = Float.intBitsToFloat((bits & 0x007fffff) + 0x3f800000);
                            float expected = Math.fma(mantissa + exponent, inverseAlpha, midpoint);
                            assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual[lane]),
                                    "code=" + code + " part=" + part + " lane=" + lane);
                        }
                    }
                }
            }
        }
    }
}
