/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.vector.ASHLutScoring;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import java.util.Arrays;
import java.util.Random;
import static org.junit.Assert.*;

/** Independent decoded-component oracle; covers padding, offsets and SIMD lane boundaries. */
public class TestASHLutScoring {
    @Test
    public void allSerializedHeadersAreFortyBits() throws Exception {
        assertEquals(40, AsymmetricHashing.HEADER_BITS);
        assertEquals(5, AsymmetricHashing.HEADER_BYTES);
        for (int bits = 1; bits <= 9; bits++) {
            for (int dimensions : new int[]{1, 33, 64, 769}) {
                var vector = AsymmetricHashing.QuantizedVector.createEmpty(dimensions, bits);
                vector.scale = 1.0003f;
                vector.offset = -0.33333f;
                vector.landmark = (byte) 255;
                int bodyBytes = vector.binaryVector.length * Long.BYTES + vector.extraBits.length;
                try (var writer = io.github.jbellis.jvector.disk.ByteBufferIndexWriter.create(bodyBytes + 5, false)) {
                    vector.write(writer, dimensions, bits);
                    assertEquals(bodyBytes + 5, writer.position());
                    assertEquals(bodyBytes + 5, AsymmetricHashing.QuantizedVector.serializedSizeBytes(dimensions, bits));
                    var decoded = AsymmetricHashing.QuantizedVector.load(
                            new io.github.jbellis.jvector.disk.ByteBufferReader(writer.getWrittenData()), dimensions, bits);
                    assertEquals(255, decoded.landmark & 255);
                    assertEquals(vector.scale, decoded.scale, 0.001f);
                    assertEquals(vector.offset, decoded.offset, 0.001f);
                }
                if (FusedASHLayout.supportsBitsPerDimension(bits)) {
                    for (int blockSize : new int[]{8, 16, 32}) {
                        assertEquals(5 * blockSize, FusedASHLayout.blockHeaderBytes(blockSize));
                        byte[] block = new byte[FusedASHLayout.blockBytes(dimensions, bits, blockSize)];
                        FusedASHLayout.packQuantizedVector(block, 0, blockSize - 1, vector, dimensions, bits, blockSize);
                        assertEquals(255, FusedASHLayout.readLandmark(block, 0, blockSize - 1, dimensions, bits, blockSize));
                    }
                }
            }
        }
    }

    @Test
    public void packedBlocksMatchDecodedComponents() {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        if (Boolean.getBoolean("jvector.test.requireAshSimd")) {
            assertTrue("Test must execute a SIMD backend: " + backend.getClass(), backend.supportsAshLutScoring());
            assertTrue(backend.supportsAshProjectionScoring());
        }
        Random random = new Random(87342);
        for (int bits : new int[]{1, 2, 4}) {
            for (int d : new int[]{1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 384, 768, 769}) {
                int groups = FusedASHLayout.codeGroups(d, bits);
                float[] q = new float[d];
                for (int i = 0; i < d; i++) q[i] = random.nextFloat() * 2 - 1;
                float[] lut = new float[groups * 16];
                FusedASHLayout.buildQueryLut(q, d, bits, lut);
                for (int size : new int[]{8, 16, 32}) {
                    int offset = 7;
                    byte[] packed = new byte[offset + FusedASHLayout.blockBodyBytes(d, bits, size)];
                    random.nextBytes(packed); // Padding is deliberately nonzero.
                    float[] expected = new float[size];
                    for (int lane = 0; lane < size; lane++) {
                        byte[] canonical = new byte[FusedASHLayout.canonicalCodeBytes(d, bits)];
                        for (int g = 0; g < groups; g++) {
                            int nibble = (lane + g) & 15; // Exercise every code, including both signs.
                            FusedASHLayout.setPackedNibble(packed, offset, lane, g, size, nibble);
                            FusedASHLayout.setFlatNibble(canonical, g, nibble);
                        }
                        for (int i = 0; i < d; i++) {
                            int field = (canonical[i * bits / 8] & 255) >>> (i * bits % 8);
                            int sign = 1 << (bits - 1);
                            float magnitude = bits == 1 ? 1 : (field & (sign - 1)) + 0.5f;
                            expected[lane] += q[i] * ((field & sign) == 0 ? -magnitude : magnitude);
                        }
                        if (bits != 1) {
                            assertEquals("single bits=" + bits + " d=" + d, expected[lane],
                                    backend.usesAshProjectionTuning()
                                            ? backend.ashProjectionDotTuned(q, canonical, d, bits)
                                            : backend.ashProjectionDot(q, canonical, d, bits), tolerance(expected[lane]));
                        }
                    }
                    for (int start = 0; start <= size; start++) {
                        for (int count = 0; count <= size - start; count++) {
                            float[] actual = new float[size + 4];
                            float[] scalar = new float[size + 4];
                            Arrays.fill(actual, Float.NaN);
                            backend.ashLutScore(packed, offset, groups, size, start, count, lut, actual, 2);
                            ASHLutScoring.score(packed, offset, groups, size, start, count, lut, scalar, 2);
                            assertTrue(Float.isNaN(actual[1]));
                            assertTrue(Float.isNaN(actual[count + 2]));
                            for (int i = 0; i < count; i++) {
                                String label = "bits=" + bits + " d=" + d + " size=" + size + " lane=" + (start + i);
                                assertEquals(label, expected[start + i], actual[2 + i], tolerance(expected[start + i]));
                                assertEquals(label, expected[start + i], scalar[2 + i], tolerance(expected[start + i]));
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void rejectsInvalidBounds() {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashLutScore(new byte[8], 0, 2, 8, 7, 2, new float[32], new float[2], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashLutScore(new byte[7], 0, 2, 8, 0, 1, new float[32], new float[1], 0));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashLutScore(new byte[8], 0, 2, 8, 0, 1, new float[31], new float[1], 0));
    }

    private static float tolerance(float value) { return 0.0003f + Math.abs(value) * 0.00002f; }
}
