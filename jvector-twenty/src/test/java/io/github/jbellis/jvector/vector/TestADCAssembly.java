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
import java.lang.reflect.Proxy;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestADCAssembly {
    private final PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();

    @Test
    void matchesLookupOracleAcrossWidthsOffsetsAndTails() {
        Random random = new Random(991);
        for (int count : new int[] {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 96, 99, 103, 128, 192, 256, 257}) {
            for (int stride : new int[] {2, 16, 256, 512}) {
                float[] table = new float[count * stride];
                for (int i = 0; i < table.length; i++) table[i] = random.nextFloat() * 2 - 1;
                for (int offset : new int[] {0, 1, 7}) {
                    byte[] codes = new byte[count + offset];
                    for (int i = 0; i < codes.length; i++) codes[i] = (byte) random.nextInt(Math.min(stride, 256));
                    check(table, stride, codes, offset, count);
                }
            }
        }
    }

    @Test
    void handlesEveryUnsignedCodeWithExactIntegerSums() {
        int stride = 256;
        int count = 33;
        float[] table = new float[count * stride];
        for (int i = 0; i < table.length; i++) table[i] = i % stride - 128;
        byte[] codes = new byte[count + 3];
        for (int first = 0; first < 256; first++) {
            for (int i = 0; i < count; i++) codes[i + 3] = (byte) (first + i);
            check(table, stride, codes, 3, count);
        }
    }

    @Test
    void shortScalarInputsDoNotRequireBackingArrays() {
        var data = (VectorFloat<?>) Proxy.newProxyInstance(VectorFloat.class.getClassLoader(),
                new Class<?>[]{VectorFloat.class}, (proxy, method, args) -> {
                    if (method.getName().equals("get") && method.getParameterCount() == 1)
                        return ((Integer) args[0]).floatValue() + 1f;
                    throw new AssertionError("Unexpected vector access: " + method);
                });
        var offsets = (ByteSequence<?>) Proxy.newProxyInstance(ByteSequence.class.getClassLoader(),
                new Class<?>[]{ByteSequence.class}, (proxy, method, args) -> {
                    if (method.getName().equals("get") && method.getParameterCount() == 1)
                        return (byte) 255;
                    throw new AssertionError("Unexpected code access: " + method);
                });
        for (int count = 0; count < 16; count++) {
            float expected = 0;
            for (int i = 0; i < count; i++) expected += 256 * i + 256;
            assertEquals(expected, support.assembleAndSum512(data, 256, offsets, 3, count));
            if (count < 8)
                assertEquals(expected, support.assembleAndSum256(data, 256, offsets, 3, count));
        }
    }

    private void check(float[] table, int stride, byte[] codes, int offset, int count) {
        double expected = 0;
        for (int i = 0; i < count; i++) expected += table[i * stride + Byte.toUnsignedInt(codes[offset + i])];
        var data = new ArrayVectorFloat(table);
        var offsets = new ArrayByteSequence(codes);
        double tolerance = Math.max(1e-5, count * 1e-6);
        assertEquals(expected, support.assembleAndSum128(data, stride, offsets, offset, count), tolerance);
        assertEquals(expected, support.assembleAndSum(data, stride, offsets, offset, count), tolerance);
        var compact = new ArrayByteSequence(java.util.Arrays.copyOfRange(codes, offset, offset + count));
        assertEquals(expected, support.assembleAndSum(data, stride, compact), tolerance);
        assertEquals(expected, support.assembleAndSum256(data, stride, offsets, offset, count), tolerance);
        assertEquals(expected, support.assembleAndSum512(data, stride, offsets, offset, count), tolerance);
        byte[] padded = new byte[codes.length + 11];
        System.arraycopy(codes, 0, padded, 5, codes.length);
        var slice = new ArrayByteSequence(padded).slice(5, codes.length);
        assertEquals(expected, support.assembleAndSum256(data, stride, slice, offset, count), tolerance);
        assertEquals(expected, support.assembleAndSum512(data, stride, slice, offset, count), tolerance);
    }
}
