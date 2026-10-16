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

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TestGatherCompatibility {
    @Test
    void oldJdkCosineFallbackPreservesUnsignedIndicesOffsetsAndTails() throws Exception {
        var fallback = PanamaVectorUtilSupport.class.getDeclaredMethod("pqDecodedCosineSimilarityFallback",
                ByteSequence.class, int.class, int.class, int.class, VectorFloat.class, VectorFloat.class,
                float.class, VectorSpecies.class, VectorSpecies.class, VectorSpecies.class, int[].class);
        fallback.setAccessible(true);
        var support = new PanamaVectorUtilSupport();
        var random = new Random(755);
        for (int clusters : new int[] {16, 256}) {
            for (int count : new int[] {0, 1, 7, 8, 9, 15, 16, 17, 33, 96, 97}) {
                int offset = 3;
                byte[] codes = new byte[offset + count];
                for (int i = 0; i < codes.length; i++) codes[i] = (byte) random.nextInt(clusters);
                float[] partials = new float[count * clusters];
                float[] magnitudes = new float[partials.length];
                for (int i = 0; i < partials.length; i++) {
                    partials[i] = i % 9 - 4;
                    magnitudes[i] = 1 + i % 7;
                }
                float sum = 0, magnitude = 0;
                for (int i = 0; i < count; i++) {
                    int index = i * clusters + Byte.toUnsignedInt(codes[i + offset]);
                    sum += partials[index];
                    magnitude += magnitudes[index];
                }
                float expected = (float) (sum / Math.sqrt(magnitude * 3.25f));
                for (boolean wide : new boolean[] {false, true}) {
                    var floats = wide ? FloatVector.SPECIES_512 : FloatVector.SPECIES_256;
                    var integers = wide ? IntVector.SPECIES_512 : IntVector.SPECIES_256;
                    var bytes = wide ? ByteVector.SPECIES_128 : ByteVector.SPECIES_64;
                    float actual = (float) fallback.invoke(support, new ArrayByteSequence(codes), offset, count,
                            clusters, new ArrayVectorFloat(partials), new ArrayVectorFloat(magnitudes), 3.25f,
                            floats, integers, bytes, new int[floats.length()]);
                    // Integer-valued lane sums are exact, including the scalar tail.
                    assertEquals(expected, actual);
                }
            }
        }
    }
}
