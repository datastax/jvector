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

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import java.util.Arrays;
import java.util.List;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Exercise the heap-backed store intrinsic on JDK 22 without compiler exclusions. */
class NativeVectorStoreTest {
    private static NativeVectorUtilSupport backend;

    @BeforeAll
    static void loadNativeLibrary() {
        String arch = System.getProperty("os.arch", "");
        assumeTrue(arch.equals("amd64") || arch.equals("x86_64"));
        assumeTrue(LibraryLoader.loadJvector(), "Native jvector library not available");
        backend = new NativeVectorUtilSupport();
    }

    @Test
    void storesPreserveOffsetsAndCanariesForEveryFloatSpecies() {
        List<VectorSpecies<Float>> species = List.of(
                FloatVector.SPECIES_64, FloatVector.SPECIES_128, FloatVector.SPECIES_256, FloatVector.SPECIES_512);
        for (VectorSpecies<Float> shape : species) {
            float[] values = new float[shape.length()];
            for (int i = 0; i < values.length; i++)
                values[i] = i + .25f;
            var vector = FloatVector.fromArray(shape, values, 0);
            float[] destination = new float[shape.length() + 5];
            Arrays.fill(destination, -17f);
            var output = new MemorySegmentVectorFloat(destination);
            // Repeated stores exercise compiled Vector API code, where JDK 22 previously crashed.
            for (int i = 0; i < 20_000; i++)
                backend.intoVectorFloat(vector, output, 3);
            for (int i = 0; i < destination.length; i++) {
                float expected = i >= 3 && i < 3 + values.length ? values[i - 3] : -17f;
                assertEquals(expected, destination[i], 0f);
            }
        }
    }
}
