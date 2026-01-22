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

package io.github.jbellis.jvector.disk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;

import io.github.jbellis.jvector.vector.types.VectorFloat;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class FvecSegmentRavvTest extends RandomizedTest {

    @TempDir
    Path tempDir;

    /**
     * Helper method to write fvec format file
     * Format: [dim:int][vector1_values:float*dim][dim:int][vector2_values:float*dim]...
     */
    private void writeFvecFile(Path path, float[][] vectors) throws IOException {
        try (var out = new DataOutputStream(new FileOutputStream(path.toFile()))) {
            for (float[] vector : vectors) {
                // Write dimension in little-endian
                ByteBuffer dimBuffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                dimBuffer.putInt(vector.length);
                out.write(dimBuffer.array());

                // Write vector values in little-endian
                for (float value : vector) {
                    ByteBuffer valueBuffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                    valueBuffer.putFloat(value);
                    out.write(valueBuffer.array());
                }
            }
        }
    }

    @Test
    public void testBasicFunctionality() throws IOException {
        // Test parsing fvec format, size/dimension calculation, and vector retrieval
        float[][] testVectors = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {-5.0f, -6.0f, 0.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 0.0f, 15.0f, 16.0f},
            {17.0f, 18.0f, -19.0f, 20.0f},
        };
        Path tempFile = tempDir.resolve("test.fvec");
        writeFvecFile(tempFile, testVectors);

        try (var ravv = new FvecSegmentRavv(tempFile)) {
            assertEquals(5, ravv.size());
            assertEquals(5L, ravv.sizeLong());

            assertEquals(4, ravv.dimension());

            // Test random access
            var randomOrder = new int[]{3, 1, 2, 0};
            for (int i : randomOrder) {
                VectorFloat<?> vector = ravv.getVector(i);
                assertNotNull(vector);
                assertEquals(4, vector.length());

                for (int j = 0; j < testVectors[i].length; j++) {
                    assertEquals(testVectors[i][j], vector.get(j), 0.0001f);
                }
            }
        }
    }

    @Test
    public void testInterfaceMethods() throws IOException {
        float[][] testVectors = {{1.0f, 2.0f}};
        Path tempFile = tempDir.resolve("test.fvec");
        writeFvecFile(tempFile, testVectors);

        try (var ravv = new FvecSegmentRavv(tempFile)) {
            // New vector is created every time
            assertFalse(ravv.isValueShared());

            // New vector is created so copies can be no-ops
            assertSame(ravv, ravv.copy());
        }
    }

    @Test
    public void testNonExistentFile() {
        Path nonExistent = tempDir.resolve("nonexistent.fvec");
        assertThrows(IOException.class, () -> new FvecSegmentRavv(nonExistent));
    }
}
