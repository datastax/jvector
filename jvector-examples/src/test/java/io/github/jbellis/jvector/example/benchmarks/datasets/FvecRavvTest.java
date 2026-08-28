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
 
package io.github.jbellis.jvector.example.benchmarks.datasets;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class FvecRavvTest {

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static float[] toArray(VectorFloat<?> vec) {
        var arr = new float[vec.length()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = vec.get(i);
        }
        return arr;
    }

    @TempDir Path testDir;
    Path fvecPath;
    float[][] referenceVecs;
    int dim;

    @BeforeEach
    void createTestData() throws IOException {
        dim = 4;
        referenceVecs = new float[][]{
            new float[]{0.0f, 0.1f, 0.2f, 0.3f},
            new float[]{1.0f, 1.1f, 1.2f, 1.3f},
            new float[]{2.0f, 2.1f, 2.2f, 2.3f},
            new float[]{3.0f, 3.1f, 3.2f, 3.3f},
            new float[]{4.0f, 4.1f, 4.2f, 4.3f}
        };
        fvecPath = testDir.resolve("fvecs");
        DataSetLoaderSimpleMFDTest.writeTestFvecs(fvecPath, dim, referenceVecs);
    }


    @Test
    void testGetVector() throws IOException {
        var ravv = FvecRavv.of(fvecPath);
        assertEquals(referenceVecs.length, ravv.size());
        assertEquals(dim, ravv.dimension());
        for (int i = 0; i < referenceVecs.length; i++) {
            assertArrayEquals(referenceVecs[i], toArray(ravv.getVector(i)));
        }
    }

    @Test
    void testGetVectorInto() throws IOException {
        var vec = vts.createFloatVector(dim);
        var ravv = FvecRavv.of(fvecPath);
        ravv.getVectorInto(3, vec, 0);
        assertArrayEquals(referenceVecs[3], toArray(vec));
    }

    @Test
    void testCopyReturnsDistinctVec() throws IOException {
        var ravv = FvecRavv.of(fvecPath);
        var a = ravv.getVector(2);
        var ravvCopy = ravv.copy();
        var b = ravvCopy.getVector(4);

        assertNotSame(a, b);
        assertArrayEquals(referenceVecs[2], toArray(a));
        assertArrayEquals(referenceVecs[4], toArray(b));
    }

    @Test
    void testMultipleChunks() throws IOException {
        // little more than the space needed for 2 complete vecs per chunk, but less that 3
        int maxBytesPerChunk = (Float.BYTES * dim + Integer.BYTES) * 2 + 2;
        var ravv = FvecRavv.of(fvecPath, maxBytesPerChunk);

        assertEquals(3, ravv.getNumGroups());
        for (int i = 0; i < referenceVecs.length; i++) {
            assertArrayEquals(referenceVecs[i], toArray(ravv.getVector(i)));
        }
    }
}
