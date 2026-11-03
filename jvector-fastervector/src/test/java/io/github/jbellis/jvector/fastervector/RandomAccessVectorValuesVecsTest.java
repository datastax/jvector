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

package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RandomAccessVectorValuesVecsTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static RandomAccessVectorValues ravv(int dim, float[]... rows) {
        List<VectorFloat<?>> vectors = new ArrayList<>();
        for (float[] row : rows) {
            vectors.add(vts.createFloatVector(row));
        }
        return new ListRandomAccessVectorValues(vectors, dim);
    }

    private static float[] readAll(VecsCursor c, int dim, int n) {
        float[] out = new float[n * dim];
        try (Arena a = Arena.ofConfined()) {
            MemorySegment seg = a.allocate((long) dim * Float.BYTES, 4);
            int i = 0;
            while (c.nextVectorInto(seg)) {
                for (int j = 0; j < dim; j++) {
                    out[i++] = seg.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                }
            }
        }
        return out;
    }

    @Test
    void sizeAndContents() {
        Vecs v = new RandomAccessVectorValuesVecs(ravv(2, new float[]{1, 2}, new float[]{3, 4}, new float[]{5, 6}));
        assertEquals(3, v.nvecs());
        assertEquals(2, v.dim());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, readAll(v.getCursor(), 2, 3));
    }

    @Test
    void cursorsAreIndependent() {
        Vecs v = new RandomAccessVectorValuesVecs(ravv(1, new float[]{1}, new float[]{2}));
        VecsCursor first = v.getCursor();
        assertArrayEquals(new float[]{1, 2}, readAll(first, 1, 2));
        assertArrayEquals(new float[]{1, 2}, readAll(v.getCursor(), 1, 2));
    }

    @Test
    void seekAndExhaustion() {
        Vecs v = new RandomAccessVectorValuesVecs(ravv(1, new float[]{1}, new float[]{2}, new float[]{3}));
        VecsCursor c = v.getCursor();
        c.seek(2);
        assertEquals(2, c.ord());
        assertArrayEquals(new float[]{3, 0, 0}, readAll(c, 1, 3));
        assertEquals(3, c.ord());
        c.seek(3);
        try (Arena a = Arena.ofConfined()) {
            assertFalse(c.nextVectorInto(a.allocate(Float.BYTES)));
        }
        assertThrows(IllegalArgumentException.class, () -> c.seek(4));
        assertThrows(IllegalArgumentException.class, () -> c.seek(-1));
    }

    @Test
    void wrongLengthVectorRejected() {
        Vecs v = new RandomAccessVectorValuesVecs(ravv(2, new float[]{1, 2}, new float[]{3}));
        VecsCursor c = v.getCursor();
        try (Arena a = Arena.ofConfined()) {
            MemorySegment seg = a.allocate(2L * Float.BYTES);
            assertTrue(c.nextVectorInto(seg));
            assertThrows(IllegalStateException.class, () -> c.nextVectorInto(seg));
        }
    }
}
