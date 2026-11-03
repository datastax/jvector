package io.github.jbellis.jvector.fastervector.utils;

import io.github.jbellis.jvector.fastervector.Vecs;
import io.github.jbellis.jvector.fastervector.VecsCursor;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class HeapVecsTest {

    private static float[] readAll(Vecs v) {
        VecsCursor c = v.getCursor();
        float[] out = new float[(int) (v.nvecs() * v.dim())];
        try (Arena a = Arena.ofConfined()) {
            MemorySegment seg = a.allocate((long) v.dim() * Float.BYTES, 4);
            int i = 0;
            while (c.nextVectorInto(seg)) {
                for (int j = 0; j < v.dim(); j++) {
                    out[i++] = seg.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                }
            }
        }
        return out;
    }

    @Test
    void list() {
        Vecs v = new ListVecs(List.of(new float[]{1, 2}, new float[]{3, 4}, new float[]{5, 6}));
        assertEquals(3, v.nvecs());
        assertEquals(2, v.dim());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, readAll(v));
    }

    @Test
    void array2d() {
        Vecs v = new Array2DVecs(new float[][]{{1, 2, 3}, {4, 5, 6}});
        assertEquals(2, v.nvecs());
        assertEquals(3, v.dim());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, readAll(v));
    }

    @Test
    void flatDense() {
        Vecs v = new FlatArrayVecs(new float[]{1, 2, 3, 4, 5, 6}, 2);
        assertEquals(3, v.nvecs());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, readAll(v));
    }

    @Test
    void flatStrided() {
        // dim 2, stride 3: padding element after each vector (last one may be absent)
        Vecs v = new FlatArrayVecs(new float[]{1, 2, 9, 3, 4, 9, 5, 6}, 2, 3);
        assertEquals(3, v.nvecs());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, readAll(v));
    }

    @Test
    void seekAndExhaustion() {
        Vecs v = new Array2DVecs(new float[][]{{1}, {2}, {3}});
        VecsCursor c = v.getCursor();
        c.seek(2);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment seg = a.allocate(4, 4);
            assertTrue(c.nextVectorInto(seg));
            assertEquals(3f, seg.get(ValueLayout.JAVA_FLOAT, 0));
            assertFalse(c.nextVectorInto(seg));
        }
        assertThrows(IllegalArgumentException.class, () -> c.seek(4));
    }

    @Test
    void validation() {
        assertThrows(IllegalArgumentException.class, () -> new ListVecs(List.of()));
        assertThrows(IllegalArgumentException.class,
            () -> new Array2DVecs(new float[][]{{1, 2}, {3}}));
        assertThrows(IllegalArgumentException.class, () -> new FlatArrayVecs(new float[4], 0));
        assertThrows(IllegalArgumentException.class, () -> new FlatArrayVecs(new float[4], 3, 2));
    }
}
