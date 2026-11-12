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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LazyMemorySegmentTest {
    @Test
    void scalarReadsAndWritesPreserveBitsAndBoundsBeforeAndAfterNativeUse() {
        float[] data = {-0.0f, Float.intBitsToFloat(0x7fc00123), Float.POSITIVE_INFINITY, 3};
        var v = new MemorySegmentVectorFloat(data);
        for (int phase = 0; phase < 2; phase++) {
            for (int i = 0; i < data.length; i++)
                assertEquals(Float.floatToRawIntBits(data[i]), Float.floatToRawIntBits(v.get(i)));
            v.set(3, -0.0f);
            assertEquals(0x80000000, Float.floatToRawIntBits(data[3]));
            data[3] = Float.intBitsToFloat(0x7fc00456);
            assertEquals(0x7fc00456, Float.floatToRawIntBits(v.get(3)));
            assertThrows(ArrayIndexOutOfBoundsException.class, () -> v.get(-1));
            assertThrows(ArrayIndexOutOfBoundsException.class, () -> v.get(Integer.MAX_VALUE));
            assertThrows(ArrayIndexOutOfBoundsException.class, () -> v.set(data.length, 0));
            v.get();
        }
    }

    @Test
    void concurrentFirstUsePublishesViewsOfTheOriginalArray() throws Exception {
        float[] data = {-0.0f, 1, 2, 3, 4, 5, 6, 7};
        var v = new MemorySegmentVectorFloat(data);
        assertSame(data, v.array());
        assertEquals(2, java.util.Arrays.stream(MemorySegmentVectorFloat.class.getDeclaredFields())
                .filter(f -> !java.lang.reflect.Modifier.isStatic(f.getModifiers())).count());
        var field = MemorySegmentVectorFloat.class.getDeclaredField("data");
        field.setAccessible(true);
        assertSame(data, field.get(v));
        try (var pool = Executors.newFixedThreadPool(48)) {
            var barrier = new CyclicBarrier(48);
            var tasks = new ArrayList<Future<MemorySegment>>();
            for (int i = 0; i < 48; i++) tasks.add(pool.submit(() -> {
                barrier.await();
                var first = v.get();
                for (int n = 0; n < 10000; n++) {
                    // The cache is a benign race: threads may see different view objects,
                    // but every view must address the same backing array.
                    assertSameView(data, v.get());
                    assertEquals(data.length, v.length());
                    assertEquals(Float.floatToRawIntBits(data[n & 7]), Float.floatToRawIntBits(v.get(n & 7)));
                }
                return first;
            }));
            MemorySegment first = tasks.get(0).get();
            for (var task : tasks) assertSameView(data, task.get());
            v.set(3, 19);
            assertEquals(19, first.getAtIndex(ValueLayout.JAVA_FLOAT, 3));
            first.setAtIndex(ValueLayout.JAVA_FLOAT, 4, 23);
            assertEquals(23, v.get(4));
        }
    }

    @Test
    void copiesZeroAndBufferKeepStorageSemantics() throws Exception {
        assertThrows(NullPointerException.class, () -> new MemorySegmentVectorFloat((float[]) null));
        float[] data = {-0.0f, Float.intBitsToFloat(0x7fc00123), 2, 3, 4, 5, 6, 7};
        var original = new MemorySegmentVectorFloat(data);
        var copy = (MemorySegmentVectorFloat) original.copy();
        var field = MemorySegmentVectorFloat.class.getDeclaredField("data");
        field.setAccessible(true);
        assertSame(data, field.get(original));
        assertSame(copy.array(), field.get(copy));
        assertNotSame(data, copy.array());
        for (int i = 0; i < data.length; i++)
            assertEquals(Float.floatToRawIntBits(data[i]), Float.floatToRawIntBits(copy.get(i)));
        assertEquals(original, copy);
        copy.copyFrom(copy, 0, 1, 7);
        assertEquals(0x80000000, Float.floatToRawIntBits(copy.get(1)));
        assertEquals(6, copy.get(7));
        var buffer = FloatBuffer.wrap(new float[]{9, 10, 11});
        buffer.position(1);
        var fromBuffer = new MemorySegmentVectorFloat(buffer);
        assertArrayEquals(new float[]{10, 11}, fromBuffer.array());
        assertEquals(2, fromBuffer.length());
        fromBuffer.zero();
        for (float value : fromBuffer.array()) assertEquals(0, Float.floatToRawIntBits(value));
    }

    private static void assertSameView(float[] data, MemorySegment view) {
        assertSame(data, view.heapBase().orElseThrow());
        assertEquals(0, view.address());
        assertEquals((long) data.length * Float.BYTES, view.byteSize());
    }

    @Test
    void panamaPathBelowThresholdDoesNotMaterializeSegment() throws Exception {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var field = MemorySegmentVectorFloat.class.getDeclaredField("segment");
        field.setAccessible(true);
        var a = new MemorySegmentVectorFloat(new float[]{1, 2, 3, 4, 5, 6, 7, 8});
        var b = new MemorySegmentVectorFloat(new float[]{8, 7, 6, 5, 4, 3, 2, 1});
        support.dotProduct(a, b);
        support.cosine(a, b);
        support.squareDistance(a, b);
        assertNull(field.get(a));
        assertNull(field.get(b));
    }

    @Test
    void nativeAndPanamaAgreeAcrossThreshold() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var rnd = new java.util.Random(42);
        for (int dim : new int[]{7, 64, 255, 256, 257, 512, 1031}) {
            float[] x = new float[dim], y = new float[dim];
            for (int i = 0; i < dim; i++) { x[i] = rnd.nextFloat() - 0.5f; y[i] = rnd.nextFloat() - 0.5f; }
            double dot = 0, l2 = 0, nx = 0, ny = 0;
            for (int i = 0; i < dim; i++) {
                dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i];
                double d = x[i] - y[i]; l2 += d * d;
            }
            var a = new MemorySegmentVectorFloat(x);
            var b = new MemorySegmentVectorFloat(y);
            assertEquals(dot, support.dotProduct(a, b), 1e-3, "dot dim=" + dim);
            assertEquals(l2, support.squareDistance(a, b), 1e-3, "l2 dim=" + dim);
            assertEquals(dot / Math.sqrt(nx * ny), support.cosine(a, b), 1e-4, "cosine dim=" + dim);
        }
    }

    @Test
    void nativeUseKeepsStorageSemantics() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        // Above the native threshold so the call actually crosses into native code.
        float[] data = new float[1024];
        java.util.Arrays.fill(data, 10);
        var vector = new MemorySegmentVectorFloat(data);
        assertEquals(0, support.squareDistance(vector, vector));
        vector.zero();
        for (float value : vector.array()) assertEquals(0, Float.floatToRawIntBits(value));
    }
}
