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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The host-side parts of the GPU build, without a GPU. */
class TestGpuBuildHost {

    private static MemorySegment rows(Arena arena, float[][] values) {
        int dim = values[0].length;
        MemorySegment rows = arena.allocate((long) values.length * dim * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            MemorySegment.copy(values[i], 0, rows, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
        }
        return rows;
    }

    @Test
    void packsToFp16PaddedToSixteenDimensions() {
        float[][] values = new float[3][20];
        for (int i = 0; i < 3; i++) {
            for (int d = 0; d < 20; d++) {
                values[i][d] = (i + 1) * 0.1f * (d - 10);
            }
        }
        try (Arena arena = Arena.ofShared()) {
            var packed = PackedVectors.of(rows(arena, values), 3, 20, false, true);
            assertEquals(32, packed.paddedDim);
            for (int i = 0; i < 3; i++) {
                float norm = 0;
                for (int d = 0; d < 32; d++) {
                    float expected = d < 20 ? Float.float16ToFloat(Float.floatToFloat16(values[i][d])) : 0;
                    assertEquals(expected, packed.get(i, d), 0);
                    norm += expected * expected;
                }
                assertEquals(norm, packed.squaredNorms[i], 1e-4f);
            }
        }
    }

    @Test
    void normalisesWhilePacking() {
        float[][] values = { { 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
        try (Arena arena = Arena.ofShared()) {
            var packed = PackedVectors.of(rows(arena, values), 1, 16, true, false);
            assertEquals(0.6f, packed.get(0, 0), 1e-3f);
            assertEquals(0.8f, packed.get(0, 1), 1e-3f);
        }
    }

    @Test
    void scoresMatchJVector() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?> a = vts.createFloatVector(new float[] { 0.6f, 0.8f });
        VectorFloat<?> b = vts.createFloatVector(new float[] { 1.0f, 0.0f });
        float dot = 0.6f;
        float squaredDistance = 0.16f + 0.64f;
        assertEquals(VectorSimilarityFunction.DOT_PRODUCT.compare(a, b), Candidates.Metric.INNER_PRODUCT.score(dot), 1e-6f);
        assertEquals(VectorSimilarityFunction.COSINE.compare(a, b), Candidates.Metric.COSINE.score(1 - dot), 1e-6f);
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(a, b), Candidates.Metric.EUCLIDEAN.score(squaredDistance), 1e-6f);
    }

    @Test
    void ordersScoresDescending() {
        float[] scores = { -1f, -0.5f, -0f, 0f, 0.25f, 0.9999f, 1f };
        for (int i = 0; i + 1 < scores.length; i++) {
            assertTrue(GpuGraphBuilder.descendingKey(scores[i]) >= GpuGraphBuilder.descendingKey(scores[i + 1]));
        }
    }

    @Test
    void declinesWithoutTheRuntimeOrForLargeDegrees() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> data = List.of(vts.createFloatVector(new float[] { 1, 0 }), vts.createFloatVector(new float[] { 0, 1 }));
        var vectors = new ListRandomAccessVectorValues(data, 2);
        var accelerator = new TornadoGraphBuildAccelerator();
        // no TornadoVM runtime on the test class path
        assertFalse(accelerator.supports(new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, 16, 100, 1.2f, 1.2f, false),
                vectors, VectorSimilarityFunction.COSINE));
        assertFalse(accelerator.supports(new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, 128, 100, 1.2f, 1.2f, false),
                vectors, VectorSimilarityFunction.COSINE));
    }

    @Test
    void copiesNormalisesAndGathersRows() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> data = List.of(vts.createFloatVector(new float[] { 3, 4 }), vts.createFloatVector(new float[] { 0, 2 }),
                vts.createFloatVector(new float[] { 1, 0 }));
        var vectors = new ListRandomAccessVectorValues(data, 2);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment rows = HostRows.copy(vectors, arena, false);
            assertEquals(4f, rows.getAtIndex(ValueLayout.JAVA_FLOAT, 1));
            assertEquals(2f, rows.getAtIndex(ValueLayout.JAVA_FLOAT, 3));
            MemorySegment unit = HostRows.copy(vectors, arena, true);
            assertEquals(0.6f, unit.getAtIndex(ValueLayout.JAVA_FLOAT, 0), 1e-6f);
            assertEquals(1f, unit.getAtIndex(ValueLayout.JAVA_FLOAT, 3), 1e-6f);
            MemorySegment gathered = HostRows.gather(rows, new int[] { 2, 0 }, 2, arena);
            assertEquals(1f, gathered.getAtIndex(ValueLayout.JAVA_FLOAT, 0));
            assertEquals(3f, gathered.getAtIndex(ValueLayout.JAVA_FLOAT, 2));
        }
    }

    @Test
    void keepsTheFirstCandidatesOfEveryRow() {
        var all = new Candidates.TopK(3, new int[] { 1, 2, 3, 4, 5, 6 }, new float[] { .9f, .8f, .7f, .6f, .5f, .4f });
        var first = all.first(2);
        assertEquals(2, first.k());
        assertArrayEquals(new int[] { 1, 2, 4, 5 }, first.ids());
        assertArrayEquals(new float[] { .9f, .8f, .6f, .5f }, first.scores());
        assertSame(all, all.first(3));
    }

    @Test
    void batchesOnlyWhatDoesNotFitOnTheDevice() {
        long gpu = 24L << 30;
        // 1M x 1536 fits: one batch, no overlap
        assertEquals(new Candidates.Batching(1, 1), Candidates.Batching.of(1_000_000, 1536, gpu, 0, 2));
        // 10M x 1024 does not: each batch holds overlap/clusters of the rows and must fit in 70% of the device
        assertEquals(new Candidates.Batching(8, 2), Candidates.Batching.of(10_000_000, 1024, gpu, 0, 2));
        assertEquals(new Candidates.Batching(4, 1), Candidates.Batching.of(10_000_000, 1024, gpu, 0, 1));
        // cuVS needs more clusters than the overlap
        assertEquals(new Candidates.Batching(3, 2), Candidates.Batching.of(3_000_000, 1024, gpu, 0, 2));
        // set by the user
        assertEquals(new Candidates.Batching(7, 2), Candidates.Batching.of(10_000_000, 1024, gpu, 7, 2));
    }

    @Test
    void parsesTheDeviceMemoryBudget() {
        String old = System.getProperty("tornado.device.memory");
        try {
            System.setProperty("tornado.device.memory", "22GB");
            assertEquals(22L << 30, TornadoGraphBuildAccelerator.deviceMemoryBudget());
            System.setProperty("tornado.device.memory", "512MB");
            assertEquals(512L << 20, TornadoGraphBuildAccelerator.deviceMemoryBudget());
        } finally {
            if (old == null) {
                System.clearProperty("tornado.device.memory");
            } else {
                System.setProperty("tornado.device.memory", old);
            }
        }
    }

    @Test
    void estimatesTheHostMemoryOfABuild() {
        var copied = HostMemory.need(10_000_000, 1024, 32, false);
        var mapped = HostMemory.need(10_000_000, 1024, 32, true);
        // the same heap; a mapped dataset saves the float32 copy of its rows
        assertEquals(copied.heap(), mapped.heap());
        assertEquals(10_000_000L * 1024 * Float.BYTES, copied.nativeMemory() - mapped.nativeMemory());
        // 10M x 1024, which builds on a 62 GB host with a 56 GB heap: about 19 GB of heap and 33 GB of native memory
        assertTrue(mapped.heap() > 15e9 && mapped.heap() < 25e9, "heap " + mapped.heap());
        assertTrue(mapped.nativeMemory() > 25e9 && mapped.nativeMemory() < 40e9, "native " + mapped.nativeMemory());
    }

    @Test
    void declinesABuildTheHostCannotHold() {
        var need = new HostMemory.Need(20L << 30, 30L << 30);
        assertEquals(null, HostMemory.shortfall(need, 50L << 30, 1L << 30, 59L << 30, null));
        String heap = HostMemory.shortfall(need, 10L << 30, 1L << 30, 59L << 30, null);
        assertTrue(heap != null && heap.contains("heap"), heap);
        // enough heap, but growing into it and the native allocations together exceed the memory available
        String memory = HostMemory.shortfall(need, 50L << 30, 1L << 30, 40L << 30, null);
        assertTrue(memory != null && memory.contains("memory available"), memory);
        // a fixed budget replaces both checks
        assertEquals(null, HostMemory.shortfall(need, 0, 0, 0, "64GB"));
        String fixed = HostMemory.shortfall(need, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, "32GB");
        assertTrue(fixed != null && fixed.contains("jvector.gpu.hostMemory=32GB"), fixed);
    }

    @Test
    void parsesHostMemorySizes() {
        assertEquals(48L << 30, HostMemory.parseBytes("48GB"));
        assertEquals(512L << 20, HostMemory.parseBytes(" 512mb "));
        assertEquals(12_345_678L * 1024, HostMemory.parseMemInfoLine("MemAvailable:   12345678 kB"));
        assertTrue(HostMemory.availableNative() > 0);
        assertTrue(HostMemory.freeHeap() > 0);
    }
}
