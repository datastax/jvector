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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import static org.junit.Assert.*;

public class TestFvecFileVectorValues {
    private static float value(int row, int column) { return column == 0 ? -0.0f : (row * 7 + column * 13) % 201 / 100f - 1; }
    private static Path file(int rows, int dimension) throws IOException {
        Path path = Files.createTempFile("vector-source-test", ".fvecs");
        var bytes = ByteBuffer.allocate(rows * (dimension + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            bytes.putInt(dimension);
            for (int col = 0; col < dimension; col++) bytes.putFloat(value(row, col));
        }
        Files.write(path, bytes.array()); return path;
    }
    private static void equal(int row, VectorFloat<?> v) {
        for (int col = 0; col < v.length(); col++) assertEquals(Float.floatToRawIntBits(value(row, col)), Float.floatToRawIntBits(v.get(col)));
    }

    @Test public void optionsAreImmutableAndValidateLimits() throws Exception {
        var defaults = FvecFileVectorValues.Options.defaults();
        var options = defaults.withMaxBufferBytes(1024).withIoThreads(2)
                .withBatchVectors(7).withReadAhead(0);
        assertEquals(64L << 20, defaults.maxBufferBytes());
        assertEquals(48, defaults.ioThreads());
        assertEquals(64, defaults.batchVectors());
        assertEquals(3, defaults.readAhead());
        assertEquals(1024, options.maxBufferBytes());
        assertEquals(2, options.ioThreads());
        assertEquals(7, options.batchVectors());
        assertEquals(0, options.readAhead());
        assertThrows(IllegalArgumentException.class, () -> defaults.withMaxBufferBytes(0));
        assertThrows(IllegalArgumentException.class, () -> defaults.withIoThreads(0));
        assertThrows(IllegalArgumentException.class, () -> defaults.withBatchVectors(0));
        assertThrows(IllegalArgumentException.class, () -> defaults.withReadAhead(-1));
        var path = file(257, 17);
        try (var source = FvecFileVectorValues.open(path, options)) {
            assertEquals(257 * 18L * 4 / 100, source.statistics().bufferLimitBytes);
            try (var cursor = VectorAccess.openRange(source, 7, 10)) {
                int row = 7;
                while (cursor.next()) { assertEquals(row, cursor.ordinal()); equal(row++, cursor.vector()); }
                assertEquals(10, row);
            }
            try (var cursor = source.openRange(source.size(), source.size())) {
                assertFalse(cursor.next());
            }
            assertThrows(IndexOutOfBoundsException.class, () -> source.openRange(10, 7));
            assertThrows(NullPointerException.class, () -> FvecFileVectorValues.open(path, null));
        } finally { Files.delete(path); }
    }

    @Test public void copiedCursorVectorSurvivesAdvanceAndSourceClose() throws Exception {
        var path = file(5, 17);
        VectorFloat<?> retained;
        try (var source = FvecFileVectorValues.open(path)) {
            try (var cursor = VectorAccess.openRange(source, 1, 3)) {
                assertTrue(cursor.isValueShared());
                assertTrue(cursor.next());
                var borrowed = cursor.vector();
                retained = borrowed.copy();
                assertTrue(cursor.next());
                assertSame(borrowed, cursor.vector());
                equal(2, borrowed);
                equal(1, retained);
                assertFalse(cursor.next());
            }
        } finally { Files.delete(path); }
        equal(1, retained);
    }

    @Test(timeout=60000) public void concurrentRangeSelectionAndPointReadsWithOneRecordBudget() throws Exception {
        for (int dimension : new int[] {17, 768, 3072}) {
            var path = file(257, dimension);
            try (var source = FvecFileVectorValues.open(path, FvecFileVectorValues.Options.defaults()
                    .withMaxBufferBytes((dimension + 1L) * 4).withIoThreads(12))) {
                var executor = Executors.newFixedThreadPool(48);
                try {
                    var tasks = new ArrayList<Callable<Void>>();
                    for (int worker = 0; worker < 48; worker++) {
                        final int seed = worker;
                        tasks.add(() -> {
                            try (var copy = source.copy()) {
                                for (int i = 0; i < 32; i++) equal((i + seed) % source.size(), copy.getVector((i + seed) % source.size()));
                            }
                            try (var cursor = source.openRange(seed, seed + 17)) {
                                int ordinal = seed;
                                while (cursor.next()) { assertEquals(ordinal, cursor.ordinal()); equal(ordinal++, cursor.vector()); }
                            }
                            int[] ordinals = {256, seed, 0, 256, seed};
                            try (var cursor = source.openSelection(ordinals, 0, ordinals.length)) {
                                int i = 0;
                                while (cursor.next()) { assertEquals(ordinals[i], cursor.ordinal()); equal(ordinals[i++], cursor.vector()); }
                            }
                            return null;
                        });
                    }
                    for (var f : executor.invokeAll(tasks)) f.get();
                } finally { executor.shutdown(); }
                assertTrue(source.statistics().bufferBytes <= source.statistics().bufferLimitBytes);
                assertEquals((dimension + 1L) * 4, source.statistics().bufferBytes);
                assertThrows(IndexOutOfBoundsException.class, () -> source.openRange(250, 258));
                assertThrows(IndexOutOfBoundsException.class, () -> source.openSelection(new int[] {257}, 0, 1));
            } finally { Files.delete(path); }
        }
    }

    @Test(timeout=60000) public void materializationOrderAndEncodingMatchResidentSource() throws Exception {
        var path = file(2049, 17);
        var pool = new ForkJoinPool(8);
        try (var source = FvecFileVectorValues.open(path)) {
            var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            List<VectorFloat<?>> values = new ArrayList<>();
            for (int row = 0; row < source.size(); row++) {
                float[] v = new float[17]; for (int col = 0; col < 17; col++) v[col] = value(row, col);
                values.add(vts.createFloatVector(v));
            }
            var resident = new ListRandomAccessVectorValues(values, 17);
            int[] ordinals = {2048, 0, 99, 2048, 10, 9, 0};
            var materialized = VectorAccess.copySelected(source, ordinals, pool);
            for (int i = 0; i < ordinals.length; i++) equal(ordinals[i], materialized.get(i));
            // Closing releases selection identity: the caller can reuse and modify its array.
            try (var cursor = source.openSelection(ordinals, 0, ordinals.length)) {
                assertTrue(cursor.next()); equal(ordinals[0], cursor.vector());
            }
            ordinals[0] = 42;
            try (var cursor = source.openSelection(ordinals, 0, ordinals.length)) {
                assertTrue(cursor.next()); equal(42, cursor.vector());
            }
            var pq = ProductQuantization.compute(resident, 2, 16, false);
            assertEquals(pq.encodeAll(resident, pool), pq.encodeAll(source, pool));
            var nvq = NVQuantization.compute(resident, 1);
            var nvqFile = NVQuantization.compute(source, 1);
            assertEquals(nvq, nvqFile);
            var residentNVQ = nvq.encodeAll(resident, pool);
            var fileNVQ = nvq.encodeAll(source, pool);
            assertEquals(residentNVQ, fileNVQ);
            try (var cursor = source.openRange(0, 0)) { assertFalse(cursor.next()); }
            var copy = source.copy(); copy.close(); equal(0, source.getVector(0));
            assertThrows(IllegalStateException.class, () -> copy.getVector(0));
        } finally { pool.shutdown(); Files.delete(path); }
    }

    @Test public void errorsAndCloseInvalidateCopies() throws Exception {
        var path = file(5, 17);
        var source = FvecFileVectorValues.open(path);
        var copy = source.copy();
        var cursor = source.openRange(0, 3); cursor.close(); cursor.close();
        source.close(); source.close();
        assertThrows(IllegalStateException.class, () -> copy.getVector(0));
        assertThrows(IllegalStateException.class, () -> source.openRange(0, 1));
        Files.write(path, new byte[] {1, 2});
        assertThrows(IOException.class, () -> FvecFileVectorValues.open(path));
        Files.write(path, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(-1).array());
        assertThrows(IOException.class, () -> FvecFileVectorValues.open(path));
        Files.write(path, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(Integer.MAX_VALUE).array());
        assertThrows(IOException.class, () -> FvecFileVectorValues.open(path));
        assertThrows(IndexOutOfBoundsException.class, () ->
                NVQuantization.compute(new ListRandomAccessVectorValues(new ArrayList<>(), 17), 1));
        Files.delete(path);
        final Path malformed = file(5, 17);
        try {
            byte[] bytes = Files.readAllBytes(malformed);
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(18 * 4, 18);
            Files.write(malformed, bytes);
            try (var bad = FvecFileVectorValues.open(malformed)) {
                assertThrows(UncheckedIOException.class, () -> bad.getVector(1));
            }
        } finally { Files.delete(malformed); }
    }
}
