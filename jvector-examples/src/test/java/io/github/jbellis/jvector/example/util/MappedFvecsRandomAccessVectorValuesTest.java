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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.example.benchmarks.datasets.InMemoryCachedDataSet;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for {@link MappedFvecsRandomAccessVectorValues}, including slab boundaries on small files.
public class MappedFvecsRandomAccessVectorValuesTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    static float[][] randomVectors(int count, int dimension, long seed) {
        var random = new Random(seed);
        float[][] vectors = new float[count][dimension];
        for (float[] v : vectors) {
            for (int d = 0; d < dimension; d++) {
                v[d] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

    static void writeFvecs(Path path, float[][] vectors, int[] headerDims) throws IOException {
        int dimension = vectors[0].length;
        int bytesPerVector = Integer.BYTES + dimension * Float.BYTES;
        var buf = ByteBuffer.allocate(vectors.length * bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < vectors.length; i++) {
            buf.putInt(headerDims == null ? dimension : headerDims[i]);
            for (float v : vectors[i]) buf.putFloat(v);
        }
        Files.write(path, buf.array());
    }

    static void assertVectorEquals(float[] expected, VectorFloat<?> actual, int offset) {
        for (int d = 0; d < expected.length; d++) {
            assertEquals(expected[d], actual.get(offset + d), 0f, "component " + d);
        }
    }

    @Test
    public void matchesStreamingReaderAcrossSlabBoundaries() throws IOException {
        int count = 1000, dimension = 7;
        float[][] expected = randomVectors(count, dimension, 42);
        Path file = tempFolder.newFile("base.fvecs").toPath();
        writeFvecs(file, expected, null);

        int stride = Integer.BYTES + dimension * Float.BYTES;
        var mapped = new MappedFvecsRandomAccessVectorValues(file, 5L * stride + 3); // 5 records per slab
        assertEquals(count, mapped.size());
        assertEquals(dimension, mapped.dimension());
        assertEquals(200, mapped.slabCount());
        assertTrue(mapped.isValueShared());
        assertEquals(file, mapped.getPath());

        List<VectorFloat<?>> streamed = SiftLoader.readFvecs(file.toString());
        assertEquals(count, streamed.size());
        for (int i = 0; i < count; i++) {
            assertVectorEquals(expected[i], mapped.getVector(i), 0);
            assertVectorEquals(expected[i], streamed.get(i), 0);

            var dest = vts.createFloatVector(2 * dimension);
            mapped.getVectorInto(i, dest, dimension);
            assertVectorEquals(expected[i], dest, dimension);
        }

        // a default-sized mapping of the same file is a single slab with identical contents
        var single = new MappedFvecsRandomAccessVectorValues(file);
        assertEquals(1, single.slabCount());
        for (int i = 0; i < count; i += 97) {
            assertVectorEquals(expected[i], single.getVector(i), 0);
        }
    }

    @Test
    public void rangeViewsReadRebasedOrdinals() throws IOException {
        float[][] expected = randomVectors(50, 3, 7);
        Path file = tempFolder.newFile("base.fvecs").toPath();
        writeFvecs(file, expected, null);
        var mapped = new MappedFvecsRandomAccessVectorValues(file, 4L * (Integer.BYTES + 3 * Float.BYTES));

        var view = mapped.range(17, 31);
        assertEquals(14, view.size());
        assertTrue(view.isValueShared());
        for (int i = 0; i < view.size(); i++) {
            assertVectorEquals(expected[17 + i], view.getVector(i), 0);
        }
        assertThrows(IndexOutOfBoundsException.class, () -> view.getVector(14));
    }

    @Test
    public void copiesShareTheMappingButNotTheScratchVector() throws IOException {
        float[][] expected = randomVectors(4, 5, 11);
        Path file = tempFolder.newFile("base.fvecs").toPath();
        writeFvecs(file, expected, null);
        var a = new MappedFvecsRandomAccessVectorValues(file);
        var b = a.copy();
        assertNotSame(a, b);
        assertTrue(b instanceof MappedFvecsRandomAccessVectorValues);
        assertEquals(a.slabCount(), ((MappedFvecsRandomAccessVectorValues) b).slabCount());

        VectorFloat<?> fromA = a.getVector(0);
        VectorFloat<?> fromB = b.getVector(3);
        assertNotSame(fromA, fromB);
        assertVectorEquals(expected[0], fromA, 0);
        assertVectorEquals(expected[3], fromB, 0);

        // the shared scratch is overwritten by the next read on the same instance
        assertSame(fromA, a.getVector(1));
        assertVectorEquals(expected[1], fromA, 0);
    }

    @Test
    public void parallelCachingThroughRangesMatchesFile() throws IOException {
        int count = 3000, dimension = 16;
        float[][] expected = randomVectors(count, dimension, 99);
        Path file = tempFolder.newFile("base.fvecs").toPath();
        writeFvecs(file, expected, null);
        var mapped = new MappedFvecsRandomAccessVectorValues(file, 64L * (Integer.BYTES + dimension * Float.BYTES));

        List<VectorFloat<?>> cached = InMemoryCachedDataSet.readAllVectors(mapped);
        assertEquals(count, cached.size());
        for (int i = 0; i < count; i++) {
            assertVectorEquals(expected[i], cached.get(i), 0);
        }
        // every cached vector is an independent object
        assertNotSame(cached.get(0), cached.get(1));
        var ravv = new ListRandomAccessVectorValues(cached, dimension);
        assertFalse(ravv.isValueShared());
    }

    @Test
    public void rejectsMalformedFiles() throws IOException {
        Path empty = tempFolder.newFile("empty.fvecs").toPath();
        assertThrows(IOException.class, () -> new MappedFvecsRandomAccessVectorValues(empty));

        Path zeroDim = tempFolder.newFile("zero.fvecs").toPath();
        Files.write(zeroDim, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(0).putFloat(1f).array());
        assertThrows(IOException.class, () -> new MappedFvecsRandomAccessVectorValues(zeroDim));

        Path truncated = tempFolder.newFile("truncated.fvecs").toPath();
        var buf = ByteBuffer.allocate(Integer.BYTES + 2 * Float.BYTES + 3).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(2).putFloat(1f).putFloat(2f).put((byte) 1).put((byte) 2).put((byte) 3);
        Files.write(truncated, buf.array());
        assertThrows(IOException.class, () -> new MappedFvecsRandomAccessVectorValues(truncated));

        Path ok = tempFolder.newFile("ok.fvecs").toPath();
        writeFvecs(ok, randomVectors(3, 2, 1), null);
        assertThrows(IllegalArgumentException.class, () -> new MappedFvecsRandomAccessVectorValues(ok, 4));
    }

    @Test
    public void detectsCorruptRecordHeaderOnRead() throws IOException {
        float[][] vectors = randomVectors(3, 4, 5);
        Path file = tempFolder.newFile("corrupt.fvecs").toPath();
        writeFvecs(file, vectors, new int[] {4, 9, 4});
        var mapped = new MappedFvecsRandomAccessVectorValues(file);
        assertEquals(3, mapped.size());
        assertVectorEquals(vectors[0], mapped.getVector(0), 0);
        assertVectorEquals(vectors[2], mapped.getVector(2), 0);
        assertThrows(IllegalStateException.class, () -> mapped.getVector(1));
    }

    @Test
    public void writeFvecsRoundTrips() throws IOException {
        float[][] expected = randomVectors(20, 6, 3);
        List<VectorFloat<?>> vectors = new java.util.ArrayList<>();
        for (float[] v : expected) vectors.add(vts.createFloatVector(v));
        Path file = tempFolder.getRoot().toPath().resolve("written.fvecs");
        SiftLoader.writeFvecs(file, new ListRandomAccessVectorValues(vectors, 6));
        var mapped = new MappedFvecsRandomAccessVectorValues(file);
        assertEquals(20, mapped.size());
        for (int i = 0; i < 20; i++) {
            assertVectorEquals(expected[i], mapped.getVector(i), 0);
        }
        // overwriting an existing file replaces it entirely
        SiftLoader.writeFvecs(file, new ListRandomAccessVectorValues(vectors.subList(0, 5), 6));
        assertEquals(5, new MappedFvecsRandomAccessVectorValues(file).size());
    }
}
