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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.util.FlatFvecLoader;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class TestFlatFvecLoader {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    /** Writes {@code rows} random rows; {@code badRow}, if nonnegative, gets a different dimension header. */
    private Path fvecs(int rows, int dimension, int badRow) throws IOException {
        var random = new Random(rows * 31L + dimension);
        var buffer = ByteBuffer.allocate(rows * (Integer.BYTES + dimension * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            buffer.putInt(row == badRow ? dimension + 1 : dimension);
            for (int d = 0; d < dimension; d++) buffer.putFloat(random.nextFloat() - 0.5f);
        }
        var file = folder.newFile().toPath();
        Files.write(file, buffer.array());
        return file;
    }

    /** Rows land at their file ordinals with their exact values, whatever the thread count or read chunking. */
    @Test
    public void loadsEveryRowAtItsOrdinal() throws IOException {
        for (int dimension : new int[] {1, 7, 384}) {
            for (int rows : new int[] {1, 3, 4099}) {
                var file = fvecs(rows, dimension, -1);
                var expected = SiftLoader.readFvecs(file.toString());
                for (int threads : new int[] {1, 5}) {
                    var pool = new ForkJoinPool(threads);
                    try {
                        var flat = FlatFvecLoader.load(file, pool);
                        assertEquals(rows, flat.size());
                        assertEquals(dimension, flat.dimension());
                        for (int row = 0; row < rows; row++) {
                            for (int d = 0; d < dimension; d++) {
                                assertEquals(expected.get(row).get(d), flat.getVector(row).get(d), 0f);
                            }
                        }
                    } finally {
                        pool.shutdown();
                    }
                }
            }
        }
    }

    @Test
    public void rejectsMalformedFiles() throws IOException {
        assertThrows(UncheckedIOException.class, () -> FlatFvecLoader.load(fvecs(50, 8, 37)));
        var truncated = fvecs(10, 8, -1);
        Files.write(truncated, java.util.Arrays.copyOf(Files.readAllBytes(truncated), (int) Files.size(truncated) - 3));
        assertThrows(UncheckedIOException.class, () -> FlatFvecLoader.load(truncated));
        var empty = folder.newFile().toPath();
        assertThrows(UncheckedIOException.class, () -> FlatFvecLoader.load(empty));
    }

    /** A dataset over a flat store reads it through its RAVV and gives copies through its compatibility list. */
    @Test
    public void datasetServesTheFlatStore() throws IOException {
        var file = fvecs(20, 6, -1);
        var flat = FlatFvecLoader.load(file);
        var queries = SiftLoader.readFvecs(file.toString()).subList(0, 2);
        var dataset = new SimpleDataSet("flat", VectorSimilarityFunction.DOT_PRODUCT, flat, queries,
                List.of(List.of(0), List.of(1)));
        assertEquals(flat, dataset.getBaseRavv());
        assertEquals(6, dataset.getDimension());
        assertEquals(20, dataset.getBaseVectors().size());
        var copy = dataset.getBaseVectors().get(4);
        for (int d = 0; d < 6; d++) assertEquals(flat.getVector(4).get(d), copy.get(d), 0f);
        copy.set(0, 99f);
        assertEquals(flat.getVector(4).get(0), dataset.getBaseVectors().get(4).get(0), 0f); // the store is unchanged
    }
}
