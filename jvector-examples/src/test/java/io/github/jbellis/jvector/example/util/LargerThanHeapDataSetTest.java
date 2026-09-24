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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.LruCachedDataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.MMapCachedDataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/// Reads a whole fvecs dataset through the `mmap` and `lru` wrappers and checks both against a
/// direct mapped scan, without ever holding the base vectors in heap memory.
///
/// Sized by `-Dfvecs.ltm.vectors` and `-Dfvecs.ltm.dim` (the file is shared with
/// {@link FvecsLoadEconomyTest} under `target/fvecs-economy`). To demonstrate larger-than-heap
/// operation, run with a file well above the heap, e.g.
/// `-Dfvecs.ltm.vectors=1000000 -Dfvecs.ltm.dim=1024 -DargLine=-Xmx1g`.
public class LargerThanHeapDataSetTest {

    @Test
    public void mmapAndLruWrappersScanWithoutHeapResidentBaseVectors() throws IOException {
        int count = Integer.getInteger("fvecs.ltm.vectors", 20_000);
        int dimension = Integer.getInteger("fvecs.ltm.dim", 256);
        long capacityBytes = Long.getLong("fvecs.ltm.lruCapacityMb", 64L) * 1024 * 1024;
        Path dir = Path.of(System.getProperty("basedir", "jvector-examples"), "target", "fvecs-economy");
        Files.createDirectories(dir);
        Path file = dir.resolve("economy-" + count + "x" + dimension + ".fvecs");
        long expectedBytes = (long) count * (Integer.BYTES + dimension * Float.BYTES);
        if (!Files.exists(file) || Files.size(file) != expectedBytes) {
            writeRandomFvecs(file, count, dimension);
        }
        long maxHeap = Runtime.getRuntime().maxMemory();
        System.out.printf("larger-than-heap: %.1f MB of vectors, max heap %.1f MB, lru capacity %.1f MB, vector provider %s%n",
                expectedBytes / (1024.0 * 1024.0), maxHeap / (1024.0 * 1024.0), capacityBytes / (1024.0 * 1024.0),
                VectorizationProvider.getInstance().getClass().getSimpleName());

        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var query = vts.createFloatVector(dimension);
        DataSet origin = new SimpleDataSet("ltm", VectorSimilarityFunction.EUCLIDEAN,
                new MappedFvecsRandomAccessVectorValues(file), List.of(query), List.of(List.of(0)));

        long t0 = System.nanoTime();
        double direct = checksum(new MappedFvecsRandomAccessVectorValues(file), 1);
        long directMs = (System.nanoTime() - t0) / 1_000_000;

        DataSet mmap = MMapCachedDataSet.of(origin);
        t0 = System.nanoTime();
        double viaMmap = checksum(mmap.getBaseRavv(), 8);
        long mmapMs = (System.nanoTime() - t0) / 1_000_000;

        var lru = new LruCachedDataSet(origin, LruCachedDataSet.DEFAULT_GRAIN_SIZE, capacityBytes);
        t0 = System.nanoTime();
        double viaLru = checksum(lru.getBaseRavv(), 8);
        long lruMs = (System.nanoTime() - t0) / 1_000_000;

        System.out.printf("direct scan %d ms, mmap wrapper (8 threads) %d ms, lru wrapper (8 threads) %d ms; lru %s%n",
                directMs, mmapMs, lruMs, lru.getBaseRavv().stats());

        assertEquals(direct, viaMmap, 0.0);
        assertEquals(direct, viaLru, 0.0);
        assertTrue(lru.getBaseRavv().residentGrains() <= lru.getBaseRavv().maxGrains() + 8, lru.getBaseRavv().stats());
        assertTrue(expectedBytes > capacityBytes || count < 100_000, "test parameters should exceed the lru capacity");
    }

    /// Order-independent checksum over first, middle and last component of every vector.
    private static double checksum(RandomAccessVectorValues ravv, int threads) {
        int n = ravv.size();
        int chunk = (n + threads - 1) / threads;
        return IntStream.range(0, threads).parallel().mapToDouble(t -> {
            RandomAccessVectorValues local = ravv.copy();
            double sum = 0;
            int end = Math.min(n, (t + 1) * chunk);
            for (int i = t * chunk; i < end; i++) {
                VectorFloat<?> v = local.getVector(i);
                sum += v.get(0) + v.get(v.length() / 2) + v.get(v.length() - 1);
            }
            return sum;
        }).sum();
    }

    private static void writeRandomFvecs(Path file, int count, int dimension) throws IOException {
        var random = new Random(1234);
        var record = ByteBuffer.allocate(Integer.BYTES + dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try (var out = new BufferedOutputStream(Files.newOutputStream(file), 1 << 20)) {
            for (int i = 0; i < count; i++) {
                record.clear();
                record.putInt(dimension);
                for (int d = 0; d < dimension; d++) record.putFloat(random.nextFloat());
                out.write(record.array());
            }
        }
    }
}
