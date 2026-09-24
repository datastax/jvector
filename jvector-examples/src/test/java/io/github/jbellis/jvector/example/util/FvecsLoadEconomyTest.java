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
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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

import static org.junit.jupiter.api.Assertions.assertEquals;

/// Compares the previous streaming fvecs loader against the memory-mapped reader plus in-memory
/// cache, both for load time and for post-load access time, and verifies they yield identical
/// vectors. Timings are reported, not asserted, since they depend on the host.
///
/// The file size is controlled by `-Dfvecs.economy.vectors` and `-Dfvecs.economy.dim`; the file is
/// generated under `target/fvecs-economy` and reused across runs when its size matches.
public class FvecsLoadEconomyTest {

    @Test
    public void mappedPathMatchesStreamingLoaderAndReportsTiming() throws IOException {
        int count = Integer.getInteger("fvecs.economy.vectors", 20_000);
        int dimension = Integer.getInteger("fvecs.economy.dim", 256);
        int rounds = Integer.getInteger("fvecs.economy.rounds", 3);
        Path dir = Path.of(System.getProperty("basedir", "jvector-examples"), "target", "fvecs-economy");
        Files.createDirectories(dir);
        Path file = dir.resolve("economy-" + count + "x" + dimension + ".fvecs");
        long expectedBytes = (long) count * (Integer.BYTES + dimension * Float.BYTES);
        if (!Files.exists(file) || Files.size(file) != expectedBytes) {
            writeRandomFvecs(file, count, dimension);
        }
        System.out.printf("fvecs economy: %d vectors x %d dims (%.1f MB), %d rounds%n",
                count, dimension, expectedBytes / (1024.0 * 1024.0), rounds);
        System.out.printf("%-8s %14s %14s %14s %14s %14s%n",
                "round", "stream-load", "map+cache", "map-only", "scan-stream", "scan-cached");

        List<VectorFloat<?>> streamed = null;
        List<VectorFloat<?>> cached = null;
        for (int round = 0; round < rounds; round++) {
            long t0 = System.nanoTime();
            streamed = SiftLoader.readFvecs(file.toString());
            long streamLoad = System.nanoTime() - t0;

            t0 = System.nanoTime();
            var mapped = new MappedFvecsRandomAccessVectorValues(file);
            long mapOnly = System.nanoTime() - t0;
            cached = InMemoryCachedDataSet.readAllVectors(mapped);
            long mapAndCache = System.nanoTime() - t0;

            var streamedRavv = new ListRandomAccessVectorValues(streamed, dimension);
            var cachedRavv = new ListRandomAccessVectorValues(cached, dimension);
            long scanStream = timeScan(streamedRavv);
            long scanCached = timeScan(cachedRavv);

            System.out.printf("%-8d %12.1fms %12.1fms %12.1fms %12.1fms %12.1fms%n",
                    round, streamLoad / 1e6, mapAndCache / 1e6, mapOnly / 1e6, scanStream / 1e6, scanCached / 1e6);
        }

        assertEquals(streamed.size(), cached.size());
        for (int i = 0; i < count; i++) {
            VectorFloat<?> a = streamed.get(i);
            VectorFloat<?> b = cached.get(i);
            for (int d = 0; d < dimension; d++) {
                assertEquals(a.get(d), b.get(d), 0f, "vector " + i + " component " + d);
            }
        }
    }

    private static long timeScan(RandomAccessVectorValues ravv) {
        long t0 = System.nanoTime();
        float sink = 0;
        for (int i = 0; i < ravv.size(); i++) {
            VectorFloat<?> v = ravv.getVector(i);
            sink += v.get(0) + v.get(v.length() - 1);
        }
        long elapsed = System.nanoTime() - t0;
        if (Float.isNaN(sink)) System.out.println("unexpected NaN");
        return elapsed;
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
        VectorizationProvider.getInstance(); // keep the provider warm for the timed runs
    }
}
