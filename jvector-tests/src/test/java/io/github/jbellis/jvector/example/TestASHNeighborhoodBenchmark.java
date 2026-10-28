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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import static org.junit.Assert.*;

/** Fixture replay, full-scan scheduling, and failure-boundary checks for DistancesASH. */
public class TestASHNeighborhoodBenchmark {
    @Test
    public void serializedNeighborhoodsMatchOracleAcrossWidthsAndTails() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> vectors = new ArrayList<>();
        Random random = new Random(5293);
        for (int i = 0; i < 35; i++) {
            float[] data = new float[64];
            for (int d = 0; d < data.length; d++)
                data[d] = random.nextFloat() - .5f;
            var vector = types.createFloatVector(data);
            VectorUtil.l2normalize(vector);
            vectors.add(vector);
        }
        var values = new ListRandomAccessVectorValues(vectors, 64);
        for (int bits : new int[] {1, 2, 4}) {
            var ash = AsymmetricHashing.initialize(
                    values, AsymmetricHashing.RANDOM, AsymmetricHashing.HEADER_BITS + 33 * bits, 1, bits);
            ASHVectors encoded = (ASHVectors) ash.encodeAll(values);
            for (int capacity : new int[] {1, 8, 16, 23, 32, 64}) {
                int blockSize = FusedASHLayout.chooseBlockSize(capacity);
                byte[][] packed = new byte[(vectors.size() - 1) / capacity + 1]
                                          [FusedASHLayout.featureSize(capacity, 33, bits, blockSize)];
                for (int i = 0; i < vectors.size(); i++) {
                    int neighbor = i % capacity;
                    FusedASHLayout.packForFused(packed[i / capacity],
                            FusedASHLayout.blockOffset(neighbor / blockSize, 33, bits, blockSize), neighbor % blockSize,
                            encoded.get(i), ash, blockSize);
                }
                var fixture = new ASHNeighborhoodFixture(
                        ash, vectors.subList(0, 2), packed, capacity, blockSize, vectors.size());
                ASHNeighborhoodValidator.validate(fixture);
                if (bits == 2 && capacity == 8) {
                    var query = fixture.queries.get(0);
                    var reference = new ASHNeighborhoodScorer(ash, query, capacity, blockSize);
                    float[] output = new float[capacity];
                    double checksum = 0;
                    for (byte[] neighborhood : packed) {
                        reference.score(neighborhood, output);
                        for (float score : output)
                            checksum += score;
                    }
                    int[] order = ASHNeighborhoodBenchmark.accessOrder(packed.length, "shuffled");
                    int[] duplicate = order.clone();
                    duplicate[0] = duplicate[1];
                    assertThrows(
                            IllegalArgumentException.class, () -> new ASHNeighborhoodPipeline(fixture, duplicate, 1));
                    assertThrows(
                            IllegalArgumentException.class, () -> new ASHNeighborhoodPipeline(fixture, new int[0], 1));
                    for (int threads : new int[] {1, 2, 4}) {
                        try (var parallel = new ASHNeighborhoodPipeline(fixture, order, threads)) {
                            for (int sweep = 0; sweep < 3; sweep++) {
                                // Both scans can be in flight; each must retain its own prepared query.
                                var firstScan = parallel.submit(query);
                                var nextQuery = fixture.queries.get(1);
                                var nextScorer = new ASHNeighborhoodScorer(ash, nextQuery, capacity, blockSize);
                                double nextChecksum = 0;
                                for (byte[] neighborhood : packed) {
                                    nextScorer.score(neighborhood, output);
                                    for (float score : output)
                                        nextChecksum += score;
                                }
                                var nextScan = parallel.submit(nextQuery);
                                assertEquals(checksum, firstScan.await(), 1e-9);
                                assertEquals(nextChecksum, nextScan.await(), 1e-9);
                            }
                        }
                    }
                    var closed = new ASHNeighborhoodPipeline(fixture, order, 1);
                    closed.close();
                    assertThrows(IllegalStateException.class, () -> closed.submit(query));
                    try (var pipeline = new ASHNeighborhoodPipeline(fixture, order, 1)) {
                        var submitted = pipeline.submit(query);
                        Thread.currentThread().interrupt();
                        try {
                            assertThrows(IllegalStateException.class, submitted::await);
                            assertTrue(Thread.currentThread().isInterrupted());
                        } finally {
                            Thread.interrupted(); // Do not leak interruption into the JUnit runner.
                        }
                    }
                    // A failed worker must report its exception and still shut down cleanly.
                    int landmark = FusedASHLayout.landmarkOffset(0, 33, bits, blockSize);
                    byte saved = packed[0][landmark];
                    packed[0][landmark] = (byte) 255;
                    try (var parallel = new ASHNeighborhoodPipeline(fixture, order, 2)) {
                        assertThrows(IllegalStateException.class, () -> parallel.submit(query).await());
                    } finally {
                        packed[0][landmark] = saved;
                    }
                }
                var path = Files.createTempFile("ash-neighborhood-", ".ashn");
                Files.delete(path); // The writer deliberately refuses to overwrite fixtures.
                try {
                    ASHNeighborhoodFixture.write(path, fixture);
                    var loaded = ASHNeighborhoodFixture.read(path);
                    ASHNeighborhoodValidator.validate(loaded);
                    assertEquals(fixture.capacity, loaded.capacity);
                    for (int i = 0; i < packed.length; i++)
                        assertArrayEquals(packed[i], loaded.neighborhoods[i]);
                    var scorer = new ASHNeighborhoodScorer(loaded.ash, loaded.queries.get(0), capacity, blockSize);
                    if (VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshNeighborhoodScoring()
                            && !"scalar".equals(System.getProperty("jvector.ash.blockKernel"))) {
                        assertTrue(
                                scorer.toString(), scorer.toString().contains("native Highway whole-neighborhood FFM"));
                    }
                    float[] out = new float[capacity + 1];
                    out[capacity] = Float.NaN;
                    scorer.score(packed[0], out);
                    assertTrue(Float.isNaN(out[capacity]));
                    assertThrows(IndexOutOfBoundsException.class, () -> scorer.score(new byte[0], out));
                    assertThrows(IndexOutOfBoundsException.class, () -> scorer.score(packed[0], new float[0]));
                    String hash = ASHNeighborhoodFixture.sha256(path);
                    assertThrows(IOException.class, () -> ASHNeighborhoodFixture.write(path, fixture));
                    assertEquals(hash, ASHNeighborhoodFixture.sha256(path));
                    assertEquals(64, hash.length());
                    byte[] original = Files.readAllBytes(path);
                    Files.write(path, Arrays.copyOf(original, original.length - 1));
                    assertThrows(IOException.class, () -> ASHNeighborhoodFixture.read(path));
                    Files.write(path, Arrays.copyOf(original, original.length + 1));
                    assertThrows(IOException.class, () -> ASHNeighborhoodFixture.read(path));
                } finally {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    @Test
    public void accessOrdersAreRepeatablePermutations() {
        int[] sequential = ASHNeighborhoodBenchmark.accessOrder(37, "sequential");
        int[] shuffled = ASHNeighborhoodBenchmark.accessOrder(37, "shuffled");
        assertArrayEquals(shuffled, ASHNeighborhoodBenchmark.accessOrder(37, "shuffled"));
        assertFalse(Arrays.equals(sequential, shuffled));
        Arrays.sort(shuffled);
        assertArrayEquals(sequential, shuffled);
        assertThrows(IllegalArgumentException.class, () -> ASHNeighborhoodBenchmark.accessOrder(37, "invalid"));
    }

    @Test
    public void rejectsMalformedIntegerSettings() {
        String name = "jvector.bench.scoringPasses";
        String previous = System.getProperty(name);
        try {
            System.setProperty(name, "not-an-integer");
            assertThrows(IllegalArgumentException.class, () -> ASHNeighborhoodBenchmark.positive(name, 5));
        } finally {
            if (previous == null)
                System.clearProperty(name);
            else
                System.setProperty(name, previous);
        }
    }

    @Test
    public void rejectsTruncatedFixturesAndFvecs() throws Exception {
        var path = Files.createTempFile("ash-truncated-", ".bin");
        try {
            Files.write(path, new byte[] {1, 2, 3});
            assertThrows(IOException.class, () -> ASHNeighborhoodFixture.read(path));
            assertThrows(IOException.class, () -> ASHNeighborhoodFixture.readFvecs(path.toString(), 1));
            assertThrows(IllegalArgumentException.class, () -> ASHNeighborhoodFixture.readFvecs(path.toString(), 0));
            var buffer = java.nio.ByteBuffer.allocate(9 * Integer.BYTES + 44);
            buffer.putInt(0x4153484e).putInt(1).putInt(32).putInt(32).putInt(1).putInt(44);
            buffer.putInt(0).putInt(0).putInt(Integer.MAX_VALUE).putInt(0);
            buffer.putInt(Integer.MAX_VALUE).putInt(4).putInt(0).putInt(1);
            Files.write(path, buffer.array());
            assertThrows(IOException.class, () -> ASHNeighborhoodFixture.read(path));
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
