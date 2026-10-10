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

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

public class VectorPrefetchBenchmarkTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void sampleIsRepeatableUniqueAndOrdered() {
        int[] selected = VectorPrefetchBenchmark.sample(1000, 128, 42);
        assertArrayEquals(selected, VectorPrefetchBenchmark.sample(1000, 128, 42));
        assertFalse(Arrays.equals(selected, VectorPrefetchBenchmark.sample(1000, 128, 43)));
        var seen = new HashSet<Integer>();
        for (int ordinal : selected) { assertTrue(ordinal >= 0 && ordinal < 1000); assertTrue(seen.add(ordinal)); }
        assertEquals(128, seen.size());
        assertEquals(1000, Arrays.stream(VectorPrefetchBenchmark.sample(1000, 1000, 42)).distinct().count());
    }

    @Test public void sampleRejectsImpossibleCounts() {
        assertThrows(IllegalArgumentException.class, () -> VectorPrefetchBenchmark.sample(100, 101, 42));
        assertThrows(IllegalArgumentException.class, () -> VectorPrefetchBenchmark.sample(100, 0, 42));
    }

    @Test public void demandCopiesDoNotOverwriteEachOther() throws Exception {
        Path file = input();
        var preload = SiftLoader.readFvecs(file.toString());
        try (var source = new DemandFvecVectorValues(file); var copy = source.copy()) {
            var first = source.getVector(5);
            var other = copy.getVector(7);
            assertEquals(preload.get(5), first);
            assertEquals(preload.get(7), other);
            assertThrows(IndexOutOfBoundsException.class, () -> source.getVector(source.size()));
        }
    }

    @Test public void fullScanAndSampleProduceSameNVQAcrossAllSources() throws Exception {
        Path file = input();
        var preload = new ListRandomAccessVectorValues(SiftLoader.readFvecs(file.toString()), 16);
        var pool = new ForkJoinPool(2);
        try (var demand = new DemandFvecVectorValues(file); var prefetch = FvecFileVectorValues.open(file)) {
            var sources = java.util.List.of(preload, demand, prefetch);
            for (boolean sampled : new boolean[]{false, true}) {
                var ordinals = VectorPrefetchBenchmark.sample(preload.size(), 32, 42);
                io.github.jbellis.jvector.quantization.NVQVectors expected = null;
                for (var source : sources) {
                    var vectors = sampled ? new ListRandomAccessVectorValues(VectorAccess.copySelected(source, ordinals, pool), 16) : source;
                    var encoded = NVQuantization.compute(vectors, 1).encodeAll(vectors, pool);
                    if (expected == null) expected = encoded;
                    else assertEquals(expected, encoded);
                    var scorer = encoded.scoreFunctionFor(preload.getVector(100), VectorSimilarityFunction.DOT_PRODUCT);
                    for (int i = 0; i < encoded.count(); i++) assertTrue(Float.isFinite(scorer.similarityTo(i)));
                }
            }
        } finally { pool.shutdown(); }
    }

    @Test public void coldCacheEvictsAndVerifiesOnlyInputFile() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        ColdFileCache.evictAndVerify(input());
    }

    @Test public void customLoaderAndNonCompressorRepresentationAreSelectable() throws Exception {
        Path file = input();
        var args = new String[]{"--workload", "sample", "--file", file.toString(), "--samples", "32",
                "--mode", CustomLoader.class.getName(), "--quantizer", CustomRepresentation.class.getName(),
                "--quantizer-options", "example=accepted", "--loader-options", "example=accepted",
                "--cache", "uncontrolled", "--progress-seconds", "0", "--child"};
        CustomLoader.closed = false;
        VectorPrefetchBenchmark.main(args);
        assertTrue(CustomLoader.closed);
    }

    @Test public void builtInPQAndNVQAcceptNamedParameters() throws Exception {
        Path file = input();
        for (var quantizer : new String[]{"pq", "nvq"}) {
            VectorPrefetchBenchmark.main(new String[]{"--workload", "sample", "--file", file.toString(),
                    "--samples", "32", "--mode", "prefetch", "--quantizer", quantizer,
                    "--quantizer-options", quantizer.equals("pq") ? "subspaces=2,clusters=16,center=false,anisotropic=-1"
                            : "subvectors=2,learn=false",
                    "--loader-options", "ioThreads=2,batchVectors=8,readAhead=1,maxBufferBytes=1024",
                    "--cache", "uncontrolled", "--progress-seconds", "0", "--child"});
        }
    }

    public static final class CustomLoader implements VectorPrefetchBenchmark.Loader {
        static boolean closed;
        public VectorPrefetchBenchmark.Input open(Path file, java.util.Map<String, String> parameters) throws Exception {
            assertEquals("accepted", parameters.get("example"));
            var source = new DemandFvecVectorValues(file);
            return new VectorPrefetchBenchmark.Input() {
                public io.github.jbellis.jvector.graph.RandomAccessVectorValues values() { return source; }
                public void close() throws Exception { source.close(); closed = true; }
            };
        }
    }

    // Exercise a representation that has a score function but does not implement VectorCompressor.
    public static final class CustomRepresentation implements VectorPrefetchBenchmark.Quantizer {
        public VectorPrefetchBenchmark.Encoder train(io.github.jbellis.jvector.graph.RandomAccessVectorValues source,
                java.util.Map<String, String> parameters, ForkJoinPool executor) {
            assertEquals("accepted", parameters.get("example"));
            return (values, pool) -> {
                var ordinals = java.util.stream.IntStream.range(0, values.size()).toArray();
                var vectors = VectorAccess.copySelected(values, ordinals, pool);
                return new VectorPrefetchBenchmark.Encoded() {
                    public int count() { return vectors.size(); }
                    public java.util.function.IntToDoubleFunction scorer(
                            io.github.jbellis.jvector.vector.types.VectorFloat<?> query, VectorSimilarityFunction metric) {
                        return ordinal -> metric.compare(query, vectors.get(ordinal));
                    }
                };
            };
        }
    }

    private Path input() throws Exception {
        Path file = temporary.newFile().toPath();
        var data = ByteBuffer.allocate(256 * 17 * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 256; i++) {
            data.putInt(16);
            for (int j = 0; j < 16; j++) data.putFloat((float) Math.sin(i * 0.1 + j * 0.7));
        }
        data.flip();
        try (var out = FileChannel.open(file, StandardOpenOption.WRITE)) {
            while (data.hasRemaining()) out.write(data);
            out.force(true); // Make test pages clean before asking the kernel to evict them.
        }
        return file;
    }
}
