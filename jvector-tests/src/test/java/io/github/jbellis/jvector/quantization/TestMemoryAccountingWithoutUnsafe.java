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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/// Retained-memory accounting must not call the `sun.misc.Unsafe` memory-access methods that
/// JEP 471 terminally deprecated. JDK 24 prints a warning on their first use (JEP 498), and
/// `--sun-misc-unsafe-memory-access=deny`, the planned future default, makes them throw.
/// The probe reads its index through [SimpleReader]; the mapped readers' `close()` still calls
/// `Unsafe.invokeCleaner`, a pre-existing use outside this change.
///
/// Each check runs [Probe] in a fresh JVM so that the very first use of the estimator is observed:
/// it creates byte sequences, mutable and immutable PQ storage, and a loaded fused-PQ graph, and
/// reads every retained-memory figure.
public class TestMemoryAccountingWithoutUnsafe {
    private static final int JAVA = Runtime.version().feature();

    @Test
    public void accountingWorksWhenUnsafeMemoryAccessIsDenied() throws Exception {
        Assume.assumeTrue("--sun-misc-unsafe-memory-access exists from JDK 23", JAVA >= 23);
        String output = probe("--sun-misc-unsafe-memory-access=deny");
        assertTrue(output, output.contains("PASS"));
    }

    @Test
    public void accountingEmitsNoTerminalDeprecationWarning() throws Exception {
        Assume.assumeTrue("JDK 24 warns on the first Unsafe memory access", JAVA >= 24);
        String output = probe();
        assertTrue(output, output.contains("PASS"));
        assertFalse("Accounting used a terminally deprecated sun.misc.Unsafe method:\n" + output,
                output.contains("sun.misc.Unsafe"));
    }

    /// Runs [Probe] in a child JVM with the given extra options and returns its combined output.
    /// The child must exit normally; its output is the assertion message otherwise.
    private static String probe(String... options) throws Exception {
        var tmp = Files.createDirectories(Path.of("target", "tmp", "unsafe-free-accounting"));
        var command = new ArrayList<String>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        if (JAVA >= 20) command.add("--add-modules=jdk.incubator.vector");
        if (JAVA >= 22) command.add("--enable-native-access=ALL-UNNAMED");
        String provider = System.getProperty("jvector.vectorization_provider");
        if (provider != null) command.add("-Djvector.vectorization_provider=" + provider);
        command.add("-Djava.io.tmpdir=" + tmp.toAbsolutePath());
        command.addAll(Arrays.asList(options));
        command.add("-cp");
        command.add(System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
        command.add(Probe.class.getName());
        var process = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            assertTrue("Probe timed out: " + command, process.waitFor(120, TimeUnit.SECONDS));
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals("Probe failed:\n" + output, 0, process.exitValue());
            return output;
        } finally {
            if (process.isAlive()) process.destroyForcibly().waitFor();
        }
    }

    /// Child entry point: exercises every built-in accounting path once and prints `PASS`.
    public static class Probe {
        public static void main(String[] args) throws Exception {
            var types = VectorizationProvider.getInstance().getVectorTypeSupport();
            var code = types.createByteSequence(48);
            long bytes = code.ramBytesUsed() + code.slice(1, 17).ramBytesUsed();

            // Fixed codebooks: two subspaces of four dimensions, no training.
            var pq = new ProductQuantization(new VectorFloat<?>[]{
                    types.createFloatVector(256 * 4), types.createFloatVector(256 * 4)},
                    256, new int[][]{{4, 0}, {4, 4}}, null, KMeansPlusPlusClusterer.UNWEIGHTED);
            var mutable = new MutablePQVectors(pq);
            mutable.setZero(2048);
            bytes += mutable.ramBytesUsed();
            var chunk = types.createByteSequence(2 * 1024);
            bytes += new ImmutablePQVectors(pq, new ByteSequence<?>[]{chunk}, 1024, 1024).ramBytesUsed();

            var graph = new TestUtil.RandomlyConnectedGraphIndex(37, 8, new Random(9182));
            var sourceCode = types.createByteSequence(2);
            var vector = types.createFloatVector(8);
            var path = Files.createTempFile("unsafe-free-accounting-", ".index");
            try {
                var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                try (var graphView = graph.getView();
                     var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                             .with(new FusedPQ(8, pq)).with(new InlineVectors(8)).build()) {
                    suppliers.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(graphView, ignored -> sourceCode, node));
                    suppliers.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(vector));
                    writer.write(suppliers);
                }
                // A plain file reader: the mapped reader's close() uses Unsafe.invokeCleaner, which
                // is terminally deprecated too but predates this change and is out of scope here.
                try (var reader = new SimpleReader.Supplier(path);
                     var loaded = OnDiskGraphIndex.load(reader, 0)) {
                    bytes += loaded.ramBytesUsed();
                }
            } finally {
                Files.deleteIfExists(path);
            }
            if (bytes <= 0) throw new AssertionError("Retained bytes must be positive: " + bytes);
            System.out.println("PASS retained=" + bytes);
        }
    }
}
