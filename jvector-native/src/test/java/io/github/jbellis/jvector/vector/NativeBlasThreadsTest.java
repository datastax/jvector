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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The native provider sets BLAS threads, and ASH training bounds them by its executor. */
class NativeBlasThreadsTest {
    // A fake process-wide library: one count shared by every thread, like openblas_set_num_threads.
    private static int processCount;
    private static void fakeSet(int threads) { processCount = threads; }
    private static int fakeGet() { return processCount; }

    private static NativeBlasThreads.Control processWide() throws Exception {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var set = lookup.findStatic(NativeBlasThreadsTest.class, "fakeSet",
                java.lang.invoke.MethodType.methodType(void.class, int.class));
        var get = lookup.findStatic(NativeBlasThreadsTest.class, "fakeGet",
                java.lang.invoke.MethodType.methodType(int.class));
        return new NativeBlasThreads.Control("fake process-wide", set, get, true);
    }

    @Test
    void overlappingProcessWideScopesHoldTheCountUntilTheLastCloses() throws Exception {
        var control = processWide();
        processCount = 48;
        int a = control.set(1);                 // scope A opens on one thread
        int b = control.set(1);                 // scope B opens on another before A closes
        assertEquals(1, processCount);
        control.set(a);                         // A closes while B is still open
        assertEquals(1, processCount, "B must keep its single thread");
        control.set(b);                         // B closes last
        assertEquals(48, processCount, "The process count must be restored");
        control.set(b);                         // a repeated restore changes nothing
        assertEquals(48, processCount);
    }

    @Test
    void aConflictingProcessWideScopeDoesNotChangeTheHeldCount() throws Exception {
        var control = processWide();
        processCount = 48;
        int a = control.set(1);
        int b = control.set(4);                 // cannot change a count another thread is using
        assertEquals(1, processCount);
        control.set(b);
        control.set(a);
        assertEquals(48, processCount);
    }

    private static boolean supported() {
        return !"none".equals(NativeBlasThreads.description());
    }

    @Test
    void setterReturnsTheValueToRestore() {
        assumeTrue(supported(), "No native BLAS thread control on this host");
        int previous = NativeBlasThreads.set(3);
        try {
            assertEquals(3, NativeBlasThreads.set(previous), NativeBlasThreads.description());
        } finally {
            NativeBlasThreads.set(previous);
        }
    }

    @Test
    void trainingUsesOneBlasThreadWithoutAnExecutorAndTheExecutorSizeWithOne() throws Exception {
        assumeTrue(supported() && Files.isDirectory(Path.of("/proc/self/task")),
                "Needs native BLAS thread control and Linux thread accounting");
        int cores = Runtime.getRuntime().availableProcessors();
        assumeTrue(cores >= 16, "Needs enough CPUs to tell one BLAS thread from many");
        int single = nativeThreadsStarted("single");
        int pooled = nativeThreadsStarted("pooled");
        assertTrue(single <= 1, "Training without an executor started " + single + " native threads");
        assertTrue(pooled >= single + 8, "Training with a " + cores + "-thread executor started only "
                + pooled + " native threads; the BLAS limit may not be applied");
    }

    /** Native-library threads a fresh JVM gains while training ASH with ITQ. */
    private static int nativeThreadsStarted(String mode) throws Exception {
        var output = Files.createTempFile("blas-threads-", ".log");
        Process child = null;
        try {
            var builder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                    "-Djvector.vectorization_provider=native", "-cp", System.getProperty("java.class.path"),
                    TrainingProbe.class.getName(), mode)
                    .redirectErrorStream(true).redirectOutput(output.toFile());
            for (var name : new String[] {"MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"}) {
                builder.environment().remove(name);
            }
            child = builder.start();
            assertTrue(child.waitFor(120, TimeUnit.SECONDS), "Training probe timed out");
            String log = Files.readString(output);
            assertEquals(0, child.exitValue(), log);
            var match = Pattern.compile("native-threads-started=(\\d+)").matcher(log);
            assertTrue(match.find(), log);
            return Integer.parseInt(match.group(1));
        } finally {
            if (child != null && child.isAlive()) child.destroyForcibly().waitFor();
            Files.deleteIfExists(output);
        }
    }

    public static final class TrainingProbe {
        public static void main(String[] args) throws Exception {
            var values = new Values(12_000, 512);
            int before = nativeThreads();
            if ("pooled".equals(args[0])) {
                var pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
                AsymmetricHashing.initialize(values, AsymmetricHashing.ITQ, AsymmetricHashing.HEADER_BITS + 512, 1, 2, pool);
                pool.shutdown();
            } else {
                AsymmetricHashing.initialize(values, AsymmetricHashing.ITQ, AsymmetricHashing.HEADER_BITS + 512, 1, 2);
            }
            System.out.println("native-threads-started=" + Math.max(0, nativeThreads() - before));
        }

        /**
         * Threads the JVM did not name. The JVM names its own and Java threads, so these are the
         * threads native libraries such as MKL's OpenMP runtime start; they keep the process name.
         */
        private static int nativeThreads() throws Exception {
            int count = 0;
            try (var tasks = Files.list(Path.of("/proc/self/task"))) {
                for (var task : (Iterable<Path>) tasks::iterator) {
                    try {
                        if ("java".equals(Files.readString(task.resolve("comm")).trim())) count++;
                    } catch (java.nio.file.NoSuchFileException exited) {
                        // The thread ended while listing.
                    }
                }
            }
            return count;
        }
    }

    private static final class Values implements RandomAccessVectorValues {
        private final VectorFloat<?>[] vectors;

        Values(int count, int dim) {
            var types = VectorizationProvider.getInstance().getVectorTypeSupport();
            var random = new Random(7);
            vectors = new VectorFloat<?>[count];
            for (int i = 0; i < count; i++) {
                float[] value = new float[dim];
                for (int j = 0; j < dim; j++) value[j] = (float) random.nextGaussian();
                vectors[i] = types.createFloatVector(value);
                VectorUtil.l2normalize(vectors[i]);
            }
        }

        public int size() { return vectors.length; }
        public int dimension() { return vectors[0].length(); }
        public VectorFloat<?> getVector(int i) { return vectors[i]; }
        public boolean isValueShared() { return false; }
        public RandomAccessVectorValues copy() { return this; }
    }
}
