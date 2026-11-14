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

import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.*;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Native default dispatch must preserve independent, query-owned prepared state. */
class NativeASHSingleScoringTest {
    private static final class Values implements RandomAccessVectorValues {
        final VectorFloat<?>[] vectors;
        Values(int count, int dim) {
            var types = VectorizationProvider.getInstance().getVectorTypeSupport();
            var random = new Random(51237);
            vectors = new VectorFloat<?>[count];
            for (int i = 0; i < count; i++) {
                float[] value = new float[dim];
                for (int j = 0; j < dim; j++) value[j] = random.nextFloat() - .5f;
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

    @Test void defaultSingleScorersUseNativeWithoutSilentFallback() throws Exception {
        boolean available = LibraryLoader.loadJvector();
        if (Boolean.getBoolean("jvector.test.requireNative")) assertTrue(available, "Native library required");
        else assumeTrue(available, "Native library unavailable");

        var output = java.nio.file.Files.createTempFile("ash-default-native-", ".log");
        Process child = null;
        try {
            child = new ProcessBuilder(java.nio.file.Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                    "-Djvector.vectorization_provider=native", "-cp", System.getProperty("java.class.path"),
                    DefaultSingleProbe.class.getName()).redirectErrorStream(true).redirectOutput(output.toFile()).start();
            assertTrue(child.waitFor(30, java.util.concurrent.TimeUnit.SECONDS));
            assertEquals(0, child.exitValue(), java.nio.file.Files.readString(output));
        } finally {
            if (child != null && child.isAlive()) child.destroyForcibly().waitFor();
            java.nio.file.Files.deleteIfExists(output);
        }
    }

    public static final class DefaultSingleProbe {
        public static void main(String[] args) throws Exception { verifyDefaultNativeSingle(); }
    }

    private static void verifyDefaultNativeSingle() throws Exception {
        assertEquals("native", System.getProperty("jvector.vectorization_provider"));
        for (int bits : new int[] {2, 4}) {
            var values = new Values(40, 64);
            var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + 33 * bits, 1, bits);
            var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
            var expected = new float[40];
            encoded.blockScorerFor(values.getVector(0), VectorSimilarityFunction.DOT_PRODUCT)
                    .scoreRange(0, 40, expected);
            var single = encoded.scoreFunctionFor(values.getVector(0), VectorSimilarityFunction.DOT_PRODUCT);
            var prepared = encoded.precomputedScoreFunctionFor(values.getVector(0), VectorSimilarityFunction.DOT_PRODUCT);
            encoded.scoreFunctionFor(values.getVector(7), VectorSimilarityFunction.DOT_PRODUCT);
            for (int i = 0; i < 40; i++) {
                assertEquals(expected[i], single.similarityTo(i), 2e-5f);
                assertEquals(expected[i], prepared.similarityTo(i), 2e-5f);
            }
        }
        // A regression to the old direct b=2 dispatch must fail rather than silently use Java.
        assertThrows(IllegalStateException.class, () -> new NativeVectorUtilSupport()
                .ashProjectionDot(new float[2], new byte[1], 2, 2));
    }

}
