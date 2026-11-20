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

package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Finds the dimension at which native SIMD kernels (cosine/dot/L2) beat Panama for the
 * graph-search access pattern: a long-lived query vector scored against a freshly allocated
 * vector per call, as {@code OnDiskGraphIndex.getVector(node)} produces.
 *
 * <p>{@code jvector.native_vector_threshold} is a static final, so each path runs in its own fork:
 * {@link Native} forces the native kernel (threshold 0), {@link Panama} forces the Panama
 * fallback (threshold MAX_VALUE). Compare the two at each {@code dimension} to find the crossover.
 *
 * <pre>
 * java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
 *   -jar benchmarks-jmh/target/benchmarks-jmh-*.jar NativeThresholdBenchmark -rf csv -rff threshold.csv
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public abstract class NativeThresholdBenchmark {
    private static final String COMMON_JVM_ARGS_1 = "--enable-native-access=ALL-UNNAMED";
    private static final String COMMON_JVM_ARGS_2 = "--add-modules=jdk.incubator.vector";
    private static final String NATIVE_ON = "-Djvector.experimental.enable_native_vectorization=true";

    /** Pool size large enough that base vectors are not all L1-resident, like real traversal. */
    private static final int POOL = 1024;

    @Param({"32", "64", "96", "128", "192", "256", "384", "512", "768", "1024", "1536"})
    public int dimension;

    private VectorTypeSupport vts;
    private VectorUtilSupport util;
    private VectorFloat<?> query;
    private float[][] pool;
    private int next;

    @Setup(Level.Trial)
    public void setup() {
        var provider = VectorizationProvider.getInstance();
        String name = provider.getClass().getSimpleName();
        if (!name.contains("Native")) {
            throw new IllegalStateException("Expected native vectorization provider, got " + name);
        }
        vts = provider.getVectorTypeSupport();
        util = provider.getVectorUtilSupport();
        var rnd = new Random(42);
        query = vts.createFloatVector(randomArray(rnd));
        pool = new float[POOL][];
        for (int i = 0; i < POOL; i++) {
            pool[i] = randomArray(rnd);
        }
    }

    private float[] randomArray(Random rnd) {
        float[] a = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            a[i] = rnd.nextFloat() - 0.5f;
        }
        return a;
    }

    /** A new vector object per call, so any lazy MemorySegment creation is paid every time. */
    private VectorFloat<?> freshBase() {
        int i = next;
        next = (i + 1) & (POOL - 1);
        return vts.createFloatVector(pool[i]);
    }

    @Benchmark
    public float dotProduct() {
        return util.dotProduct(query, freshBase());
    }

    @Benchmark
    public float cosine() {
        return util.cosine(query, freshBase());
    }

    @Benchmark
    public float squareDistance() {
        return util.squareDistance(query, freshBase());
    }

    @Fork(value = 2, jvmArgsAppend = {COMMON_JVM_ARGS_1, COMMON_JVM_ARGS_2, NATIVE_ON,
            "-Djvector.native_vector_threshold=0"})
    public static class Native extends NativeThresholdBenchmark {}

    @Fork(value = 2, jvmArgsAppend = {COMMON_JVM_ARGS_1, COMMON_JVM_ARGS_2, NATIVE_ON,
            "-Djvector.native_vector_threshold=" + Integer.MAX_VALUE})
    public static class Panama extends NativeThresholdBenchmark {}
}
