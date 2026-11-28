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

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Exact and NVQ scores, as reranking calls them. PQ scoring is in {@link PanamaKernelBenchmark}. Run with {@code -prof gc} to see bytes allocated per operation, and with
 * {@code -jvmArgsAppend -Djvector.vectorization_provider=panama} (or native) to pick the provider.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Threads(1)
public class PanamaDistanceBenchmark {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Dimension of the exact and NVQ scores. */
    @Param({"25", "100", "768"})
    int dimension;

    private VectorFloat<?>[] exact;
    private ByteSequence<?> nvqCode;
    private int next;

    @Setup
    public void setup() {
        Random random = new Random(42);
        exact = new VectorFloat<?>[64];
        for (int i = 0; i < exact.length; i++) {
            exact[i] = VTS.createFloatVector(dimension);
            for (int j = 0; j < dimension; j++) exact[i].set(j, (float) random.nextGaussian());
        }
        nvqCode = VTS.createByteSequence(dimension);
        for (int i = 0; i < dimension; i++) nvqCode.set(i, (byte) random.nextInt(256));
    }

    @Benchmark
    public float exactDot() {
        int i = next++;
        return VectorUtil.dotProduct(exact[i & 63], exact[(i >>> 6) & 63]);
    }

    @Benchmark
    public float exactEuclidean() {
        int i = next++;
        return VectorUtil.squareL2Distance(exact[i & 63], exact[(i >>> 6) & 63]);
    }

    @Benchmark
    public float exactCosine() {
        int i = next++;
        return VectorUtil.cosine(exact[i & 63], exact[(i >>> 6) & 63]);
    }

    @Benchmark
    public float nvqDot() {
        return VectorUtil.nvqDotProduct8bit(exact[next++ & 63], nvqCode, 1.5f, 0.1f, -2f, 2f);
    }

    @Benchmark
    public float nvqEuclidean() {
        return VectorUtil.nvqSquareL2Distance8bit(exact[next++ & 63], nvqCode, 1.5f, 0.1f, -2f, 2f);
    }
}
