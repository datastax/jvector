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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Times PQ code-pair scoring as graph construction uses it: {@code diversityFunctionFor(a).similarityTo(b)}
 * for random pairs of saved (encodeAll) or in-progress (MutablePQVectors) codes. Pick the provider with
 * {@code -jvmArgsAppend -Djvector.vectorization_provider=native} (or panama).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Threads(1)
public class PQDiversityBenchmark {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int PAIRS = 4096;

    /** Dimension and subspace count, as "dimension/M". */
    @Param({"64/64", "64/32", "64/8",
            "128/128", "128/64", "128/32", "128/16",
            "100/32", "384/192", "384/128", "384/96", "384/48",
            "768/256", "768/192", "768/96", "768/48",
            "1024/128", "1024/64", "1024/16",
            "1536/192", "1536/96",
            "3072/384", "3072/192"})
    private String shape;

    @Param({"saved", "inprogress"})
    private String codes;

    @Param({"DOT_PRODUCT"})
    private VectorSimilarityFunction vsf;

    @Param({"10000"})
    private int vectorCount;

    private PQVectors pqVectors;
    private int[] left, right;

    @Setup
    public void setup() {
        String[] parts = shape.split("/");
        int dimension = Integer.parseInt(parts[0]), m = Integer.parseInt(parts[1]);
        var random = new Random(42);
        List<VectorFloat<?>> vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) v.set(d, (float) random.nextGaussian());
            vectors.add(v);
        }
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var pq = ProductQuantization.compute(ravv, m, 256, false);
        if ("saved".equals(codes)) {
            pqVectors = pq.encodeAll(ravv, PhysicalCoreExecutor.pool());
        } else {
            var mutable = new MutablePQVectors(pq);
            for (int i = 0; i < vectorCount; i++) mutable.encodeAndSet(i, vectors.get(i));
            pqVectors = mutable;
        }
        left = new int[PAIRS];
        right = new int[PAIRS];
        for (int i = 0; i < PAIRS; i++) {
            left[i] = random.nextInt(vectorCount);
            right[i] = random.nextInt(vectorCount);
        }
        // Build the codebook layout or pair table before timing.
        pqVectors.diversityFunctionFor(0, vsf).similarityTo(1);
    }

    @Benchmark
    @OperationsPerInvocation(PAIRS)
    public float pairScore() {
        float sum = 0;
        for (int i = 0; i < PAIRS; i++) {
            sum += pqVectors.diversityFunctionFor(left[i], vsf).similarityTo(right[i]);
        }
        return sum;
    }
}
