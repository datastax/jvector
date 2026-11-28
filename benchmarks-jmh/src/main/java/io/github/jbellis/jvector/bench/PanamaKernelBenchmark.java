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
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

/**
 * PQ scoring through the calls a build or search makes: the query table (one per score function) and a node score.
 * Exact and NVQ scores are in {@link PanamaDistanceBenchmark}. Run with {@code -prof gc} to see bytes allocated per operation, and with
 * {@code -jvmArgsAppend -Djvector.vectorization_provider=panama} (or native) to pick the provider.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Threads(1)
public class PanamaKernelBenchmark {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Floats per PQ subspace; the dimension is 768, so 96 subspaces at 8. */
    @Param({"2", "4", "8", "16", "32"})
    int subspaceSize;

    private PQVectors pqVectors;
    private VectorFloat<?>[] queries;
    private int next;

    @Setup
    public void setup() {
        Random random = new Random(42);
        int pqDim = 768, count = 4096;
        List<VectorFloat<?>> data = new ArrayList<>(count);
        for (int i = 0; i < count; i++) data.add(randomVector(random, pqDim));
        var ravv = new ListRandomAccessVectorValues(data, pqDim);
        var pq = ProductQuantization.compute(ravv, pqDim / subspaceSize, 256, true);
        pqVectors = pq.encodeAll(ravv, ForkJoinPool.commonPool());
        queries = new VectorFloat<?>[64];
        for (int i = 0; i < queries.length; i++) queries[i] = randomVector(random, pqDim);
    }

    private static VectorFloat<?> randomVector(Random random, int dim) {
        var v = VTS.createFloatVector(dim);
        for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian());
        return v;
    }

    private VectorFloat<?> nextQuery() {
        return queries[next++ & 63];
    }

    /** Builds the PQ query table and scores one node, as a graph insertion does per search. */
    @Benchmark
    public float pqTableDot() {
        return pqVectors.precomputedScoreFunctionFor(nextQuery(), VectorSimilarityFunction.DOT_PRODUCT).similarityTo(next & 4095);
    }

    @Benchmark
    public float pqTableEuclidean() {
        return pqVectors.precomputedScoreFunctionFor(nextQuery(), VectorSimilarityFunction.EUCLIDEAN).similarityTo(next & 4095);
    }

    @Benchmark
    public float pqTableCosine() {
        return pqVectors.precomputedScoreFunctionFor(nextQuery(), VectorSimilarityFunction.COSINE).similarityTo(next & 4095);
    }
}
