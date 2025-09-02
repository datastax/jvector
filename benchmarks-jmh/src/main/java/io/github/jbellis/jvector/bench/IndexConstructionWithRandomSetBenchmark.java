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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;


@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=false"})
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@Threads(1)
public class IndexConstructionWithRandomSetBenchmark {
    private static final Logger log = LoggerFactory.getLogger(IndexConstructionWithRandomSetBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private RandomAccessVectorValues ravv;
    private BuildScoreProvider buildScoreProvider;
    private int M = 32; // graph degree
    private int beamWidth = 100;
    @Param({"768", "1536"})
    private int originalDimension;
    @Param({/*"10000",*/ "100000"/*, "1000000"*/})
    int numBaseVectors;
    @Param({"0", "16"})
    private int numberOfPQSubspaces;

    @Setup(Level.Trial)
    public void setup() throws IOException {

        final var baseVectors = new ArrayList<VectorFloat<?>>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        if (numberOfPQSubspaces > 0) {
            log.info("Using PQ build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            final ProductQuantization pq = ProductQuantization.compute(ravv,
                    numberOfPQSubspaces,
                    256,
                    true);
            final PQVectors pqVectors = (PQVectors) pq.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);
        } else {
            log.info("Using Exact build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            // score provider using the raw, in-memory vectors
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        }

    }

    @TearDown(Level.Invocation)
    public void tearDown() throws IOException {

    }

    @Benchmark
    public void buildIndexBenchmark(Blackhole blackhole) throws IOException {
        // score provider using the raw, in-memory vectors
        try (final var graphIndexBuilder = new GraphIndexBuilder(buildScoreProvider, ravv.dimension(), M, beamWidth, 1.2f, 1.2f, true)) {
            final var graphIndex = graphIndexBuilder.build(ravv);
            blackhole.consume(graphIndex);
        }
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}
