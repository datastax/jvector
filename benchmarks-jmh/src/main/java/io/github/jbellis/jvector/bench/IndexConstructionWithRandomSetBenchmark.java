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
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
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

    @Param({"Exact", "PQ"})
    String buildScoreProviderType;

    @Setup(Level.Invocation)
    public void setup() throws IOException {

        final var baseVectors = new ArrayList<VectorFloat<?>>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        if (buildScoreProviderType.equals("PQ")) {
            log.info("Using PQ build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            int numberOfSubspaces = getDefaultNumberOfSubspacesPerVector(originalDimension);
            final ProductQuantization pq = ProductQuantization.compute(ravv,
                    numberOfSubspaces,
                    256,
                    true);
            final PQVectors pqVectors = (PQVectors) pq.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);
        } else if (buildScoreProviderType.equals("Exact")) {
            log.info("Using Exact build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            // score provider using the raw, in-memory vectors
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        } else {
            throw new IllegalArgumentException("Unknown build score provider type: " + buildScoreProviderType);
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

    /**
     * This method returns the default number of subspaces per vector for a given original dimension.
     * Should be used as a default value for the number of subspaces per vector in case no value is provided.
     *
     * @param originalDimension original vector dimension
     * @return default number of subspaces per vector
     */
    public static int getDefaultNumberOfSubspacesPerVector(int originalDimension) {
        // the idea here is that higher dimensions compress well, but not so well that we should use fewer bits
        // than a lower-dimension vector, which is what you could get with cutoff points to switch between (e.g.)
        // D*0.5 and D*0.25. Thus, the following ensures that bytes per vector is strictly increasing with D.
        int compressedBytes;
        if (originalDimension <= 32) {
            // We are compressing from 4-byte floats to single-byte codebook indexes,
            // so this represents compression of 4x
            // * GloVe-25 needs 25 BPV to achieve good recall
            compressedBytes = originalDimension;
        } else if (originalDimension <= 64) {
            // * GloVe-50 performs fine at 25
            compressedBytes = 32;
        } else if (originalDimension <= 200) {
            // * GloVe-100 and -200 perform well at 50 and 100 BPV, respectively
            compressedBytes = (int) (originalDimension * 0.5);
        } else if (originalDimension <= 400) {
            // * NYTimes-256 actually performs fine at 64 BPV but we'll be conservative
            // since we don't want BPV to decrease
            compressedBytes = 100;
        } else if (originalDimension <= 768) {
            // allow BPV to increase linearly up to 192
            compressedBytes = (int) (originalDimension * 0.25);
        } else if (originalDimension <= 1536) {
            // * ada002 vectors have good recall even at 192 BPV = compression of 32x
            compressedBytes = 192;
        } else {
            // We have not tested recall with larger vectors than this, let's let it increase linearly
            compressedBytes = (int) (originalDimension * 0.125);
        }
        return compressedBytes;
    }
}
