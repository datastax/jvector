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
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark that compares the distance calculation of Product Quantized vectors vs full precision vectors.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class PQDistanceCalculationBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQDistanceCalculationBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

    private List<VectorFloat<?>> vectors;
    private PQVectors pqVectors;
    private List<VectorFloat<?>> queryVectors;
    private ProductQuantization pq;
    private BuildScoreProvider buildScoreProvider;
    
    @Param({"768"})
    private int dimension;
    
    @Param({"10000"})
    private int vectorCount;
    
    @Param({"100"})
    private int queryCount;
    
    @Param({"0", "16", "64", "192"})
    private int M; // Number of subspaces for PQ
    

    @Setup
    public void setup() throws IOException {
        log.info("Creating dataset with dimension: {}, vector count: {}, query count: {}", dimension, vectorCount, queryCount);
        
        // Create random vectors
        vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            vectors.add(createRandomVector(dimension));
        }
        
        // Create query vectors
        queryVectors = new ArrayList<>(queryCount);
        for (int i = 0; i < queryCount; i++) {
            queryVectors.add(createRandomVector(dimension));
        }
        
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimension);
        if (M == 0) {
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        } else {
            // Create PQ vectors
            pq = ProductQuantization.compute(ravv, M, 256, true);
            pqVectors = (PQVectors) pq.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(vsf, pqVectors);
        }
        log.info("Created dataset with dimension: {}, vector count: {}, query count: {}", dimension, vectorCount, queryCount);
    }

    @Benchmark
    public void distanceCalculation(Blackhole blackhole) {
        float totalSimilarity = 0;

        for (VectorFloat<?> query : queryVectors) {
            final SearchScoreProvider searchScoreProvider = buildScoreProvider.searchProviderFor(query);
            for (int i = 0; i < vectorCount; i++) {
                float similarity = searchScoreProvider.scoreFunction().similarityTo(i);
                totalSimilarity += similarity;
            }
        }

        blackhole.consume(totalSimilarity);
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}