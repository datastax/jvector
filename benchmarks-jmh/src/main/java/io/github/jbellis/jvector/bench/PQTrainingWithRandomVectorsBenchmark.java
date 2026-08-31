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
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class PQTrainingWithRandomVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQTrainingWithRandomVectorsBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private RandomAccessVectorValues ravv;
    @Param({"16", "32", "64"})
    private int M; // Number of subspaces
    @Param({"768"})
    int originalDimension;
    @Param({"100000"})
    int vectorCount;

    @Setup
    public void setup() throws IOException {
        log.info("Pre-creating vector dataset with original dimension: {}, vector count: {}", originalDimension, vectorCount);
        final List<VectorFloat<?>> vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            float[] vector = new float[originalDimension];
            for (int j = 0; j < originalDimension; j++) {
                vector[j] = (float) Math.random();
            }
            VectorFloat<?> floatVector = VECTOR_TYPE_SUPPORT.createFloatVector(vector);
            vectors.add(floatVector);
        }
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(vectors, originalDimension);
        log.info("Pre-created vector dataset with original dimension: {}, vector count: {}", originalDimension, vectorCount);
    }

    @TearDown
    public void tearDown() throws IOException, InterruptedException {

    }

    @Benchmark
    public void productQuantizationComputeBenchmark(Blackhole blackhole) throws IOException {
        // Compress the original vectors using PQ. this represents a compression ratio of 128 * 4 / 16 = 32x
        ProductQuantization pq = ProductQuantization.compute(ravv,
                M, // number of subspaces
                256, // number of centroids per subspace
                true // center the dataset
        );

        blackhole.consume(pq);
    }
}
