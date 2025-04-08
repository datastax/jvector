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

import io.github.jbellis.jvector.example.SiftSmall;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class RandomVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(RandomVectorsBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private RandomAccessVectorValues ravv;
    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> queryVectors;
    private GraphIndexBuilder graphIndexBuilder;
    private GraphIndex graphIndex;
    int originalDimension;
    @Param({"1000", "10000", "100000", "1000000"})
    int numBaseVectors;
    @Param({"10"})
    int numQueryVectors;

    @Setup
    public void setup() throws IOException {
        originalDimension = 128; // Example dimension, can be adjusted

        baseVectors = new ArrayList<>(numBaseVectors);
        queryVectors = new ArrayList<>(numQueryVectors);

        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }

        for (int i = 0; i < numQueryVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            queryVectors.add(vector);
        }

        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16, // graph degree
                100, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f, // relax neighbor diversity requirement by this factor
                true); // add the hierarchy
        graphIndex = graphIndexBuilder.build(ravv);
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        graphIndexBuilder.close();
    }

    @Benchmark
    public void testOnHeapRandomVectors(Blackhole blackhole) {
        var queryVector = SiftSmall.randomVector(originalDimension);
        // Your benchmark code here
        var searchResult = GraphSearcher.search(queryVector,
                10, // number of results
                ravv, // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL); // valid ordinals to consider
        blackhole.consume(searchResult);
    }
}
