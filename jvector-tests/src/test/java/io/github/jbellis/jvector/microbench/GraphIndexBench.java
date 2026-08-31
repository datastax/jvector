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
package io.github.jbellis.jvector.microbench;


import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
public class GraphIndexBench {

    static VectorFloat<?>[] createRandomFloatVectors(int size, int dimension, Random random) {
        VectorFloat<?>[] vectors = new VectorFloat<?>[size];
        for (int offset = 0; offset < size; offset++) {
            vectors[offset] = TestUtil.randomVector(random, dimension);
        }
        return vectors;
    }

    static class TestRandomAccessReader implements RandomAccessVectorValues {
        private final VectorFloat<?>[] values;

        TestRandomAccessReader(VectorFloat<?>[] values) {
            this.values = values;
        }

        @Override
        public int size() {
            return values.length;
        }

        @Override
        public int dimension() {
            return values[0].length();
        }

        @Override
        public VectorFloat<?> getVector(int targetOrd) {
            return values[targetOrd];
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return new TestRandomAccessReader(values);
        }
    }

    @State(Scope.Benchmark)
    public static class Parameters {
        final Random r;
        final TestRandomAccessReader vectors;

        public Parameters() {
            this.r = new Random(1337);
            this.vectors = new TestRandomAccessReader(createRandomFloatVectors(100000, 1536, r));
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testGraphBuild(Blackhole bh, Parameters p) {
        GraphIndexBuilder graphIndexBuilder =  new GraphIndexBuilder(p.vectors, VectorSimilarityFunction.DOT_PRODUCT, 8, 60, 1.2f, 1.4f, false);
        bh.consume(graphIndexBuilder.build(p.vectors));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testGraphBuildWithHierarchy(Blackhole bh, Parameters p) {
        GraphIndexBuilder graphIndexBuilder =  new GraphIndexBuilder(p.vectors, VectorSimilarityFunction.DOT_PRODUCT, 8, 60, 1.2f, 1.4f, true);
        bh.consume(graphIndexBuilder.build(p.vectors));
    }

}

