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

import io.github.jbellis.jvector.bench.output.TextTable;
import io.github.jbellis.jvector.example.util.SiftLoader;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class StaticSetVectorsTabularBenchmark extends AbstractVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(StaticSetVectorsTabularBenchmark.class);

    @Setup
    public void setup() throws IOException {
        tableRepresentation = new TextTable();
        var siftPath = "siftsmall";
        baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        commonSetupStatic(true);
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
        graphIndexBuilder.close();
        scheduler.shutdown();
        tableRepresentation.print();
    }

    @Benchmark
    public void testOnHeapWithStaticQueryVectors(Blackhole blackhole) {
        commonTest(blackhole);
    }

}
