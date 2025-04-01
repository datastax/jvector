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
import io.github.jbellis.jvector.example.SiftSmall;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.util.concurrent.*;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class RandomVectorsTabularBenchmark extends AbstractVectorsBenchmark {
    @Param({"1000", "10000", "100000", "1000000"})
    int numBaseVectors;

    @Param({"10"})
    int numQueryVectors;

    @Setup
    public void setup() throws IOException {
        tableRepresentation = new TextTable();
        commonSetupRandom(numBaseVectors, numQueryVectors);

        transactionCount.set(0);
        totalLatency.set(0);
        latencySamples.clear();
        visitedSamples.clear();
        recallSamples.clear();
        startTime = System.currentTimeMillis();
        scheduler = Executors.newScheduledThreadPool(1);
        totalTransactions = 0;

        scheduler.scheduleAtFixedRate(() -> {
            long elapsed = (System.currentTimeMillis() - startTime) / 1000;
            long count = transactionCount.getAndSet(0);
            double meanLatency = (totalTransactions > 0) ? (double) totalLatency.get() / totalTransactions : 0.0;
            double p999Latency = calculateP999Latency();
            double meanVisited = (totalTransactions > 0) ? (double) visitedSamples.stream().mapToInt(Integer::intValue).sum() / totalTransactions : 0.0;
            double recall = (totalTransactions > 0) ? recallSamples.stream().mapToDouble(Double::doubleValue).sum() / totalTransactions : 0.0;
            tableRepresentation.addEntry(elapsed, count, meanLatency, p999Latency, meanVisited, recall);
        }, 1, 1, TimeUnit.SECONDS);
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        graphIndexBuilder.close();
        scheduler.shutdown();
        tableRepresentation.print();
    }

    @Benchmark
    public void testOnHeapRandomVectors(Blackhole blackhole) {
        long start = System.nanoTime();
        var queryVector = SiftSmall.randomVector(originalDimension);
        var searchResult = GraphSearcher.search(queryVector,
                10,                            // number of results
                ravv,                               // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL);                          // valid ordinals to consider
        blackhole.consume(searchResult);
        long duration = System.nanoTime() - start;
        long durationMicro = TimeUnit.NANOSECONDS.toMicros(duration);

        visitedSamples.add(searchResult.getVisitedCount());
        transactionCount.incrementAndGet();
        totalLatency.addAndGet(durationMicro);
        latencySamples.add(durationMicro);
        totalTransactions++;
    }

}
