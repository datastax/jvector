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

package io.github.jbellis.jvector.example.benchmarks;

import java.util.List;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.ArrayVectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Measures throughput (queries/sec) with an optional warmup phase.
 */
public class ThroughputBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".1f";

    private static volatile long SINK;

    private final int warmupRuns;
    private final int testRuns;
    private String format;

    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static ThroughputBenchmark createDefault(int warmupRuns, int testRuns) {
        return new ThroughputBenchmark(warmupRuns, testRuns, DEFAULT_FORMAT);
    }

    private ThroughputBenchmark(int warmupRuns, int testRuns, String format) {
        this.warmupRuns = warmupRuns;
        this.testRuns = testRuns;
        this.format = format;
    }

    public ThroughputBenchmark setFormat(String format) {
        this.format = format;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "ThroughputBenchmark";
    }

    @Override
    public List<Metric> runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        int totalQueries = cs.getDataSet().queryVectors.size();
        int dim = cs.getDataSet().getDimension();
        double maxQps    = 0;

        for (int testRun = 0; testRun < testRuns; testRun++) {
        // Warmup Phase
            for (int warmupRun = 0; warmupRun < warmupRuns; warmupRun++) {
                IntStream.range(0, totalQueries)
                        .parallel()
                        .forEach(k -> {
                            // Generate a random vector
                            VectorFloat<?> randQ = vts.createFloatVector(dim);
                            for (int j = 0; j < dim; j++) {
                                randQ.set(j, ThreadLocalRandom.current().nextFloat());
                            }
                            SearchResult sr = QueryExecutor.executeQuery(
                                    cs, topK, rerankK, usePruning, randQ);
                            SINK += sr.getVisitedCount();
                        });
            }

            // Test Phase
            LongAdder visitedAdder = new LongAdder();
            long startTime = System.nanoTime();
            IntStream.range(0, totalQueries)
                    .parallel()
                    .forEach(i -> {
                        SearchResult sr = QueryExecutor.executeQuery(
                                cs, topK, rerankK, usePruning, i);
                        // “Use” the result to prevent optimization
                        visitedAdder.add(sr.getVisitedCount());
                    });
            double elapsedSec = (System.nanoTime() - startTime) / 1e9;
            double runQps = totalQueries / elapsedSec;      // ← new
            maxQps       = Math.max(maxQps, runQps);
        }

        return List.of(Metric.of("QPS", format, maxQps));
    }
}
