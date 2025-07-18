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

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;
import org.apache.commons.math3.stat.StatUtils;

/**
 * Measures throughput (queries/sec) with an optional warmup phase.
 */
public class ThroughputBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".1f";

    private static volatile long SINK;

    private final int numWarmupRuns;
    private final int numTestRuns;
    private boolean computeAvgQps;
    private boolean computeMedianQps;
    private boolean computeMaxQps;
    private String formatAvgQps;
    private String formatMedianQps;
    private String formatMaxQps;

    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static ThroughputBenchmark createDefault() {
        return new ThroughputBenchmark(3, 3,
                true, false, false,
                DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    public static ThroughputBenchmark createEmpty(int numWarmupRuns, int numTestRuns) {
        return new ThroughputBenchmark(numWarmupRuns, numTestRuns,
                false, false, false,
                DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    private ThroughputBenchmark(int numWarmupRuns, int numTestRuns,
                                boolean computeAvgQps, boolean computeMedianQps, boolean computeMaxQps,
                                String formatAvgQps, String formatMedianQps, String formatMaxQps) {
        this.numWarmupRuns = numWarmupRuns;
        this.numTestRuns = numTestRuns;
        this.computeAvgQps = computeAvgQps;
        this.computeMedianQps = computeMedianQps;
        this.computeMaxQps = computeMaxQps;
        this.formatAvgQps = formatAvgQps;
        this.formatMedianQps = formatMedianQps;
        this.formatMaxQps = formatMaxQps;
    }

    public ThroughputBenchmark displayAvgQps() {
        return displayAvgQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayAvgQps(String format) {
        this.computeAvgQps = true;
        this.formatAvgQps = format;
        return this;
    }

    public ThroughputBenchmark displayMedianQps() {
        return displayMedianQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayMedianQps(String format) {
        this.computeMedianQps = true;
        this.formatMedianQps = format;
        return this;
    }

    public ThroughputBenchmark displayMaxQps() {
        return displayMaxQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayMaxQps(String format) {
        this.computeMaxQps = true;
        this.formatMaxQps = format;
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

        if (!(computeAvgQps || computeMedianQps || computeMaxQps)) {
            throw new RuntimeException("At least one metric must be displayed");
        }

        int totalQueries = cs.getDataSet().queryVectors.size();
        int dim = cs.getDataSet().getDimension();

        // Warmup Phase: Use randomly-generated vectors
        for (int warmupRun = 0; warmupRun < numWarmupRuns; warmupRun++) {
            IntStream.range(0, totalQueries)
                    .parallel()
                    .forEach(k -> {
                        // Generate a random vector
                        VectorFloat<?> randQ = vts.createFloatVector(dim);
                        for (int j = 0; j < dim; j++) {
                            randQ.set(j, ThreadLocalRandom.current().nextFloat());
                        }
                        VectorUtil.l2normalize(randQ);
                        SearchResult sr = QueryExecutor.executeQuery(
                                cs, topK, rerankK, usePruning, randQ);
                        SINK += sr.getVisitedCount();
                    });
        }

        double[] qpsSamples = new double[numTestRuns];
        for (int testRun = 0; testRun < numTestRuns; testRun++) {

            // Clear Eden and let GC complete....
            System.gc();
            System.runFinalization();
            try {
                Thread.sleep(500);   // 100 ms is usually plenty
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }

            // Test Phase: Execute the query set or more times
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
            double runQps = totalQueries / elapsedSec;
            qpsSamples[testRun] = runQps;

        }

        Arrays.sort(qpsSamples);
        double medianQps = qpsSamples[numTestRuns/2];  // middle element (for odd)
        double avgQps = StatUtils.mean(qpsSamples);
        double stdDevQps = Math.sqrt(StatUtils.variance(qpsSamples));
        double maxQps = StatUtils.max(qpsSamples);

        var list = new ArrayList<Metric>();
        if (computeAvgQps) {
            list.add(Metric.of("Avg QPS (of " + numTestRuns + ")", formatAvgQps, avgQps));
            list.add(Metric.of("± Std Dev", formatAvgQps, stdDevQps));
        }
        if (computeMedianQps) {
            list.add(Metric.of("Median QPS (of " + numTestRuns + ")", formatMedianQps, medianQps));
        }
        if (computeMaxQps) {
            list.add(Metric.of("Max QPS (of " + numTestRuns + ")", formatMaxQps, maxQps));
        }
        return list;
    }
}
