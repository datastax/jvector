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

import io.github.jbellis.jvector.bench.output.TableRepresentation;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public abstract class AbstractVectorsBenchmark {
    protected static final Logger log = LoggerFactory.getLogger(AbstractVectorsBenchmark.class);
    protected List<VectorFloat<?>> baseVectors;
    protected List<VectorFloat<?>> queryVectors;
    protected int originalDimension;
    protected RandomAccessVectorValues ravv;
    protected GraphIndexBuilder graphIndexBuilder;
    protected GraphIndex graphIndex;
    protected long totalTransactions;
    protected final AtomicLong transactionCount = new AtomicLong(0);
    protected final AtomicLong totalLatency = new AtomicLong(0);
    protected final Queue<Long> latencySamples = new ConcurrentLinkedQueue<>(); // Store latencies for P99.9
    protected final Queue<Integer> visitedSamples = new ConcurrentLinkedQueue<>();
    protected final Queue<Double> recallSamples = new ConcurrentLinkedQueue<>();
    protected ScheduledExecutorService scheduler;
    protected long startTime;

    protected List<? extends Set<Integer>> groundTruth;
    protected final AtomicInteger testCycle = new AtomicInteger(0);
    protected TableRepresentation tableRepresentation;

    protected static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    protected void commonSetupStatic(boolean schedule) {
        log.info("base vectors size: {}, query vectors size: {}, loaded, dimensions {}",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
        originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16,  // graph degree
                100,    // construction search depth
                1.2f,   // allow degree overflow during construction by this factor
                1.2f,   // relax neighbor diversity requirement by this factor
                true);  // relax neighbor diversity requirement by this factor
        graphIndex = graphIndexBuilder.build(ravv);
        if (schedule) {
            schedule();
        }
    }

    protected void schedule() {
        totalTransactions = 0;
        transactionCount.set(0);
        totalLatency.set(0);
        latencySamples.clear();
        visitedSamples.clear();
        recallSamples.clear();
        startTime = System.currentTimeMillis();
        scheduler = Executors.newScheduledThreadPool(1);
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

    protected void commonSetupRandom(int numBaseVectors, int numQueryVectors) {
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
                true); // relax neighbor diversity requirement by this factor
        graphIndex = graphIndexBuilder.build(ravv);
    }

    protected VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }

    protected void commonTest(Blackhole blackhole) {
        long start = System.nanoTime();
        int offset = testCycle.getAndIncrement() % queryVectors.size();
        var queryVector = queryVectors.get(offset);
        var searchResult = GraphSearcher.search(queryVector,
                10, // number of results
                ravv, // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL); // valid ordinals to consider
        blackhole.consume(searchResult);
        long duration = System.nanoTime() - start;
        long durationMicro = TimeUnit.NANOSECONDS.toMicros(duration);

        calculateRecall(searchResult, offset);
        visitedSamples.add(searchResult.getVisitedCount());
        transactionCount.incrementAndGet();
        totalLatency.addAndGet(durationMicro);
        latencySamples.add(durationMicro);
        totalTransactions++;
    }

    protected double calculateP999Latency() {
        if (latencySamples.isEmpty()) return 0.0;

        List<Long> sortedLatencies = new ArrayList<>(latencySamples);
        Collections.sort(sortedLatencies);

        int p_count = sortedLatencies.size() / 1000;
        int start_index = Math.max(0, sortedLatencies.size() - p_count);
        int end_index = sortedLatencies.size() - 1;
        List<Long> subList = sortedLatencies.subList(start_index, end_index);
        long total = subList.stream().mapToLong(Long::longValue).sum();
        return (double) total / p_count;
    }

    /*
     * Note that in this interpretation of recall we are simply counting the number of vectors in the result set
     * that are also present in the ground truth. We are not factoring in the possibility of a mismatch in size
     * between top k and depth of ground truth, meaning e.g. for topk=10 and gt depth=100 as long as the 10 returned
     * values are in the ground truth we get 100% recall despite missing 90% of the ground truth or conversely if
     * topk=100 and gt depth=10 as long as all 10 ground truth values are in the result set we get 100% recall despite
     * having returned 90 extraneous results which may or may not be correct in terms of distance from the query vector.
     *
     * Ordering is also not considered so that, going back to the example of topk=100 and gt depth=10, even if the first
     * 90 results are incorrect but the last 10 match the ground truth we still get 100% recall.
     */
    protected void calculateRecall(SearchResult searchResult, int offset) {
        Set<Integer> gt = groundTruth.get(offset);
        long n = Arrays.stream(searchResult.getNodes())
                .filter(ns -> gt.contains(ns.node))
                .count();
        recallSamples.add(((double)n / (searchResult.getNodes().length)));
    }
}
