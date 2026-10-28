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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.quantization.ASHNeighborhoodScorer;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.Random;

/**
 * DistancesASH entry point for isolated full scans through the production neighborhood scorer.
 * Fixtures, the independent oracle, and the pipelined workload have separate owners. The timed
 * paths include exactly one preparation per complete scan; input loading and graph I/O are untimed.
 */
final class ASHNeighborhoodBenchmark {
    private static volatile double sink;

    private ASHNeighborhoodBenchmark() {}

    static void run(String[] args) throws IOException {
        String filename = System.getProperty("jvector.bench.neighborhoodFixture");
        if (filename == null)
            throw new IllegalArgumentException("Set -Djvector.bench.neighborhoodFixture=/path/to/fixture.ashn");
        Path path = Path.of(filename);
        ASHNeighborhoodFixture fixture;
        if (Files.exists(path))
            fixture = ASHNeighborhoodFixture.read(path);
        else {
            if (args.length != 2 && args.length != 3)
                throw new IllegalArgumentException(
                        "Creating a fixture requires base.fvecs query.fvecs [ground-truth.ivecs]");
            fixture = ASHNeighborhoodFixture.create(args[0], args[1]);
            ASHNeighborhoodFixture.write(path, fixture);
        }
        checkSetting("jvector.ash.bitsPerDimension", fixture.ash.bitsPerDimension);
        checkSetting("jvector.ash.quantizedDimensions", fixture.ash.quantizedDim);
        checkSetting("jvector.ash.landmarkCount", 1);
        checkSetting("jvector.bench.neighborhoodSize", fixture.capacity);
        checkSetting("jvector.bench.blockSize", fixture.blockSize);
        boolean scaling = Boolean.getBoolean("jvector.bench.neighborhood-scaling");
        int threads = positive("jvector.bench.scoringThreads", 1);
        if (!scaling && threads != 1)
            throw new IllegalArgumentException("Multiple scoring threads require neighborhood-scaling=true");
        int queryIndex = Integer.parseInt(System.getProperty("jvector.bench.queryIndex", "0"));
        Objects.checkIndex(queryIndex, fixture.queries.size());
        int passes = positive("jvector.bench.scoringPasses", 5);
        if (System.getProperty("jvector.bench.neighborhoodsPerQuery") != null) {
            throw new IllegalArgumentException(
                    "A full scan scores every saved neighborhood; remove neighborhoodsPerQuery");
        }
        double seconds = seconds("jvector.bench.scoringMinSeconds", .5);
        double warmup = seconds("jvector.bench.scoringWarmupSeconds", 1);
        String access = System.getProperty("jvector.bench.neighborhoodAccess", "sequential");
        int[] order = accessOrder(fixture.neighborhoods.length, access);
        VectorFloat<?> query = fixture.queries.get(queryIndex);
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        // A requested native comparison must fail clearly rather than time the Java fallback.
        if ("native".equals(System.getProperty("jvector.vectorization_provider", "").trim().toLowerCase(Locale.ROOT))
                && !"scalar".equals(
                        System.getProperty("jvector.ash.blockKernel", "auto").trim().toLowerCase(Locale.ROOT))
                && !backend.supportsAshNeighborhoodScoring()) {
            throw new IllegalStateException(
                    "Native ASH comparison requested, but the library lacks the neighborhood kernel");
        }
        System.out.printf(Locale.ROOT,
                "Neighborhood benchmark: java=%s, C=1, singleQuery=true, bits=%d, dimensions=%d, capacity=%d, blockSize=%d%n",
                Runtime.version(), fixture.ash.bitsPerDimension,
                fixture.ash.quantizedDim, fixture.capacity, fixture.blockSize);
        System.out.printf(Locale.ROOT,
                "Fixture: sha256=%s, queries=%d, queryIndex=%d, baseVectors=%d, neighborhoods=%d, packedBytes=%d, access=%s%n",
                ASHNeighborhoodFixture.sha256(path), fixture.queries.size(), queryIndex, fixture.baseCount,
                order.length, (long) order.length * fixture.neighborhoods[0].length, access);
        System.out.printf("Backend: %s; %s%n", backend.getClass().getSimpleName(), backend.ashKernelDescription());
        var scorer = new ASHNeighborhoodScorer(fixture.ash, query, fixture.capacity, fixture.blockSize);
        System.out.println("Scorer: " + scorer);
        ASHNeighborhoodValidator.validate(fixture);
        if (scaling) {
            runPipelinedScans(fixture, query, order, threads, passes, warmup, seconds);
            return;
        }
        System.out.println(
                "Full scan: query preparation once per scan, all neighborhoods, headers, similarity conversion and checksum consumption are timed. Loading, packing, graph traversal and I/O are outside timing.");
        float[] scores = new float[fixture.capacity];
        Runnable fullScan = () -> {
            var current = new ASHNeighborhoodScorer(fixture.ash, query, fixture.capacity, fixture.blockSize);
            double checksum = 0;
            for (int n : order) {
                current.score(fixture.neighborhoods[n], scores);
                for (float score : scores) checksum += score;
            }
            sink = checksum;
        };
        measure(fullScan, warmup);
        System.out.println(
                "mode,pass,ns_per_scan,ns_per_neighborhood,base_vectors_per_second,scored_lanes_per_second,checksum");
        for (int pass = 1; pass <= passes; pass++) {
            double ns = measure(fullScan, seconds);
            System.out.printf(Locale.ROOT, "full_scan,%d,%.3f,%.3f,%.3f,%.3f,%.9f%n", pass, ns, ns / order.length,
                    fixture.baseCount * 1e9 / ns, order.length * (double) fixture.capacity * 1e9 / ns, sink);
        }
    }

    private static void runPipelinedScans(ASHNeighborhoodFixture fixture, VectorFloat<?> query, int[] order,
            int threads, int passes, double warmup, double seconds) {
        System.out.printf(Locale.ROOT,
                "Strong scaling: workers=%d, fixedNeighborhoodsPerSweep=%d, fixedScoredLanesPerSweep=%d; one shared query, disjoint neighborhood partitions%n",
                threads, order.length, (long) order.length * fixture.capacity);
        System.out.println(
                "Pipelined full scans: prepare each query once while workers score preceding scans. At most two scans are in flight. Preparation, submission, completion, checksum consumption, fill and drain are timed. Persistent-worker creation is outside timing; all threads share the stated CPU affinity.");
        try (var workload = new ASHNeighborhoodPipeline(fixture, order, threads)) {
            workload.measure(query, warmup);
            System.out.println(
                    "workers,pass,ns_per_scan,ns_per_neighborhood,base_vectors_per_second,scored_lanes_per_second,ns_preparation_per_scan,checksum");
            for (int pass = 1; pass <= passes; pass++) {
                ASHNeighborhoodPipeline.Timing timing = workload.measure(query, seconds);
                double nanos = timing.nanosPerScan;
                System.out.printf(Locale.ROOT, "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.9f%n", threads, pass, nanos,
                        nanos / order.length, fixture.baseCount * 1e9 / nanos,
                        order.length * (double) fixture.capacity * 1e9 / nanos, timing.preparationNanosPerScan,
                        timing.checksum);
            }
        }
    }

    private static double measure(Runnable task, double seconds) {
        long start = System.nanoTime(), elapsed;
        long count = 0;
        do {
            task.run();
            count++;
            elapsed = System.nanoTime() - start;
        } while (elapsed < seconds * 1e9);
        return (double) elapsed / count;
    }

    static int[] accessOrder(int count, String access) {
        if (!access.equals("sequential") && !access.equals("shuffled"))
            throw new IllegalArgumentException("neighborhoodAccess must be sequential or shuffled");
        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        if (access.equals("shuffled")) {
            Random random = new Random(87342);
            for (int i = count - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), swap = order[i];
                order[i] = order[j];
                order[j] = swap;
            }
        }
        return order;
    }

    static int positive(String name, int fallback) {
        int value;
        try {
            value = Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(name + " must be an integer", error);
        }
        if (value <= 0)
            throw new IllegalArgumentException(name + " must be positive");
        return value;
    }

    private static double seconds(String name, double fallback) {
        double value = Double.parseDouble(System.getProperty(name, Double.toString(fallback)));
        if (!Double.isFinite(value) || value <= 0)
            throw new IllegalArgumentException(name + " must be finite and positive");
        return value;
    }

    static void checkSetting(String name, int expected) {
        String value = System.getProperty(name);
        if (value != null && Integer.parseInt(value) != expected)
            throw new IllegalArgumentException(name + " disagrees with saved fixture: " + expected);
    }
}
