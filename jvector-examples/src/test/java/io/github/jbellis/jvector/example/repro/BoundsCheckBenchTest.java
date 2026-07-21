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

package io.github.jbellis.jvector.example.repro;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/// A/B microbenchmark for the F3 per-record-access bounds checks (`requireValidNode` + the real
/// degree check in `OnDiskGraphIndex.View`): measures single-threaded exact-scored search
/// throughput over an on-disk graph, the hot path where every neighbor-block read and every
/// scored vector read passes the guard.
///
/// Not part of the normal suite — gated on `REPRO_BENCH=1`. Protocol (one fresh JVM per run, the
/// graph file is built once and reused so both variants read identical bytes):
///
/// ```
/// REPRO_BENCH=1 mvn test -pl jvector-examples -am -Dtest='BoundsCheckBenchTest' \
///   -Dsurefire.failIfNoSpecifiedTests=false \
///   -DargLine="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED"
/// # run WITH checks (working tree), then:
/// git stash push -- jvector-base/src/main/java/io/github/jbellis/jvector/graph/disk/OnDiskGraphIndex.java
/// # run WITHOUT, pop, repeat interleaved
/// ```
///
/// The printed `checksum` folds every result node id and score; it must be identical across
/// variants — proof that both did exactly the same traversal work.
public class BoundsCheckBenchTest {
    private static final int N = 32_768;
    private static final int DIM = 64;
    private static final int QUERIES = 4096;
    private static final int TOP_K = 10;
    private static final int WARMUP_ROUNDS = 5;
    private static final int MEASURED_ROUNDS = 15;

    @Test(timeout = 1_800_000)
    public void searchThroughputAB() throws Exception {
        Assume.assumeTrue("set REPRO_BENCH=1 to run the bounds-check benchmark",
                "1".equals(System.getenv("REPRO_BENCH")));

        System.out.println("BENCH|provider|" + VectorizationProvider.getInstance().getClass().getSimpleName());

        // Build once, reuse across variant runs: identical bytes for both sides of the A/B.
        Path benchDir = ReproGraphs.newWorkDir("bench").getParent().resolve("bench-fixed");
        Files.createDirectories(benchDir);
        Path graphPath = benchDir.resolve("bench_n" + N + "_d" + DIM + ".graph");
        if (!Files.exists(graphPath)) {
            long t0 = System.nanoTime();
            ReproGraphs.buildInlineGraph(graphPath, ReproGraphs.randomVectors(N, DIM, 777), DIM);
            System.out.println("BENCH|built|" + ((System.nanoTime() - t0) / 1_000_000) + "ms|" + Files.size(graphPath) + "bytes");
        } else {
            System.out.println("BENCH|reused|" + Files.size(graphPath) + "bytes");
        }

        List<VectorFloat<?>> queries = ReproGraphs.randomVectors(QUERIES, DIM, 999);

        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath)) {
            OnDiskGraphIndex graph = OnDiskGraphIndex.load(rs);
            try (var scoringView = graph.getView();
                 GraphSearcher searcher = new GraphSearcher(graph)) {

                long[] roundUs = new long[MEASURED_ROUNDS];
                long checksum = 0;
                for (int round = 0; round < WARMUP_ROUNDS + MEASURED_ROUNDS; round++) {
                    long sink = 0;
                    long start = System.nanoTime();
                    for (VectorFloat<?> q : queries) {
                        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, scoringView);
                        SearchResult sr = searcher.search(ssp, TOP_K, Bits.ALL);
                        for (var node : sr.getNodes()) {
                            sink += node.node;
                            sink += Float.floatToIntBits(node.score);
                        }
                    }
                    long elapsedUs = (System.nanoTime() - start) / 1_000;
                    boolean measured = round >= WARMUP_ROUNDS;
                    if (measured) {
                        roundUs[round - WARMUP_ROUNDS] = elapsedUs;
                        checksum = sink;   // identical every round; keep the last
                    }
                    System.out.println("BENCH|round|" + (measured ? "measured" : "warmup") + "|" + round
                            + "|" + elapsedUs + "us|" + (QUERIES * 1_000_000L / Math.max(1, elapsedUs)) + "qps|sink=" + sink);
                }

                long[] sorted = roundUs.clone();
                Arrays.sort(sorted);
                long median = sorted[sorted.length / 2];
                long best = sorted[0];
                System.out.println("BENCH|summary|median=" + median + "us|best=" + best
                        + "us|medianQps=" + (QUERIES * 1_000_000L / Math.max(1, median))
                        + "|nsPerQuery=" + (median * 1_000L / QUERIES)
                        + "|checksum=" + checksum);
            }
        }
    }
}
