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

import java.util.Collections;
import java.util.Locale;
import java.util.Map;

/**
 * Prints a table of benchmark results.
 * First call prints the header, then each call to printRow() prints one row.
 */
public class BenchmarkTablePrinter {
    private final int topK;
    private boolean headerPrinted = false;

    /**
     * Create a table printer that uses the given K for the recall column header.
     *
     * @param topK  Used to produce a header of “Recall@<topK>”
     */
    public BenchmarkTablePrinter(int topK) {
        this.topK = topK;
    }

    /**
     * Call this once to print all the global parameters before the table.
     * 
     * @param params  A map from parameter name (e.g. "mGrid") to its List value.
     */
    public void printConfig(Map<String, ?> params) {
        System.out.println();
        System.out.println("Configuration:");
        params.forEach((name, values) ->
                System.out.printf(Locale.US, "  %-22s: %s%n", name, values)
        );
    }

    /**
     * Prints one row.  On the very first call, prints the header first.
     *
     * @param overquery       the sweep parameter (first column)
     * @param throughput      from ThroughputBenchmark (for QPS)
     * @param countSummary    from CountBenchmark (for avg nodes visited)
     * @param latencySummary  from LatencyBenchmark (for mean latency)
     * @param recallSummary   from RecallBenchmark (for recall)
     */
    public void printRow(double overquery,
                         ThroughputBenchmark.Summary throughput,
                         CountBenchmark.Summary countSummary,
                         LatencyBenchmark.Summary latencySummary,
                         RecallBenchmark.Summary recallSummary) {
        if (!headerPrinted) {
            System.out.println();
            printHeader();
            headerPrinted = true;
        }

        double qps           = throughput.getQueriesPerSecond();
        double avgVisited    = countSummary.getAvgNodesVisited();
        double meanLatencyMs = latencySummary.getAverageLatency();
        double p999LatencyMs = latencySummary.getP999Latency();
        double recallPct     = recallSummary.getAverageRecall() * 100.0;

        // Column header formatting....
        System.out.printf(Locale.US,
                "%-12.2f %-12.1f %-15.1f %-20.3f %-20.3f %-12.2f%n",
                overquery,
                qps,
                avgVisited,
                meanLatencyMs,
                p999LatencyMs,
                recallPct
        );
    }

    private void printHeader() {
        String recallHeader = "Recall@" + topK;
        String format = "%-12s %-12s %-15s %-20s %-20s %-12s";
        String headerLine = String.format(
                Locale.US,
                format,
                "Overquery", "QPS", "Avg Visited", "Mean Latency (ms)", "p999 Latency (ms)", recallHeader
        );

        System.out.println(headerLine);

        System.out.println(
            String.join("", Collections.nCopies(headerLine.length(), "-"))
        );
    }
}


