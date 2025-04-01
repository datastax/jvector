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
package io.github.jbellis.jvector.bench.output;

import java.util.ArrayList;
import java.util.List;

public class TextTable implements TableRepresentation {
    private final List<String> resultsTable = new ArrayList<>();

    public TextTable() {
        resultsTable.add("\n");
        resultsTable.add(String.format(
                "%-12s | %-12s | %-18s | %-18s | %-15s | %-12s",
                "Elapsed(s)", "QPS", "Mean Latency (µs)", "P99.9 Latency (µs)", "Mean Visited", "Recall (%)"
        ));
        resultsTable.add("-----------------------------------------------------------------------------------------------");
    }

    @Override
    public void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited, double recallPercentage) {
        resultsTable.add(String.format(
                "%-12d | %-12d | %-18.3f | %-18.3f | %-15.3f | %10.2f%%",
                elapsedSeconds, qps, meanLatency, p999Latency, meanVisited, recallPercentage*100
        ));
    }

    @Override
    public void print() {
        for (String line : resultsTable) {
            System.out.println(line);
        }
    }
}
