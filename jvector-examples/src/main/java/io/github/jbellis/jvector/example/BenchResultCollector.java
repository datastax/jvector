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

import io.github.jbellis.jvector.example.benchmarks.Metric;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Collects one {@link BenchResult} per measurement for AutoBenchYAML. BenchmarkSummarizer and the CI
 * regression comparison read these parameter and metric names, so keep them stable.
 */
final class BenchResultCollector implements GridListener {
    private final List<BenchResult> results = new ArrayList<>();

    List<BenchResult> results() {
        return results;
    }

    @Override
    public void onQueryResult(SearchContext ctx, QueryRow row) {
        results.add(toBenchResult(ctx, row));
    }

    private static BenchResult toBenchResult(SearchContext ctx, QueryRow row) {
        Map<String, Object> params = new LinkedHashMap<>(ctx.build.reportedParameters());
        params.put("features", ctx.requestedFeatures.toString());
        params.put("buildCompressor", String.valueOf(ctx.build.buildCompressor));
        params.put("searchCompressor", ctx.searchCompressor.toString());
        params.put("topK", row.topK);
        params.put("overquery", row.overquery);
        params.put("usePruning", row.usePruning);

        // Collect all metrics including memory and disk usage
        Map<String, Object> allMetrics = new HashMap<>();
        for (Metric metric : row.metrics) {
            allMetrics.put(metric.getHeader(), metric.getValue());
        }

        // Add build time if available
        if (ctx.indexBuildTimeS != null) {
            allMetrics.put("Index Build Time", ctx.indexBuildTimeS);
        }

        // Add memory metrics if available
        var buildSnapshot = ctx.buildSnapshot;
        if (buildSnapshot != null) {
            allMetrics.put("Heap Memory Used (MB)", buildSnapshot.memoryStats.heapUsed / 1024.0 / 1024.0);
            allMetrics.put("Heap Memory Max (MB)", buildSnapshot.memoryStats.heapMax / 1024.0 / 1024.0);
            allMetrics.put("Off-Heap Direct (MB)", buildSnapshot.memoryStats.directBufferMemory / 1024.0 / 1024.0);
            allMetrics.put("Off-Heap Mapped (MB)", buildSnapshot.memoryStats.mappedBufferMemory / 1024.0 / 1024.0);
            allMetrics.put("Total Off-Heap (MB)", buildSnapshot.memoryStats.getTotalOffHeapMemory() / 1024.0 / 1024.0);
        }

        // Add per-feature-set index file size
        if (ctx.indexFileSizeBytes != null) {
            allMetrics.put("Disk Usage (MB)", ctx.indexFileSizeBytes / 1024.0 / 1024.0);
        }

        return new BenchResult(ctx.ds.getName(), params, allMetrics);
    }
}
