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
import java.util.List;
import java.util.Map;

/**
 * Construction/search quantization timing metrics captured while building an index and while encoding
 * vectors for search-time evaluation.
 *
 * <p>Values may be {@code null} when not measured or not applicable (e.g., cache hit).</p>
 *
 * <p>Call {@link #appendTo(List)} to emit these as {@link Metric} entries so they can be selected for
 * console/CSV reporting.</p>
 */
final class ConstructionMetrics {
    // null means “not applicable / not measured”
    public Double indexBuildTimeS;
    public Long indexFileSizeBytes;
    // Schema 3 on-disk phases: graph construction and cleanup, writing the completed graph,
    // and reopening the written files. Graph build plus flush equals indexBuildTimeS.
    public Double graphBuildTimeS;
    public Double flushTimeS;
    public Double reopenTimeS;

    // Index-construction phase (single pass per run)
    private final Map<String, QuantStats> indexByType = new HashMap<>();

    // Search phase (reset per evaluated config)
    private final Map<String, QuantStats> searchByType = new HashMap<>();

    private final boolean sumIndexQuantTimes;

    ConstructionMetrics() { this(false); }

    ConstructionMetrics(boolean sumIndexQuantTimes) {
        this.sumIndexQuantTimes = sumIndexQuantTimes;
    }

    /** Clear search-phase quant timings for the next evaluated (index, compression) configuration. */
    void resetSearch() {
        searchByType.clear();
    }

    QuantHandle index(String quantType)  { return handle(indexByType, quantType); }
    QuantHandle search(String quantType) { return handle(searchByType, quantType); }

    private static QuantHandle handle(Map<String, QuantStats> map, String quantType) {
        if (quantType == null) return QuantHandle.NOOP;
        QuantStats qs = map.computeIfAbsent(quantType, __ -> new QuantStats());
        return new QuantHandle(qs);
    }

    /** Adds construction + quant metrics as Metric entries (null => blank in CSV). */
    void appendTo(List<Metric> out) {
        if (indexBuildTimeS != null) {
            out.add(Metric.of("construction.index_build_time_s",
                    "Index build time (s)", ".2f", indexBuildTimeS));
        }
        if (indexFileSizeBytes != null) {
            out.add(Metric.of("construction.index_file_size_mb",
                    "Index file size (MB)", ".2f", indexFileSizeBytes / 1024.0 / 1024.0));
        }

        appendPhase(out, "construction.graph_build_time_s", "Graph build time (s)", graphBuildTimeS);
        appendPhase(out, "construction.flush_time_s", "Flush time (s)", flushTimeS);
        appendPhase(out, "construction.reopen_time_s", "Reopen time (s)", reopenTimeS);

        // Index-construction quant timings
        appendQuant(out, "construction.index_quant_time_s", "Idx", indexByType, sumIndexQuantTimes);

        // Search-phase quant timings (note: search.* namespace matches other query-time metrics)
        appendQuant(out, "search.search_quant_time_s", "Search", searchByType, false);
    }

    private static void appendPhase(List<Metric> out, String key, String header, Double seconds) {
        if (seconds != null) out.add(Metric.of(key, header, ".2f", seconds));
    }

    private static void appendQuant(List<Metric> out,
                                    String prefix,
                                    String phaseLabel,
                                    Map<String, QuantStats> byType, boolean sumTimes) {
        final String hPrefix = (phaseLabel == null || phaseLabel.isBlank()) ? "" : (phaseLabel + " ");

        for (var e : byType.entrySet()) {
            String qt = e.getKey();
            QuantStats qs = e.getValue();

            // Compute
            if (!qs.computeTimesS.isEmpty()) {
                double computeTime = sumTimes ? qs.computeTimesS.stream().mapToDouble(Double::doubleValue).sum()
                        : qs.computeTimesS.get(qs.computeTimesS.size() - 1);
                out.add(Metric.of(prefix + "." + qt + ".compute_time_s",
                        hPrefix + qt + " compute (s)", ".2f", computeTime));
            }

            // Encoding (usually single; suffix only if multiple)
            int n = qs.encodeTimesS.size();
            if (n > 0 && (n == 1 || sumTimes)) {
                double encodeTime = qs.encodeTimesS.stream().mapToDouble(Double::doubleValue).sum();
                out.add(Metric.of(prefix + "." + qt + ".encoding_time_s",
                        hPrefix + qt + " encoding (s)", ".2f", encodeTime));
            } else if (n > 1) {
                for (int i = 0; i < n; i++) {
                    out.add(Metric.of(prefix + "." + qt + ".encoding_time_s." + i,
                            hPrefix + qt + " encoding (s) #" + i, ".2f", qs.encodeTimesS.get(i)));
                }
            }
        }
    }

    private static final class QuantStats {
        final List<Double> computeTimesS = new ArrayList<>();
        final List<Double> encodeTimesS = new ArrayList<>();
    }

    static final class QuantHandle {
        static final QuantHandle NOOP = new QuantHandle(null);

        private final QuantStats qs;

        private QuantHandle(QuantStats qs) {
            this.qs = qs;
        }

        <T> T timeCompute(java.util.function.Supplier<T> f) {
            if (qs == null) return f.get();
            long t0 = System.nanoTime();
            try { return f.get(); }
            finally { qs.computeTimesS.add((System.nanoTime() - t0) / 1_000_000_000.0); }
        }

        <T> T timeEncode(java.util.function.Supplier<T> f) {
            if (qs == null) return f.get();
            long t0 = System.nanoTime();
            try { return f.get(); }
            finally { qs.encodeTimesS.add((System.nanoTime() - t0) / 1_000_000_000.0); }
        }
    }
}
