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

    // Index-construction phase (single pass per run)
    private final Map<String, QuantStats> indexByType = new HashMap<>();

    // Search phase (reset per evaluated config)
    private final Map<String, QuantStats> searchByType = new HashMap<>();

    ConstructionMetrics() {}

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

        // Index-construction quant timings
        appendQuant(out, "construction.index_quant_time_s", "Idx", indexByType);

        // Search-phase quant timings (note: search.* namespace matches other query-time metrics)
        appendQuant(out, "search.search_quant_time_s", "Search", searchByType);
    }

    private static void appendQuant(List<Metric> out,
                                    String prefix,
                                    String phaseLabel,
                                    Map<String, QuantStats> byType) {
        final String hPrefix = (phaseLabel == null || phaseLabel.isBlank()) ? "" : (phaseLabel + " ");

        for (var e : byType.entrySet()) {
            String qt = e.getKey();
            QuantStats qs = e.getValue();

            // Compute
            if (qs.computeTimeS != null) {
                out.add(Metric.of(prefix + "." + qt + ".compute_time_s",
                        hPrefix + qt + " compute (s)", ".2f", qs.computeTimeS));
            }

            // Encoding (usually single; suffix only if multiple)
            int n = qs.encodeTimesS.size();
            if (n == 1) {
                out.add(Metric.of(prefix + "." + qt + ".encoding_time_s",
                        hPrefix + qt + " encoding (s)", ".2f", qs.encodeTimesS.get(0)));
            } else if (n > 1) {
                for (int i = 0; i < n; i++) {
                    out.add(Metric.of(prefix + "." + qt + ".encoding_time_s." + i,
                            hPrefix + qt + " encoding (s) #" + i, ".2f", qs.encodeTimesS.get(i)));
                }
            }
        }
    }

    private static final class QuantStats {
        Double computeTimeS;
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
            finally { qs.computeTimeS = (System.nanoTime() - t0) / 1_000_000_000.0; }
        }

        <T> T timeEncode(java.util.function.Supplier<T> f) {
            if (qs == null) return f.get();
            long t0 = System.nanoTime();
            try { return f.get(); }
            finally { qs.encodeTimesS.add((System.nanoTime() - t0) / 1_000_000_000.0); }
        }
    }
}
