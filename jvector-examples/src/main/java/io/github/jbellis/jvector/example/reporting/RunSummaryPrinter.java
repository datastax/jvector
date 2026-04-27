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

package io.github.jbellis.jvector.example.reporting;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Reads a written {@code experiments.csv} file and prints a console summary
 * that collapses rows across the {@code repetition} dimension. For each group
 * (identical fixed columns excluding {@code repetition}), metric columns are
 * aggregated into count, mean, stddev, and coefficient of variation.
 *
 * <p>Only used when experiments logging is enabled; otherwise the caller should
 * print a one-line notice instead.</p>
 */
public final class RunSummaryPrinter {

    /** Fixed columns that are NOT part of the grouping key. */
    private static final Set<String> NON_GROUPING_FIXED = Set.of(
            "schema_version",
            "run_id",
            "run_uuid",
            "system_id",
            "repetition"
    );

    private RunSummaryPrinter() {}

    /**
     * Read {@code experiments.csv} and print a grouped summary to {@code out}.
     */
    public static void print(Path experimentsCsv, PrintStream out) throws IOException {
        if (!Files.exists(experimentsCsv)) {
            out.println("Summary skipped: " + experimentsCsv + " does not exist.");
            return;
        }

        List<String> lines = Files.readAllLines(experimentsCsv, StandardCharsets.UTF_8);
        if (lines.size() < 2) {
            out.println("Summary skipped: " + experimentsCsv + " has no data rows.");
            return;
        }

        List<String> header = parseCsvRow(lines.get(0));
        Set<String> fixedCols = new LinkedHashSet<>(ExperimentsSchemaV1.fixedColumns());

        List<String> groupingCols = new ArrayList<>();
        List<String> metricCols = new ArrayList<>();
        for (String col : header) {
            if (fixedCols.contains(col)) {
                if (!NON_GROUPING_FIXED.contains(col)) {
                    groupingCols.add(col);
                }
            } else {
                metricCols.add(col);
            }
        }

        // group-key -> (metric-col -> accumulator)
        Map<String, Group> groups = new LinkedHashMap<>();

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line.isBlank()) continue;
            List<String> row = parseCsvRow(line);
            if (row.size() != header.size()) {
                // Skip malformed rows rather than crash
                continue;
            }

            Map<String, String> byCol = new LinkedHashMap<>();
            for (int c = 0; c < header.size(); c++) {
                byCol.put(header.get(c), row.get(c));
            }

            StringBuilder keyBuf = new StringBuilder();
            Map<String, String> keyMap = new LinkedHashMap<>();
            for (String gc : groupingCols) {
                String v = byCol.getOrDefault(gc, "");
                keyBuf.append(gc).append('=').append(v).append('\u001f');
                keyMap.put(gc, v);
            }
            String key = keyBuf.toString();

            Group g = groups.computeIfAbsent(key, k -> new Group(keyMap, metricCols));
            g.observedRows++;

            for (String mc : metricCols) {
                String raw = byCol.getOrDefault(mc, "");
                if (raw == null || raw.isBlank()) continue;
                try {
                    double val = Double.parseDouble(raw);
                    g.accumulate(mc, val);
                } catch (NumberFormatException ignored) {
                    // non-numeric metric cell; skip
                }
            }
        }

        if (groups.isEmpty()) {
            out.println("Summary skipped: no data rows in " + experimentsCsv + ".");
            return;
        }

        out.println();
        out.println("=== Summary (aggregated across repetitions) ===");
        out.println("Source: " + experimentsCsv);
        out.println();

        int groupIdx = 0;
        for (Group g : groups.values()) {
            groupIdx++;
            out.printf("Group %d  (N=%d rows)%n", groupIdx, g.observedRows);
            for (Map.Entry<String, String> e : g.keyMap.entrySet()) {
                out.printf("  %-20s %s%n", e.getKey(), e.getValue());
            }
            for (String mc : metricCols) {
                Accumulator acc = g.metrics.get(mc);
                if (acc == null || acc.count == 0) continue;
                double mean = acc.mean();
                double std = acc.sampleStddev();
                double cv = (mean != 0.0) ? (std / Math.abs(mean)) * 100.0 : 0.0;
                out.printf(Locale.ROOT,
                        "  %-40s  n=%d  mean=%.6g  std=%.6g  cv=%.2f%%%n",
                        mc, acc.count, mean, std, cv);
            }
            out.println();
        }
    }

    private static final class Group {
        final Map<String, String> keyMap;
        final Map<String, Accumulator> metrics;
        int observedRows;

        Group(Map<String, String> keyMap, List<String> metricCols) {
            this.keyMap = keyMap;
            this.metrics = new LinkedHashMap<>();
            for (String mc : metricCols) {
                metrics.put(mc, new Accumulator());
            }
        }

        void accumulate(String metricCol, double value) {
            metrics.get(metricCol).add(value);
        }
    }

    /** Welford's online mean/variance. */
    private static final class Accumulator {
        int count;
        double mean;
        double m2;

        void add(double x) {
            count++;
            double delta = x - mean;
            mean += delta / count;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }

        double mean() { return mean; }

        double sampleStddev() {
            return (count < 2) ? 0.0 : Math.sqrt(m2 / (count - 1));
        }
    }

    // ---------------------------
    // Minimal CSV row parser
    // ---------------------------
    // Handles:
    //  - double-quote-escaped fields (with ""-doubling)
    //  - commas inside quoted fields
    // Does NOT handle embedded newlines inside quoted fields, which the writer does not produce.
    private static List<String> parseCsvRow(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        int i = 0;
        while (i < line.length()) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i += 2;
                        continue;
                    }
                    inQuotes = false;
                    i++;
                    continue;
                }
                cur.append(c);
                i++;
            } else {
                if (c == ',') {
                    out.add(cur.toString());
                    cur.setLength(0);
                    i++;
                } else if (c == '"') {
                    inQuotes = true;
                    i++;
                } else {
                    cur.append(c);
                    i++;
                }
            }
        }
        out.add(cur.toString());
        return out;
    }
}
