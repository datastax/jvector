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

import io.github.jbellis.jvector.example.benchmarks.Metric;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Append-only writer for experiments.csv inside a run directory.
 *
 * Header = fixed columns + output-key columns (Metric.key values).
 * Missing/non-applicable outputs are written as empty CSV fields.
 */
public final class ExperimentsCsvWriter {

    private final RunContext run;
    private final Path out;
    private final List<String> fixedColumns;
    private final List<String> outputKeyColumns;

    public ExperimentsCsvWriter(RunContext run,
                                List<String> fixedColumns,
                                List<String> outputKeyColumns) throws IOException {
        this.run = run;
        this.out = run.runDir().resolve("experiments.csv");
        this.fixedColumns = List.copyOf(fixedColumns);
        this.outputKeyColumns = List.copyOf(outputKeyColumns);

        Files.createDirectories(run.runDir());
        if (!Files.exists(out)) {
            writeHeader();
        }
    }

    private void writeHeader() throws IOException {
        List<String> cols = new ArrayList<>(fixedColumns.size() + outputKeyColumns.size());
        cols.addAll(fixedColumns);
        cols.addAll(outputKeyColumns);
        String header = String.join(",", cols) + "\n";
        Files.writeString(out, header, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    /**
     * Append one experiment row.
     *
     * @param fixedValues map for fixed columns (missing keys -> empty)
     * @param outputs benchmark stats/metrics/telemetry produced for this row (keyed by Metric.getKey())
     */
    public void appendRow(Map<String, String> fixedValues, List<Metric> outputs) throws IOException {
        // Build output key -> numeric string
        Map<String, String> outputValues = new HashMap<>();
        for (Metric m : outputs) {
            outputValues.put(m.getKey(), Double.toString(m.getValue()));
        }

        StringBuilder sb = new StringBuilder(1024);

        // fixed columns
        for (int i = 0; i < fixedColumns.size(); i++) {
            String col = fixedColumns.get(i);
            String v = fixedValues.getOrDefault(col, "");
            sb.append(csv(v));
            sb.append(',');
        }

        // output-key columns
        for (int i = 0; i < outputKeyColumns.size(); i++) {
            String key = outputKeyColumns.get(i);
            String v = outputValues.getOrDefault(key, "");
            sb.append(csv(v));
            if (i + 1 < outputKeyColumns.size()) sb.append(',');
        }
        sb.append('\n');

        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    }

    public Path path() {
        return out;
    }

    private static String csv(String s) {
        if (s == null) return "";
        boolean needsQuote = s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0;
        if (!needsQuote) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}
