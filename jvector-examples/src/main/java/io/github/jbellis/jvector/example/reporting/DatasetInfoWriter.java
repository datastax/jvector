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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.LinkedHashMap;

/**
 * Writes dataset_info.csv for a run (one row per dataset used).
 *
 * Incremental API: open once per run, then register datasets as they are actually loaded.
 * This avoids preloading datasets just to collect metadata.
 */
public final class DatasetInfoWriter {

    private final RunContext run;
    private final Path out;
    private final Set<String> seenDatasetNames = new HashSet<>();

    private DatasetInfoWriter(RunContext run, Path out) {
        this.run = run;
        this.out = out;
    }

    /** Open writer and create dataset_info.csv with header if needed. */
    public static DatasetInfoWriter open(RunContext run) throws IOException {
        Files.createDirectories(run.runDir());
        Path out = run.runDir().resolve("dataset_info.csv");

        if (!Files.exists(out)) {
            String header = String.join(",",
                    "schema_version",
                    "run_id",
                    "dataset_name",
                    "dataset_id",
                    "base_path",
                    "query_path",
                    "ground_truth_path",
                    "base_count",
                    "query_count",
                    "ground_truth_count",
                    "dimension",
                    "similarity_function"
            ) + "\n";
            Files.writeString(out, header, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }

        return new DatasetInfoWriter(run, out);
    }

    /** One dataset row for dataset_info.csv (paths are expected to be absolute if available). */
    public static final class Row {
        public final String datasetName;

        public final String basePath;
        public final String queryPath;
        public final String groundTruthPath;

        public final long baseCount;
        public final long queryCount;
        public final long groundTruthCount;

        public final int dimension;
        public final String similarityFunction;

        public Row(String datasetName,
                   String basePath,
                   String queryPath,
                   String groundTruthPath,
                   long baseCount,
                   long queryCount,
                   long groundTruthCount,
                   int dimension,
                   String similarityFunction) {
            this.datasetName = Objects.requireNonNull(datasetName, "datasetName");
            this.basePath = basePath == null ? "" : basePath;
            this.queryPath = queryPath == null ? "" : queryPath;
            this.groundTruthPath = groundTruthPath == null ? "" : groundTruthPath;
            this.baseCount = baseCount;
            this.queryCount = queryCount;
            this.groundTruthCount = groundTruthCount;
            this.dimension = dimension;
            this.similarityFunction = similarityFunction == null ? "" : similarityFunction;
        }
    }

    /**
     * Convenience factory that builds a dataset_info row from an already-loaded DataSet.
     * Callers supply only the dataset name and resolved base/query/gt paths; counts, dimension,
     * and similarity are derived from the DataSet to keep schema population consistent.
     */
    public static Row fromDataSet(String datasetName,
                                  String basePath,
                                  String queryPath,
                                  String groundTruthPath,
                                  DataSet ds) {
        return new Row(
                datasetName,
                basePath,
                queryPath,
                groundTruthPath,
                ds.getBaseVectors().size(),
                ds.getQueryVectors().size(),
                ds.getGroundTruth().size(),
                ds.getDimension(),
                ds.getSimilarityFunction().toString()
        );
    }

    /**
     * Register a dataset (append one row) the first time it is seen.
     *
     * @return dataset_id for joining with experiments.csv
     */
    public String register(Row r) throws IOException {
        if (!seenDatasetNames.add(r.datasetName)) {
            // already written this dataset in this run
            return computeDatasetId(r);
        }

        String datasetId = computeDatasetId(r);

        StringBuilder sb = new StringBuilder(512);
        sb.append(Integer.toString(run.schemaVersion())).append(',')
                .append(csv(run.runId())).append(',')
                .append(csv(r.datasetName)).append(',')
                .append(csv(datasetId)).append(',')
                .append(csv(r.basePath)).append(',')
                .append(csv(r.queryPath)).append(',')
                .append(csv(r.groundTruthPath)).append(',')
                .append(Long.toString(r.baseCount)).append(',')
                .append(Long.toString(r.queryCount)).append(',')
                .append(Long.toString(r.groundTruthCount)).append(',')
                .append(Integer.toString(r.dimension)).append(',')
                .append(csv(r.similarityFunction))
                .append('\n');

        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        return datasetId;
    }

    private static String computeDatasetId(Row r) {
        // Canonical field map (stable order)
        Map<String, String> m = new LinkedHashMap<>();
        m.put("dataset_name", r.datasetName);
        m.put("base_path", r.basePath);
        m.put("query_path", r.queryPath);
        m.put("ground_truth_path", r.groundTruthPath);
        m.put("base_count", Long.toString(r.baseCount));
        m.put("query_count", Long.toString(r.queryCount));
        m.put("ground_truth_count", Long.toString(r.groundTruthCount));
        m.put("dimension", Integer.toString(r.dimension));
        m.put("similarity_function", r.similarityFunction);

        StringBuilder canon = new StringBuilder(512);
        for (var e : m.entrySet()) {
            canon.append(e.getKey()).append('=').append(e.getValue()).append('\n');
        }
        return Hashing.shortSha256Hex(canon.toString());
    }

    private static String csv(String s) {
        if (s == null) return "";
        boolean needsQuote = s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0;
        if (!needsQuote) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}
