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

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.Set;

/**
 * Schema definition for experiments.csv (v1).
 *
 * Fixed columns are the join keys + resolved index/search parameters that appear on every row.
 * Output-key columns ({@code Metric.key} strings) are appended after these and are computed per run.
 */
public final class ExperimentsSchemaV1 {
    private static final Pattern CSV_SPECIAL_CHARS = Pattern.compile("[\\r\\n\",]");

    private ExperimentsSchemaV1() {}

    static String csvEscape(String s) {
        if (s == null) return "";

        // Use a Matcher to check for existence
        if (!CSV_SPECIAL_CHARS.matcher(s).find()) {
            return s;
        }

        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    public static List<String> fixedColumns() {
        List<String> columns = new ArrayList<>(List.of(
                "schema_version",
                "run_id",
                "run_uuid",
                "system_id",
                "dataset_name"
        ));
        columns.addAll(BUILD_PARAMETER_COLUMNS);
        columns.addAll(List.of(
                "feature_set",
                "build_compressor",
                "search_compressor",
                "usePruning",
                "topK",
                "overquery",
                "rerankK"
        ));
        return List.copyOf(columns);
    }

    /**
     * The construction-parameter columns, in column order. Each names a key of the build parameters passed
     * to {@link #fixedValues(RunContext, String, Map, Set, String, String, boolean, int, double, int)}.
     * Adding one changes the schema.
     */
    private static final List<String> BUILD_PARAMETER_COLUMNS = List.of(
            "M",
            "efConstruction",
            "neighborOverflow",
            "addHierarchy",
            "refineFinalGraph"
    );

    /**
     * Build the fixed-column value map for one experiments.csv row.
     *
     * Grid/orchestrators supply raw values; this method owns column naming consistency.
     *
     * @param buildParameters construction parameters by name; every build-parameter column must be present,
     *                        and parameters without a column are not written
     * @throws IllegalArgumentException if a build-parameter column is missing from {@code buildParameters}
     */
    public static Map<String, String> fixedValues(RunContext run,
                                                  String datasetName,
                                                  Map<String, ?> buildParameters,
                                                  Set<FeatureId> featureSet,
                                                  String buildCompressorString,
                                                  String searchCompressorString,
                                                  boolean usePruning,
                                                  int topK,
                                                  double overquery,
                                                  int rerankK) {
        Map<String, String> fixed = new HashMap<>();
        fixed.put("schema_version", Integer.toString(run.schemaVersion()));
        fixed.put("run_id", run.runId());
        fixed.put("run_uuid", run.runUuid().toString());
        fixed.put("system_id", run.systemId());
        fixed.put("dataset_name", datasetName);

        for (String column : BUILD_PARAMETER_COLUMNS) {
            if (!buildParameters.containsKey(column)) {
                throw new IllegalArgumentException("Missing build parameter for experiments.csv column " + column);
            }
            fixed.put(column, String.valueOf(buildParameters.get(column)));
        }
        fixed.put("feature_set", featureSet == null ? "" : featureSet.toString());
        fixed.put("build_compressor", csvEscape(buildCompressorString));
        fixed.put("search_compressor", csvEscape(searchCompressorString));

        fixed.put("usePruning", Boolean.toString(usePruning));
        fixed.put("topK", Integer.toString(topK));
        fixed.put("overquery", Double.toString(overquery));
        fixed.put("rerankK", Integer.toString(rerankK));

        return fixed;
    }

    /**
     * @deprecated Use {@link #fixedValues(RunContext, String, Map, Set, String, String, boolean, int, double, int)},
     * which takes the construction parameters by name.
     */
    @Deprecated
    public static Map<String, String> fixedValues(RunContext run,
                                                  String datasetName,
                                                  int M,
                                                  int efConstruction,
                                                  float neighborOverflow,
                                                  boolean addHierarchy,
                                                  boolean refineFinalGraph,
                                                  Set<FeatureId> featureSet,
                                                  String buildCompressorString,
                                                  String searchCompressorString,
                                                  boolean usePruning,
                                                  int topK,
                                                  double overquery,
                                                  int rerankK) {
        Map<String, Object> buildParameters = new LinkedHashMap<>();
        buildParameters.put("M", M);
        buildParameters.put("efConstruction", efConstruction);
        buildParameters.put("neighborOverflow", neighborOverflow);
        buildParameters.put("addHierarchy", addHierarchy);
        buildParameters.put("refineFinalGraph", refineFinalGraph);
        return fixedValues(run, datasetName, buildParameters, featureSet, buildCompressorString,
                searchCompressorString, usePruning, topK, overquery, rerankK);
    }
}
