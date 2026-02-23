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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Schema definition for experiments.csv (v1).
 *
 * Fixed columns are the join keys + resolved index/search parameters that appear on every row.
 * Output-key columns ({@code Metric.key} strings) are appended after these and are computed per run.
 */
public final class ExperimentsSchemaV1 {
    private ExperimentsSchemaV1() {}

    public static List<String> fixedColumns() {
        return List.of(
                "schema_version",
                "run_id",
                "run_uuid",
                "system_id",
                "dataset_name",
                "M",
                "efConstruction",
                "neighborOverflow",
                "addHierarchy",
                "refineFinalGraph",
                "feature_set",
                "usePruning",
                "topK",
                "overquery",
                "rerankK"
        );
    }

    /**
     * Build the fixed-column value map for one experiments.csv row.
     *
     * Grid/orchestrators supply raw values; this method owns column naming consistency.
     */
    public static Map<String, String> fixedValues(RunContext run,
                                                  String datasetName,
                                                  int M,
                                                  int efConstruction,
                                                  float neighborOverflow,
                                                  boolean addHierarchy,
                                                  boolean refineFinalGraph,
                                                  Set<FeatureId> featureSet,
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

        fixed.put("M", Integer.toString(M));
        fixed.put("efConstruction", Integer.toString(efConstruction));
        fixed.put("neighborOverflow", Float.toString(neighborOverflow));
        fixed.put("addHierarchy", Boolean.toString(addHierarchy));
        fixed.put("refineFinalGraph", Boolean.toString(refineFinalGraph));
        fixed.put("feature_set", featureSet == null ? "" : featureSet.toString());

        fixed.put("usePruning", Boolean.toString(usePruning));
        fixed.put("topK", Integer.toString(topK));
        fixed.put("overquery", Double.toString(overquery));
        fixed.put("rerankK", Integer.toString(rerankK));

        return fixed;
    }
}
