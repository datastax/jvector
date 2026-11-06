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

import io.github.jbellis.jvector.example.reporting.ExperimentsSchemaV1;
import io.github.jbellis.jvector.example.reporting.RunContext;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import org.junit.Test;

import java.nio.file.Path;
import java.time.Instant;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

import static org.junit.Assert.*;

/** The build parameters every output reports, and the experiments.csv columns they feed. */
public class TestReportedParameters {
    private static BuildConfig build() {
        return BuildConfig.builder()
                .M(32)
                .efConstruction(100)
                .neighborOverflow(1.2f)
                .addHierarchy(true)
                .refineFinalGraph(false)
                .build();
    }

    private static RunContext run() {
        return new RunContext("run", UUID.fromString("00000000-0000-0000-0000-000000000001"), Instant.EPOCH,
                Path.of("unused"), "system");
    }

    @Test
    public void reportedParametersKeepTheirNamesAndOrder() {
        var params = build().reportedParameters();
        // Console, BenchResult and experiments.csv consumers key on these names.
        assertEquals(List.of("M", "efConstruction", "neighborOverflow", "addHierarchy", "refineFinalGraph"),
                List.copyOf(params.keySet()));
        assertEquals(32, params.get("M"));
        assertEquals(1.2f, params.get("neighborOverflow"));
        assertEquals(false, params.get("refineFinalGraph"));
        assertThrows(UnsupportedOperationException.class, () -> params.put("alpha", 1.2f));
    }

    @Test
    public void experimentsColumnsAreUnchanged() {
        assertEquals(List.of("schema_version", "run_id", "run_uuid", "system_id", "dataset_name",
                        "M", "efConstruction", "neighborOverflow", "addHierarchy", "refineFinalGraph",
                        "feature_set", "build_compressor", "search_compressor",
                        "usePruning", "topK", "overquery", "rerankK"),
                ExperimentsSchemaV1.fixedColumns());
    }

    @Test
    @SuppressWarnings("deprecation")
    public void mapAndPositionalRowsAreIdentical() {
        var features = EnumSet.of(FeatureId.NVQ_VECTORS);
        var fromMap = ExperimentsSchemaV1.fixedValues(run(), "ds", build().reportedParameters(), features,
                "PQ, 16", "None", true, 10, 2.0, 20);
        var positional = ExperimentsSchemaV1.fixedValues(run(), "ds", 32, 100, 1.2f, true, false, features,
                "PQ, 16", "None", true, 10, 2.0, 20);
        assertEquals(positional, fromMap);
        assertEquals("1.2", fromMap.get("neighborOverflow"));
        assertEquals("\"PQ, 16\"", fromMap.get("build_compressor"));
    }

    @Test
    public void parametersWithoutAColumnAreNotWritten() {
        var params = new HashMap<String, Object>(build().reportedParameters());
        params.put("newKnob", 7);
        var row = ExperimentsSchemaV1.fixedValues(run(), "ds", params, null, "None", "None", false, 10, 1.0, 10);
        assertFalse(row.containsKey("newKnob"));
        assertTrue(ExperimentsSchemaV1.fixedColumns().containsAll(row.keySet()));
    }

    @Test
    public void aMissingColumnIsRejected() {
        var params = new HashMap<String, Object>(build().reportedParameters());
        params.remove("efConstruction");
        var e = assertThrows(IllegalArgumentException.class, () -> ExperimentsSchemaV1.fixedValues(
                run(), "ds", params, null, "None", "None", false, 10, 1.0, 10));
        assertTrue(e.getMessage(), e.getMessage().contains("efConstruction"));
    }
}
