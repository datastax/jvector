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
package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.reporting.SearchReportingCatalog;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import static org.junit.Assert.*;

public class QueryExecutionConfigTest {
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    @Test
    public void defaultsPreserveDirectExecutionAndQueuedModeIsSelectedInYaml() throws Exception {
        var file = directory.getRoot().toPath().resolve("run.yml");
        Files.writeString(file, "yamlSchemaVersion: 1\nqueryExecution:\n  mode: queued\n  maxBatchSize: 8\n");
        var config = RunConfig.load(file.toString());
        config.benchmarks = SearchReportingCatalog.defaultComputeBenchmarks();
        var artifacts = RunArtifacts.open(config, List.of());
        assertEquals("queued", artifacts.queryExecutionMode());
        assertEquals(8, artifacts.queryBatchSize());
        assertEquals("direct", RunArtifacts.disabled().queryExecutionMode());
    }

    @Test
    public void invalidExecutionSettingsFailBeforeCollectingResults() {
        var config = new RunConfig();
        config.benchmarks = SearchReportingCatalog.defaultComputeBenchmarks();
        config.queryExecution.mode = "unknown";
        assertThrows(IllegalArgumentException.class, () -> RunArtifacts.open(config, List.of()));
        config.queryExecution.mode = "queued";
        config.queryExecution.maxBatchSize = 0;
        assertThrows(IllegalArgumentException.class, () -> RunArtifacts.open(config, List.of()));
    }
}
