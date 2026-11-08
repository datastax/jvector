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

import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import static org.junit.Assert.*;

public class QueryExecutionConfigTest {
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    @Test
    public void runConfigRejectsQueryDeliverySettings() throws Exception {
        var file = directory.newFile("run.yml").toPath();
        Files.writeString(file, "yamlSchemaVersion: 1\nqueryExecution:\n  mode: queued\n");
        assertThrows(org.yaml.snakeyaml.error.YAMLException.class, () -> RunConfig.load(file.toString()));
    }

    @Test
    public void schema2RejectsNestedAshDeliverySettings() throws Exception {
        var file = directory.newFile("index.yml").toPath();
        Files.writeString(file, "yamlSchemaVersion: 2\nonDiskIndexVersion: 7\ndataset: test\n"
                + "search:\n  compression:\n    - type: ASH\n      parameters:\n        queryExecution: queued\n");
        var error = assertThrows(IllegalArgumentException.class, () -> MultiConfig.getConfig(file.toString()));
        assertTrue(error.getMessage().contains("require schema 3"));
    }
}
