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
                + "search:\n  queryExecution: queued\n  compression:\n    - type: ASH\n");
        var error = assertThrows(IllegalArgumentException.class, () -> MultiConfig.getConfig(file.toString()));
        assertTrue(error.getMessage().contains("require schema 3"));
    }

    private MultiConfig schema3(String construction, String search) throws Exception {
        var file = directory.newFile().toPath();
        Files.writeString(file, String.join("\n",
                "yamlSchemaVersion: 3", "onDiskIndexVersion: 7", "dataset: test",
                "construction:", construction,
                "  outDegree: [8]", "  efConstruction: [32]", "  neighborOverflow: [1.2f]",
                "  addHierarchy: [No]", "  refineFinalGraph: [No]",
                "  compression:", "    - type: None", "  reranking:", "    type: None",
                "search:", "  topKOverquery:", "    10: [1.0]", "  useSearchPruning: [No]", search, ""));
        return MultiConfig.getConfig(file.toString());
    }

    private static final String ON_DISK = "  type: onDisk";
    private static final String ASH = "  compression:\n    - type: ASH\n      parameters:\n"
            + "        optimizer: RANDOM\n        encodedBits: 72\n        bitsPerDimension: 2";

    @Test
    public void queuedExecutionAndRequestsPerWorkerAreSearchSettings() throws Exception {
        var config = schema3(ON_DISK, "  queryExecution: queued\n  requestsPerWorker: 1\n" + ASH + "\n        maxBatchSize: 1");
        assertEquals("queued", config.search.queryExecution());
        assertEquals(Integer.valueOf(1), config.search.requestsPerWorker);
        // Queued execution is not specific to ASH.
        var pq = schema3(ON_DISK, "  queryExecution: queued\n  compression:\n    - type: PQ\n      parameters: {m: 4, k: 256}");
        assertEquals("queued", pq.search.queryExecution());
        assertEquals("direct", schema3(ON_DISK, ASH).search.queryExecution());
    }

    @Test
    public void queuedOnlySettingsRequireQueuedExecution() throws Exception {
        var perWorker = assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK, "  requestsPerWorker: 1\n" + ASH));
        assertTrue(perWorker.getMessage().contains("requestsPerWorker requires queued"));
        var batch = assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK, ASH + "\n        maxBatchSize: 4"));
        assertTrue(batch.getMessage().contains("maxBatchSize requires"));
        assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK, "  queryExecution: queued\n  requestsPerWorker: 0\n" + ASH));
        assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK, "  queryExecution: batched\n" + ASH));
        // Execution settings are no longer ASH compression parameters.
        assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK, ASH + "\n        queryExecution: queued"));
    }

    @Test
    public void pageAlignmentIsAnOnDiskConstructionSetting() throws Exception {
        assertEquals(Integer.valueOf(4096), schema3(ON_DISK + "\n  pageAlignment: 4096", ASH).construction.pageAlignment);
        assertThrows(IllegalArgumentException.class, () -> schema3("  type: inMemory\n  pageAlignment: 4096", ASH));
        assertThrows(IllegalArgumentException.class, () -> schema3(ON_DISK + "\n  pageAlignment: 1000", ASH));
    }
}
