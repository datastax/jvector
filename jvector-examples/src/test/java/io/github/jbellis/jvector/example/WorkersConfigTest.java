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

import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Files;
import java.util.List;

import static org.junit.Assert.*;

/** construction.workers and search.workers in schema 3 index YAML. */
public class WorkersConfigTest {
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    private MultiConfig load(int schema, String constructionWorkers, String searchWorkers) throws Exception {
        var file = directory.newFile().toPath();
        Files.writeString(file, "yamlSchemaVersion: " + schema + "\nonDiskIndexVersion: 7\ndataset: test\n"
                + "construction:\n" + (schema == 3 ? "  type: onDisk\n" : "")
                + "  outDegree: [16]\n  efConstruction: [50]\n"
                + "  neighborOverflow: [1.2]\n  addHierarchy: [No]\n  refineFinalGraph: [Yes]\n"
                + "  compression:\n    - type: None\n" + (schema == 3 ? "  reranking:\n    type: None\n" : "  reranking: [NVQ]\n")
                + "  useSavedIndexIfExists: No\n"
                + (constructionWorkers == null ? "" : "  workers: " + constructionWorkers + "\n")
                + "search:\n  topKOverquery:\n    10: [1.0]\n  useSearchPruning: [No]\n"
                + (searchWorkers == null ? "" : "  workers: " + searchWorkers + "\n")
                + "  compression:\n    - type: None\n");
        return MultiConfig.getConfig(file.toString());
    }

    @Test
    public void workersResolveToCountsAndMaxIsThePhysicalCoreCount() throws Exception {
        var config = load(3, "max", "[1, 8, max]");
        int max = PhysicalCoreExecutor.getPhysicalCoreCount();
        assertEquals(Integer.valueOf(max), config.construction.resolvedWorkers());
        assertEquals(List.of(1, 8, max), config.search.resolvedWorkers());

        var spec = GridSpec.fromConfig(config);
        assertEquals(Integer.valueOf(max), spec.buildPoints().get(0).workers);
        // One search case per compression entry; it queries one index with each worker count.
        var cases = spec.search.cases();
        assertEquals(1, cases.size());
        assertEquals(List.of(1, 8, max), cases.get(0).workerCounts());
    }

    @Test
    public void omittedWorkersKeepThePlatformDefaults() throws Exception {
        var spec = GridSpec.fromConfig(load(3, null, null));
        assertNull(spec.buildPoints().get(0).workers);
        assertEquals(1, spec.search.cases().size());
        assertEquals(java.util.Collections.singletonList(null), spec.search.cases().get(0).workerCounts());
    }

    @Test
    public void invalidWorkerCountsNameTheFieldAndTheRule() {
        var nl = System.lineSeparator();
        var zero = assertThrows(IllegalArgumentException.class, () -> load(3, "0", null));
        assertTrue(zero.getMessage(), zero.getMessage().endsWith("construction.workers: 0 is not a worker count." + nl
                + "Use a positive integer, or max for the platform's physical cores."));
        var word = assertThrows(IllegalArgumentException.class, () -> load(3, null, "[1, all]"));
        assertTrue(word.getMessage(), word.getMessage().endsWith("search.workers: all is not a worker count." + nl
                + "Use a positive integer, or max for the platform's physical cores."));
        var empty = assertThrows(IllegalArgumentException.class, () -> load(3, null, "[]"));
        assertTrue(empty.getMessage(), empty.getMessage().endsWith("search.workers must list at least one worker count"));
    }

    @Test
    public void workersRequireSchema3() {
        var error = assertThrows(IllegalArgumentException.class, () -> load(2, "max", null));
        assertTrue(error.getMessage(), error.getMessage().endsWith("construction.workers and search.workers require schema 3"));
    }
}
