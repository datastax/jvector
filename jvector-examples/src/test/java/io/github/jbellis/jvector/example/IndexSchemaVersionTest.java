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
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/** Index YAML schema versions BenchYAML accepts. */
public class IndexSchemaVersionTest {
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    /** The layout of the dataset defaults in earlier releases, which declared schema 1. */
    private static final String BODY = "onDiskIndexVersion: 7\ndataset: test\n"
            + "construction:\n  outDegree: [32]\n  efConstruction: [100]\n  neighborOverflow: [1.2f]\n"
            + "  addHierarchy: [Yes]\n  refineFinalGraph: [Yes]\n  fusedGraph: [No]\n"
            + "  compression:\n    - type: PQ\n      parameters:\n        mFactor: 8\n"
            + "  reranking:\n    - NVQ\n  useSavedIndexIfExists: No\n"
            + "search:\n  topKOverquery:\n    10: [1.0, 2.0]\n  useSearchPruning: [No]\n"
            + "  compression:\n    - type: PQ\n      parameters:\n        mFactor: 8\n";

    private MultiConfig load(int schema) throws Exception {
        var file = new File(directory.getRoot(), "schema" + schema + ".yml");
        Files.writeString(file.toPath(), "yamlSchemaVersion: " + schema + "\n" + BODY);
        return MultiConfig.getConfig(file.getPath());
    }

    private static String plan(MultiConfig config) {
        var spec = GridSpec.fromConfig(config);
        return spec.buildPoints().stream().map(p -> p.reportedParameters().toString()).collect(Collectors.joining(";"))
                + " | " + spec.search.cases().size() + " search cases";
    }

    @Test
    public void schemaOneFilesFromEarlierReleasesLoadLikeSchemaTwo() throws Exception {
        var one = load(1);
        assertEquals(1, one.getYamlSchemaVersion());
        assertEquals(plan(load(2)), plan(one));
    }

    @Test
    public void versionsOutsideOneToThreeAreRejected() {
        for (int unsupported : new int[] {0, 4}) {
            var e = assertThrows(IllegalArgumentException.class, () -> load(unsupported));
            assertTrue(e.getMessage(), e.getMessage().startsWith("Unsupported yamlSchemaVersion=" + unsupported));
        }
    }
}
