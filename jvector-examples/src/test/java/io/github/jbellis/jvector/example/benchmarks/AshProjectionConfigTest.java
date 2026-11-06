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

import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import java.util.List;
import org.junit.Before;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import static org.junit.Assert.*;

public class AshProjectionConfigTest {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();
    private String original;
    @Before public void isolateProperty() {
        original = System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
        System.clearProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
    }
    @After public void restoreProperty() {
        if (original == null) System.clearProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
        else System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, original);
    }
    private MultiConfig yaml(String body) throws Exception {
        var file = temporary.newFile().toPath();
        Files.writeString(file, "yamlSchemaVersion: 2\nonDiskIndexVersion: 7\ndataset: precision-test\n"
                + "construction:\n  compression:\n    - type: ASH\n      parameters:\n        encodedBits: 48\n        optimizer: RANDOM\n        bitsPerDimension: 2\n"
                + (body.isEmpty() ? "search: {}\n" : body));
        return MultiConfig.getConfig(file.toString());
    }
    @Test public void omissionDoesNotSetAPropertyOrChangeExistingDefaults() throws Exception {
        var config = yaml("");
        assertNull(config.search.queryProjection);
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            setting.apply(List.of(config));
            assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
            assertEquals("direct", new RunConfig().queryExecution.mode);
            assertEquals(8, new RunConfig().queryExecution.maxBatchSize);
        }
        assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void eachYamlSelectionIsAppliedAndRestored() throws Exception {
        for (String mode : new String[]{"auto", "fp32", "fp16", "int8"}) {
            var config = yaml("search:\n  queryProjection: " + mode + "\n");
            try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
                setting.apply(List.of(config));
                assertEquals(mode, System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
            }
            assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
        }
    }
    @Test public void explicitJvmSettingOverridesYamlAndIsPreserved() throws Exception {
        System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, "fp32");
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            setting.apply(List.of(yaml("search:\n  queryProjection: int8\n")));
            assertEquals("fp32", System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
        }
        assertEquals("fp32", System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void failedRunsRestoreOnlyTheirTemporarySetting() throws Exception {
        var config = yaml("search:\n  queryProjection: int8\n");
        assertThrows(IllegalStateException.class, () -> {
            try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
                setting.apply(List.of(config));
                throw new IllegalStateException("failed run");
            }
        });
        assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void invalidNewValuesFailWithoutChangingTheProperty() throws Exception {
        var bad = yaml("search:\n  queryProjection: pf32\n");
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            assertThrows(IllegalArgumentException.class, () -> setting.apply(List.of(bad)));
        }
        assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
        System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, "unknown");
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            assertThrows(IllegalArgumentException.class, () -> setting.apply(List.of(yaml(""))));
        }
        assertEquals("unknown", System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void conflictingAshPrecisionsFailBeforeModelsAreLoaded() throws Exception {
        var first = yaml("search:\n  queryProjection: int8\n");
        var second = yaml("search:\n  queryProjection: fp32\n");
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            assertThrows(IllegalArgumentException.class, () -> setting.apply(List.of(first,second)));
        }
        assertNull(System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void runConfigRejectsAlgorithmParameters() throws Exception {
        var file = temporary.newFile().toPath();
        Files.writeString(file,"yamlSchemaVersion: 1\nash:\n  queryProjection: int8\n");
        assertThrows(org.yaml.snakeyaml.error.YAMLException.class, () -> RunConfig.load(file.toString()));
    }
    @Test public void pqConfigurationDoesNotOverrideAshSelection() throws Exception {
        var ash = yaml("search:\n  queryProjection: int8\n");
        var pq = yaml("");
        pq.construction.compression.get(0).type = "PQ";
        try (var setting = new SearchParameters.AshQueryProjectionSetting()) {
            setting.apply(List.of(ash,pq));
            assertEquals("int8",System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY));
        }
    }

}
