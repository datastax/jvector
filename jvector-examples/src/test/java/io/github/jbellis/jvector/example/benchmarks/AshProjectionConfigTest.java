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
        original = System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY);
        System.clearProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY);
    }
    @After public void restoreProperty() {
        if (original == null) System.clearProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY);
        else System.setProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY, original);
    }
    private RunConfig yaml(String body) throws Exception {
        var file = temporary.newFile().toPath();
        Files.writeString(file, "yamlSchemaVersion: 1\n" + body);
        return RunConfig.load(file.toString());
    }
    @Test public void omissionDoesNotSetAPropertyOrChangeExistingDefaults() throws Exception {
        var config = yaml("");
        assertNull(config.ash);
        try (var setting = new RunConfig.AshQueryProjectionSetting()) {
            setting.apply(config);
            assertNull(System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
            assertEquals("direct", config.queryExecution.mode);
            assertEquals(8, config.queryExecution.maxBatchSize);
        }
        assertNull(System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void eachYamlSelectionIsAppliedAndRestored() throws Exception {
        for (String mode : new String[]{"auto", "fp32", "fp16", "int8"}) {
            var config = yaml("ash:\n  queryProjection: " + mode + "\n");
            try (var setting = new RunConfig.AshQueryProjectionSetting()) {
                setting.apply(config);
                assertEquals(mode, System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
            }
            assertNull(System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
        }
    }
    @Test public void explicitJvmSettingOverridesYamlAndIsPreserved() throws Exception {
        System.setProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY, "fp32");
        try (var setting = new RunConfig.AshQueryProjectionSetting()) {
            setting.apply(yaml("ash:\n  queryProjection: int8\n"));
            assertEquals("fp32", System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
        }
        assertEquals("fp32", System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void failedRunsRestoreOnlyTheirTemporarySetting() throws Exception {
        var config = yaml("ash:\n  queryProjection: int8\n");
        assertThrows(IllegalStateException.class, () -> {
            try (var setting = new RunConfig.AshQueryProjectionSetting()) {
                setting.apply(config);
                throw new IllegalStateException("failed run");
            }
        });
        assertNull(System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
    }
    @Test public void invalidNewValuesFailWithoutChangingTheProperty() throws Exception {
        var bad = yaml("ash:\n  queryProjection: pf32\n");
        try (var setting = new RunConfig.AshQueryProjectionSetting()) {
            assertThrows(IllegalArgumentException.class, () -> setting.apply(bad));
        }
        assertNull(System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
        System.setProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY, "unknown");
        try (var setting = new RunConfig.AshQueryProjectionSetting()) {
            assertThrows(IllegalArgumentException.class, () -> setting.apply(new RunConfig()));
        }
        assertEquals("unknown", System.getProperty(RunConfig.ASH_QUERY_PROJECTION_PROPERTY));
    }
}
