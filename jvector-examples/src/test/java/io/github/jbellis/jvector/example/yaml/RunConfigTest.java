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
package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.example.benchmarks.WarmupQueries;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class RunConfigTest {
    private static final Path YAML_CONFIGS = Path.of("jvector-examples/yaml-configs");

    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    private RunConfig load(String measurement) throws Exception {
        var file = directory.newFile().toPath();
        Files.writeString(file, "yamlSchemaVersion: 1\nonDiskIndexVersion: 7\nbenchmarks:\n  throughput: [ AVG ]\n" + measurement);
        return RunConfig.load(file.toString());
    }

    @Test
    public void everyCheckedInRunConfigurationLoads() throws Exception {
        java.util.List<Path> files;
        try (Stream<Path> paths = Files.walk(YAML_CONFIGS)) {
            files = paths.filter(p -> p.getFileName().toString().matches("run.*\\.yml"))
                    .filter(p -> !p.startsWith(YAML_CONFIGS.resolve("index-parameters")))
                    .collect(Collectors.toList());
        }
        assertTrue(files.toString(), files.size() >= 6);
        for (var file : files) assertTrue(file.toString(), RunConfig.load(file.toString()).benchmarks != null);
    }

    @Test
    public void measurementIsOptionalAndDefaultsToTheBuiltInBehavior() throws Exception {
        assertNull(load("").measurement);
        var throughput = new Measurement.Throughput();
        assertEquals(WarmupQueries.RANDOM, throughput.warmup());
        assertEquals(3, throughput.warmupPasses());
        assertFalse(throughput.untilCompiled());
        assertEquals(3, throughput.passes());
        assertEquals(1.0, throughput.minSeconds(), 0);
        assertFalse(new Measurement().evictIndex());
    }

    @Test
    public void measurementSettingsParse() throws Exception {
        var run = load("measurement:\n  evictIndex: true\n  throughput:\n    warmup: base\n    warmupPasses: 2\n"
                + "    untilCompiled: true\n    passes: 1\n    minSeconds: 0\n");
        assertTrue(run.measurement.evictIndex());
        assertEquals(WarmupQueries.BASE, run.measurement.throughput.warmup());
        assertEquals(2, run.measurement.throughput.warmupPasses());
        assertTrue(run.measurement.throughput.untilCompiled());
        assertEquals(1, run.measurement.throughput.passes());
        assertEquals(0.0, run.measurement.throughput.minSeconds(), 0);
    }

    @Test
    public void invalidMeasurementSettingsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> load("measurement:\n  throughput:\n    warmup: queries\n"));
        assertThrows(IllegalArgumentException.class, () -> load("measurement:\n  throughput:\n    passes: 0\n"));
        assertThrows(IllegalArgumentException.class, () -> load("measurement:\n  throughput:\n    warmupPasses: -1\n"));
        assertThrows(IllegalArgumentException.class, () -> load("measurement:\n  throughput:\n    minSeconds: -1\n"));
        assertThrows(IllegalArgumentException.class,
                () -> load("measurement:\n  throughput:\n    warmup: none\n    untilCompiled: true\n"));
        assertThrows(org.yaml.snakeyaml.error.YAMLException.class, () -> load("measurement:\n  unknown: 1\n"));
    }
}
