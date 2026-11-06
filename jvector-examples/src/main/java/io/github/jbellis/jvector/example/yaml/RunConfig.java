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

import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Map;

/**
 * Run-level configuration loaded from yaml-configs/run-config.yml.
 *
 * This controls:
 * - benchmark computation (benchmarks)
 * - console projection (console)
 * - experiments.csv logging (logging)
 *
 */
public class RunConfig {
    private static final String defaultDirectory = "jvector-examples/yaml-configs/";
    private static final String defaultRunFile = "run-config.yml";

    public int yamlSchemaVersion;
    public int onDiskIndexVersion;

    /** Optional ASH run settings; omission preserves JVM properties and library defaults. */
    public Ash ash;

    public static class Ash {
        public String queryProjection;
    }

    public static final String ASH_QUERY_PROJECTION_PROPERTY = "jvector.ash.queryProjection";

    /** Applies only the optional setting for this invocation and restores it on exit. */
    public static final class AshQueryProjectionSetting implements AutoCloseable {
        private final String original = System.getProperty(ASH_QUERY_PROJECTION_PROPERTY);
        private boolean applied;

        public void apply(RunConfig config) {
            String yaml = config.ash == null ? null : config.ash.queryProjection;
            if (yaml != null) validatePrecision(yaml);
            String selected = original != null ? original : yaml;
            if (selected != null) validatePrecision(selected);
            if (original == null && yaml != null) {
                System.setProperty(ASH_QUERY_PROJECTION_PROPERTY, yaml);
                applied = true;
            }
        }

        private static void validatePrecision(String value) {
            if (!List.of("auto", "fp32", "fp16", "int8").contains(value))
                throw new IllegalArgumentException("ash.queryProjection must be auto, fp32, fp16 or int8: " + value);
        }

        @Override
        public void close() {
            if (applied) {
                System.clearProperty(ASH_QUERY_PROJECTION_PROPERTY);
                applied = false;
            }
        }
    }

    /** Query delivery policy; omitted settings preserve the synchronous benchmark. */
    public QueryExecution queryExecution = new QueryExecution();

    public static class QueryExecution {
        public String mode = "direct";
        public int maxBatchSize = 8;
    }

    /** Benchmark computation policy (what is computed). */
    public Map<String, List<String>> benchmarks;

    /** Console selection (subset + extra metrics). */
    public BenchmarkSelection console;

    /** Logging selection and run-level logging settings. */
    public RunLogging logging;

    /** Select an explicit run YAML without editing the checkout's default file. */
    public static RunConfig loadDefault() throws FileNotFoundException {
        return load(System.getProperty("jvector.bench.runConfig", defaultDirectory + defaultRunFile));
    }

    public static RunConfig load(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        if (!configFile.exists() && !configName.startsWith("/") && !configName.contains(":")) {
            configFile = new File(defaultDirectory + configName);
        }
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }
        try (InputStream inputStream = new FileInputStream(configFile)) {
            Yaml yaml = new Yaml();
            RunConfig config = yaml.loadAs(inputStream, RunConfig.class);
            if (config == null) throw new IllegalArgumentException("Empty run configuration: " + configFile);
            return config;
        } catch (FileNotFoundException failure) {
            throw failure;
        } catch (IOException failure) {
            throw new UncheckedIOException("Cannot read run configuration: " + configFile, failure);
        }
    }

    /** Run-level logging settings + selection for experiments.csv. */
    public static class RunLogging extends BenchmarkSelection {
        public String dir;       // e.g. "logging"
        public String runId;     // optional template; supports "{ts}" (UTC). Default: "{ts}"
        public String jvectorRef; // tag/commit to record
        public String type;      // e.g. "csv"
    }
}
