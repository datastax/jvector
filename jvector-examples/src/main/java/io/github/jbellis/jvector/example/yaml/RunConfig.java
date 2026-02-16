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
import java.util.List;
import java.util.Map;

/**
 * Run-level configuration loaded from yaml-configs/run.yml.
 *
 * This controls:
 * - benchmark computation (benchmarks)
 * - console projection (console)
 * - experiments.csv logging (logging)
 *
 */
public class RunConfig {
    private static final String defaultDirectory = "jvector-examples/yaml-configs/";
    private static final String defaultRunFile = "run.yml";

    public int yamlSchemaVersion;
    public int onDiskIndexVersion;

    /** Benchmark computation policy (what is computed). */
    public Map<String, List<String>> benchmarks;

    /** Console selection (subset + extra metrics). */
    public BenchmarkSelection console;

    /** Logging selection and run-level logging settings. */
    public RunLogging logging;

    public static RunConfig loadDefault() throws FileNotFoundException {
        return load(defaultDirectory + defaultRunFile);
    }

    public static RunConfig load(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        if (!configFile.exists() && !configName.startsWith("/") && !configName.contains(":")) {
            configFile = new File(defaultDirectory + configName);
        }
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }
        InputStream inputStream = new FileInputStream(configFile);
        Yaml yaml = new Yaml();
        return yaml.loadAs(inputStream, RunConfig.class);
    }

    /** Run-level logging settings + selection for experiments.csv. */
    public static class RunLogging extends BenchmarkSelection {
        public String dir;       // e.g. "logging"
        public String runId;     // optional template; supports "{ts}" (UTC). Default: "{ts}"
        public String jvectorRef; // tag/commit to record
        public String type;      // e.g. "csv"
        public boolean sysStats; // optional background /proc stats
        public boolean jfr;      // optional JFR recording
    }
}
