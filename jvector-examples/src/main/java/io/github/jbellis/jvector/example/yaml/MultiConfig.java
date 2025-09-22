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

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import org.yaml.snakeyaml.Yaml;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class MultiConfig {
    private static final String defaultDirectory = "./jvector-examples/yaml-configs/";

    private int version;
    public String dataset;
    public ConstructionParameters construction;
    public SearchParameters search;

    public static MultiConfig getDefaultConfig(String datasetName) throws FileNotFoundException {
        var name = defaultDirectory + datasetName;
        if (!name.endsWith(".yml")) {
            name += ".yml";
        }
        File configFile = new File(name);
        boolean useDefault = !configFile.exists();
        if (useDefault) {
            configFile = new File(defaultDirectory + "default.yml");
            System.out.println("Default YAML config file: " + configFile.getAbsolutePath());
        }
        var config = getConfig(configFile);
        if (useDefault) {
            config.dataset = datasetName;
        }
        return config;
    }

    public static MultiConfig getConfig(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        // If the file doesn't exist as an absolute path, try relative to the default directory
        if (!configFile.exists() && !configName.startsWith("/") && !configName.contains(":")) {
            configFile = new File(defaultDirectory + configName);
        }
        return getConfig(configFile);
    }

    public static MultiConfig getConfig(File configFile) throws FileNotFoundException {
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }
        InputStream inputStream = new FileInputStream(configFile);
        Yaml yaml = new Yaml();
        return yaml.loadAs(inputStream, MultiConfig.class);
    }

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        if (version != OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Invalid version: " + version);
        }
        this.version = version;
    }
}