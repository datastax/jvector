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

/**
 * Configuration container for benchmark runs, loaded from YAML files.
 * Includes dataset selection, construction parameters, and search parameters.
 */
public class MultiConfig {
    /** Default directory for YAML configuration files. */
    private static final String defaultDirectory = "./jvector-examples/yaml-configs/";

    /** The version of the configuration format. */
    private int version;
    /** The name of the dataset to use. */
    public String dataset;
    /** Parameters for graph construction. */
    public ConstructionParameters construction;
    /** Parameters for search operations. */
    public SearchParameters search;

    /**
     * Constructs an empty MultiConfig.
     */
    public MultiConfig() {
    }

    /**
     * Loads the default configuration for the specified dataset.
     * If a dataset-specific config file exists, it is used; otherwise the default.yml is loaded
     * and the dataset name is set.
     *
     * @param datasetName the name of the dataset
     * @return the loaded configuration
     * @throws FileNotFoundException if neither dataset-specific nor default config is found
     */
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

    /**
     * Loads a configuration from the specified file name.
     *
     * @param configName the path to the configuration file
     * @return the loaded configuration
     * @throws FileNotFoundException if the configuration file is not found
     */
    public static MultiConfig getConfig(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        return getConfig(configFile);
    }

    /**
     * Loads a configuration from the specified file.
     *
     * @param configFile the configuration file to load
     * @return the loaded configuration
     * @throws FileNotFoundException if the configuration file is not found
     */
    public static MultiConfig getConfig(File configFile) throws FileNotFoundException {
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }
        InputStream inputStream = new FileInputStream(configFile);
        Yaml yaml = new Yaml();
        return yaml.loadAs(inputStream, MultiConfig.class);
    }

    /**
     * Returns the configuration format version.
     *
     * @return the version number
     */
    public int getVersion() {
        return version;
    }

    /**
     * Sets the configuration format version.
     * The version must match the current OnDiskGraphIndex version.
     *
     * @param version the version number to set
     * @throws IllegalArgumentException if the version does not match CURRENT_VERSION
     */
    public void setVersion(int version) {
        if (version != OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Invalid version: " + version);
        }
        this.version = version;
    }
}