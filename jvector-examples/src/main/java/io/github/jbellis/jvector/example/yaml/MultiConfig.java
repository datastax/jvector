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
    private int version;
    public String dataset;
    public ConstructionParameters construction;
    public SearchParameters search;

    public static MultiConfig getConfig(String datasetName) throws FileNotFoundException {
        File configFile = new File("./jvector-examples/yaml-examples/" + datasetName + ".yml");
        if (!configFile.exists()) {
            configFile = new File("./jvector-examples/yaml-examples/default.yml");
            System.out.println("Default YAML config file: " + configFile.getAbsolutePath());
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