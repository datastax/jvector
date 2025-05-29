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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DatasetCollection {
    private static final String defaultFile = "./jvector-examples/yaml-configs/datasets.yml";

    public final Map<String, List<String>> datasetNames;

    private DatasetCollection(Map<String, List<String>> datasetNames) {
        this.datasetNames = datasetNames;
    }

    public static DatasetCollection load() throws IOException  {
        return load(defaultFile);
    }

    public static DatasetCollection load(String file) throws IOException  {
        InputStream inputStream = new FileInputStream(file);
        Yaml yaml = new Yaml();
        return new DatasetCollection(yaml.load(inputStream));
    }

    public List<String> getAll() {
        List<String> allDatasetNames = new ArrayList<>();
        for (var key : datasetNames.keySet()) {
            allDatasetNames.addAll(datasetNames.get(key));
        }
        return allDatasetNames;
    }
}
