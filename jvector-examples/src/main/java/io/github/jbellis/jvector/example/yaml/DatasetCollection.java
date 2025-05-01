package io.github.jbellis.jvector.example.yaml;

import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DatasetCollection {
    private static final String defaultFile = "./jvector-examples/yaml-examples/datasets.yml";

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
