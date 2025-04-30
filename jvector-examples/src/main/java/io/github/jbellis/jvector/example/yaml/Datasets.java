package io.github.jbellis.jvector.example.yaml;

import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Datasets {
    private static final String directory = "./jvector-examples/yaml-examples/";

    public final Map<String, List<String>> datasetNames;

    private Datasets(Map<String, List<String>> datasetNames) {
        this.datasetNames = datasetNames;
    }

    public static Datasets load() throws IOException  {
        InputStream inputStream = new FileInputStream(directory + "datasets.yml");
        Yaml yaml = new Yaml();
        return new Datasets(yaml.load(inputStream));
    }

    public List<String> getAll() {
        List<String> allDatasetNames = new ArrayList<>();
        for (var key : datasetNames.keySet()) {
            allDatasetNames.addAll(datasetNames.get(key));
        }
        return allDatasetNames;
    }
}
