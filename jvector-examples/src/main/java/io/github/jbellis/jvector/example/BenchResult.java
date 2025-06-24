package io.github.jbellis.jvector.example;

import java.util.Map;

public class BenchResult {
    public String dataset;
    public Map<String, Object> parameters;
    public Map<String, Object> metrics;

    public BenchResult() {}
    public BenchResult(String dataset, Map<String, Object> parameters, Map<String, Object> metrics) {
        this.dataset = dataset;
        this.parameters = parameters;
        this.metrics = metrics;
    }
}
