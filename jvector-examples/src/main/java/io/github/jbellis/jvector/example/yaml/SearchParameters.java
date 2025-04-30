package io.github.jbellis.jvector.example.yaml;

import java.util.List;
import java.util.Map;

public class SearchParameters extends CommonParameters {
    public Map<Integer, List<Double>> topKOverquery;
    public List<Boolean> useSearchPruning;
}