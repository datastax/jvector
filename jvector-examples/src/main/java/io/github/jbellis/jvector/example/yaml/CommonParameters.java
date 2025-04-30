package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

public class CommonParameters {
    public List<Compression> compression;

    public List<Function<DataSet, CompressorParameters>> getCompressorParameters() {
        return compression.stream().map(Compression::getCompressorParameters).collect(Collectors.toList());
    }
}
