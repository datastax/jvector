package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;

import java.util.Map;
import java.util.function.Function;

public class Compression {
    public String type;
    public Map<String, String> parameters;

    public Function<DataSet, CompressorParameters> getCompressorParameters() {
        switch (type) {
            case "None":
                return __ -> CompressorParameters.NONE;
            case "PQ":
                int k = Integer.parseInt(parameters.getOrDefault("k", "256"));
                String strCenterData = parameters.get("centerData");
                if (strCenterData == null || !(strCenterData.equals("Yes") || strCenterData.equals("No"))) {
                    throw new IllegalArgumentException("centerData must be Yes or No");
                }
                boolean centerData = strCenterData.equals("Yes");;
                float anisotropicThreshold = Float.parseFloat(parameters.getOrDefault("anisotropicThreshold", "-1"));

                return ds -> {
                    if (parameters.containsKey("m")) {
                        int m = Integer.parseInt(parameters.get("m"));
                        return new CompressorParameters.PQParameters(m, k, centerData, anisotropicThreshold);
                    } else if (parameters.containsKey("mFactor")) {
                        String strMFactor = parameters.get("mFactor");
                        int mFactor = Integer.parseInt(strMFactor);
                        return new CompressorParameters.PQParameters(ds.getDimension() / mFactor, k, centerData, anisotropicThreshold);
                    } else {
                        throw new IllegalArgumentException("Need to specify either 'm' or 'mFactor'");
                    }
                };
            case "BQ":
                return ds -> new CompressorParameters.BQParameters();
            default:
                throw new IllegalArgumentException("Unsupported compression type: " + type);

        }
    }
}