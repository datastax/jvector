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

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

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
                if (!(strCenterData == null || strCenterData.equals("Yes") || strCenterData.equals("No"))) {
                    throw new IllegalArgumentException("centerData must be Yes or No, or not specified at all.");
                }
                float anisotropicThreshold = Float.parseFloat(parameters.getOrDefault("anisotropicThreshold", "-1"));

                return ds -> {
                    boolean centerData;
                    if (strCenterData == null) {
                        centerData = ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
                    } else {
                        centerData = strCenterData.equals("Yes");;
                    }

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