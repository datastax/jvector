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
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.Map;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class Compression {
    public String type;
    public String scoring;
    public Map<String, String> parameters;

    void validateSchema3(String section, boolean search) {
        String path = section + ".compression";
        if (type == null || !List.of("None", "ASH", "PQ", "BQ").contains(type)) {
            throw new IllegalArgumentException(path + ".type must be None, ASH, PQ, or BQ");
        }
        if (scoring != null && (search || !"ASH".equals(type)
                || !List.of("symmetric", "asymmetric", "mixed").contains(scoring))) {
            throw new IllegalArgumentException(path + ".scoring requires ASH construction and must be symmetric, asymmetric, or mixed");
        }

        Set<String> allowed;
        switch (type) {
            case "ASH":
                allowed = search
                        ? Set.of("optimizer", "encodedBits", "compressionRatio", "landmarkCount", "bitsPerDimension",
                                 "queryProjection", "queryExecution", "maxBatchSize")
                        : Set.of("optimizer", "encodedBits", "compressionRatio", "landmarkCount", "bitsPerDimension",
                                 "queryProjection");
                break;
            case "PQ":
                allowed = Set.of("m", "mFactor", "k", "centerData", "anisotropicThreshold");
                break;
            default:
                allowed = Set.of();
        }
        var settings = parameters == null ? Map.<String, String>of() : parameters;
        for (var setting : settings.entrySet()) {
            var key = setting.getKey();
            if (key == null || !allowed.contains(key) || setting.getValue() == null) {
                throw new IllegalArgumentException(path + ".parameters." + key + " is invalid for " + type);
            }
        }
        if ("PQ".equals(type) && settings.containsKey("m") == settings.containsKey("mFactor")) {
            throw new IllegalArgumentException(path + " requires exactly one of m or mFactor for PQ");
        }
        if ("PQ".equals(type) && settings.containsKey("mFactor") && Integer.parseInt(settings.get("mFactor")) < 1) {
            throw new IllegalArgumentException(path + ".parameters.mFactor must be positive");
        }
        if ("ASH".equals(type)) {
            if (!settings.containsKey("optimizer")
                    || settings.containsKey("encodedBits") == settings.containsKey("compressionRatio")) {
                throw new IllegalArgumentException(path + " requires optimizer and exactly one of encodedBits or compressionRatio for ASH");
            }
            if (!List.of("ITQ", "RANDOM").contains(settings.get("optimizer"))) {
                throw new IllegalArgumentException(path + ".parameters.optimizer must be ITQ or RANDOM");
            }
            var projection = settings.get("queryProjection");
            if (projection != null) SearchParameters.AshQueryProjectionSetting.validatePrecision(projection);
            var execution = queryExecution();
            if (!List.of("direct", "queued").contains(execution)) {
                throw new IllegalArgumentException(path + ".parameters.queryExecution must be direct or queued");
            }
            if (settings.containsKey("maxBatchSize") && !"queued".equals(execution)) {
                throw new IllegalArgumentException(path + ".parameters.maxBatchSize requires queued queryExecution");
            }
            maxBatchSize();
        }
    }

    public String queryExecution() {
        return parameters == null ? "direct" : parameters.getOrDefault("queryExecution", "direct");
    }

    public int maxBatchSize() {
        String configured = parameters == null ? null : parameters.get("maxBatchSize");
        if (configured == null) return 8;
        int size = Integer.parseInt(configured);
        if (size <= 0 || size == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("maxBatchSize must be positive and fit its histogram");
        }
        return size;
    }

    public String ashConstructionMode() {
        if (scoring == null) return null;
        switch (scoring) {
            case "symmetric": return "symmetric-all";
            case "mixed": return "symmetric";
            case "asymmetric": return "asymmetric";
            default: throw new IllegalArgumentException("Unknown ASH construction scoring: " + scoring);
        }
    }

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
                        centerData = ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
                    } else {
                        centerData = strCenterData.equals("Yes");
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
            case "ASH":
                return ds -> {
                    if (parameters == null) {
                        throw new IllegalArgumentException("ASH requires parameters");
                    }

                    int landmarkCount = Integer.parseInt(parameters.getOrDefault("landmarkCount", "1"));
                    int bitsPerDimension = Integer.parseInt(parameters.getOrDefault("bitsPerDimension", "1"));
                    if (bitsPerDimension < 1 || bitsPerDimension > 9) {
                        throw new IllegalArgumentException(
                                "ASH bitsPerDimension must be in [1,9], got " + bitsPerDimension);
                    }

                    // optimizer can be numeric or name (RANDOM/ITQ)
                    String optStr = parameters.get("optimizer");
                    if (optStr == null) {
                        throw new IllegalArgumentException("ASH requires 'optimizer' parameter");
                    }

                    int optimizer;
                    switch (optStr.trim().toUpperCase()) {
                        case "RANDOM":
                            optimizer = AsymmetricHashing.RANDOM;
                            break;
                        case "ITQ":
                            optimizer = AsymmetricHashing.ITQ;
                            break;
                        default:
                            optimizer = Integer.parseInt(optStr);
                    }

                    boolean hasEncodedBits = parameters.containsKey("encodedBits");
                    boolean hasCompressionRatio = parameters.containsKey("compressionRatio");
                    if (hasEncodedBits == hasCompressionRatio) {
                        throw new IllegalArgumentException(
                                "ASH requires exactly one of 'encodedBits' or 'compressionRatio'");
                    }

                    int encodedBits;
                    if (hasEncodedBits) {
                        encodedBits = Integer.parseInt(parameters.get("encodedBits"));
                    } else {
                        int requestedCompressionRatio = Integer.parseInt(parameters.get("compressionRatio"));
                        if (requestedCompressionRatio != 8
                                && requestedCompressionRatio != 16
                                && requestedCompressionRatio != 32
                                && requestedCompressionRatio != 64
                                && requestedCompressionRatio != 128
                                && requestedCompressionRatio != 256) {
                            throw new IllegalArgumentException(
                                    "ASH compressionRatio must be one of 8, 16, 32, 64, 128, or 256");
                        }

                        int originalBits = ds.getDimension() * Float.SIZE;
                        int targetEncodedBits = Math.max(AsymmetricHashing.HEADER_BITS + 64,
                                originalBits / requestedCompressionRatio);

                        int minimumPayloadBits = ((64 + bitsPerDimension - 1) / bitsPerDimension)
                                * bitsPerDimension;
                        int payloadTargetBits = Math.max(minimumPayloadBits,
                                targetEncodedBits - AsymmetricHashing.HEADER_BITS);
                        int lowerPayloadBits = Math.max(minimumPayloadBits,
                                (payloadTargetBits / bitsPerDimension) * bitsPerDimension);
                        int upperPayloadBits = Math.max(minimumPayloadBits,
                                ((payloadTargetBits + bitsPerDimension - 1) / bitsPerDimension) * bitsPerDimension);

                        int payloadBits = Math.abs(lowerPayloadBits - payloadTargetBits)
                                <= Math.abs(upperPayloadBits - payloadTargetBits)
                                ? lowerPayloadBits
                                : upperPayloadBits;

                        encodedBits = AsymmetricHashing.HEADER_BITS + payloadBits;

                        double actualCompressionRatio = (double) originalBits / encodedBits;
                        System.out.printf(
                                "ASH requested compressionRatio=%d bitsPerDimension=%d resolved to encodedBits=%d "
                                        + "(header=%d, payload=%d) actualCompressionRatio=%.2f%n",
                                requestedCompressionRatio,
                                bitsPerDimension,
                                encodedBits,
                                AsymmetricHashing.HEADER_BITS,
                                payloadBits,
                                actualCompressionRatio);
                    }

                    return new CompressorParameters.ASHParameters(
                            optimizer, encodedBits, landmarkCount, bitsPerDimension);
                };
            default:
                throw new IllegalArgumentException("Unsupported compression type: " + type);

        }
    }
}
