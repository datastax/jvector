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
                        centerData = ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
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
            case "ASH":
                return ds -> {
                    if (parameters == null) {
                        throw new IllegalArgumentException("ASH requires parameters");
                    }

                    int landmarkCount = Integer.parseInt(parameters.getOrDefault("landmarkCount", "1"));

                    // optimizer can be numeric or name (RANDOM/ITQ/LANDING)
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
                        case "LANDING":
                            optimizer = AsymmetricHashing.LANDING;
                            break;
                        default:
                            optimizer = Integer.parseInt(optStr);
                    }

                    boolean hasEncodedBits = parameters.containsKey("encodedBits");
                    boolean hasCompressionRatio = parameters.containsKey("compressionRatio");
                    boolean hasQuantizedDim = parameters.containsKey("quantizedDim");

                    if ((hasEncodedBits ? 1 : 0) + (hasCompressionRatio ? 1 : 0) + (hasQuantizedDim ? 1 : 0) != 1) {
                        throw new IllegalArgumentException(
                                "ASH requires exactly one of 'encodedBits', 'compressionRatio' or 'quantizedDim'");
                    }

                    int bitDepth = Integer.parseInt(parameters.getOrDefault("bitDepth", "1"));

                    int quantizedDim;
                    if (hasEncodedBits) {
                        int encodedBits = Integer.parseInt(parameters.get("encodedBits"));
                        var payloadBits = encodedBits - AsymmetricHashing.HEADER_BITS;
                        quantizedDim = payloadBits / bitDepth;
                        if (quantizedDim * bitDepth != payloadBits) {
                            throw new IllegalArgumentException("Couldn't create ASH with exactly " + encodedBits + " encoded bits for bit depth " + bitDepth);
                        }
                    } else if (hasCompressionRatio) {
                        int requestedCompressionRatio = Integer.parseInt(parameters.get("compressionRatio"));
                        if (requestedCompressionRatio != 32
                                && requestedCompressionRatio != 64
                                && requestedCompressionRatio != 128
                                && requestedCompressionRatio != 256) {
                            throw new IllegalArgumentException(
                                    "ASH compressionRatio must be one of 32, 64, 128, or 256");
                        }

                        int originalBits = ds.getDimension() * Float.SIZE;
                        int targetEncodedBits = Math.max(AsymmetricHashing.HEADER_BITS + 64,
                                originalBits / requestedCompressionRatio);

                        int payloadTargetBits = Math.max(64, targetEncodedBits - AsymmetricHashing.HEADER_BITS);
                        int lowerPayloadBits = Math.max(64, (payloadTargetBits / 64) * 64);
                        int upperPayloadBits = Math.max(64, ((payloadTargetBits + 63) / 64) * 64);

                        int payloadBits = Math.abs(lowerPayloadBits - payloadTargetBits)
                                <= Math.abs(upperPayloadBits - payloadTargetBits)
                                ? lowerPayloadBits
                                : upperPayloadBits;

                        int encodedBits = AsymmetricHashing.HEADER_BITS + payloadBits;

                        double actualCompressionRatio = (double) originalBits / encodedBits;
                        // TODO this might no longer hold if the container isn't a long[] anymore
                        System.out.printf(
                                "ASH requested compressionRatio=%d resolved to encodedBits=%d "
                                        + "(header=%d, payload=%d) actualCompressionRatio=%.2f%n",
                                requestedCompressionRatio,
                                encodedBits,
                                AsymmetricHashing.HEADER_BITS,
                                payloadBits,
                                actualCompressionRatio);

                        quantizedDim = payloadBits / bitDepth;
                        // TODO this doesn't really play well with the approximation in the previous step
                        if (quantizedDim * bitDepth != payloadBits) {
                            throw new IllegalArgumentException("Couldn't create ASH with exactly " + encodedBits + " encoded bits for bit depth " + bitDepth);
                        }
                    } else {
                        quantizedDim = Integer.parseInt(parameters.get("quantizedDim"));
                    }

                    return new CompressorParameters.ASHParameters(optimizer, quantizedDim, bitDepth, landmarkCount);
                };
            default:
                throw new IllegalArgumentException("Unsupported compression type: " + type);

        }
    }
}