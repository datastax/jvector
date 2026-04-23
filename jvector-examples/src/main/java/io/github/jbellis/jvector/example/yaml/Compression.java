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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/// YAML-bound representation of a single {@code compression:} entry.
///
/// Scalar parameter values (e.g. {@code mFactor: 2}) and list-valued parameters
/// (e.g. {@code mFactor: [2, 4]}) are both accepted; list-valued parameters cause
/// {@link #getCompressorParameters()} to return one entry per point in the
/// Cartesian product across all list-valued axes. A single compression entry with
/// {@code mFactor: [2, 4]} and {@code anisotropicThreshold: [-1.0, 0.2]} therefore
/// expands to four compressor configurations.
public class Compression {
    public String type;
    public Map<String, Object> parameters;

    /// Expand this compression entry into one or more {@link CompressorParameters}
    /// suppliers. Non-PQ types (None, BQ) always produce a single-element list;
    /// PQ produces the Cartesian product across {@code m}/{@code mFactor},
    /// {@code k}, {@code centerData}, and {@code anisotropicThreshold}.
    public List<Function<DataSet, CompressorParameters>> getCompressorParameters() {
        switch (type) {
            case "None":
                return List.of(__ -> CompressorParameters.NONE);
            case "BQ":
                return List.of(ds -> new CompressorParameters.BQParameters());
            case "PQ":
                return pqCombinations();
            default:
                throw new IllegalArgumentException("Unsupported compression type: " + type);
        }
    }

    private List<Function<DataSet, CompressorParameters>> pqCombinations() {
        Map<String, Object> params = parameters == null ? Map.of() : parameters;

        List<Integer> ks = asIntList(params.getOrDefault("k", 256));
        List<Float> thresholds = asFloatList(params.getOrDefault("anisotropicThreshold", -1.0f));

        // centerData absent => use dataset-similarity-based default at resolve time (null sentinel)
        List<Boolean> centerings = params.containsKey("centerData")
                ? asBooleanList(params.get("centerData"))
                : Collections.singletonList(null);

        boolean hasM = params.containsKey("m");
        boolean hasMFactor = params.containsKey("mFactor");
        if (!hasM && !hasMFactor) {
            throw new IllegalArgumentException("PQ compression: need to specify either 'm' or 'mFactor'");
        }

        // 'm' takes precedence when both are present (matches prior behavior).
        List<MSource> mSources = new ArrayList<>();
        if (hasM) {
            for (Integer mv : asIntList(params.get("m"))) {
                mSources.add(MSource.exact(mv));
            }
        } else {
            for (Integer f : asIntList(params.get("mFactor"))) {
                mSources.add(MSource.factor(f));
            }
        }

        List<Function<DataSet, CompressorParameters>> out = new ArrayList<>(
                mSources.size() * ks.size() * centerings.size() * thresholds.size());
        for (MSource ms : mSources) {
            for (Integer k : ks) {
                for (Boolean cd : centerings) {
                    for (Float at : thresholds) {
                        out.add(pqFunction(ms, k, cd, at));
                    }
                }
            }
        }
        return out;
    }

    private static Function<DataSet, CompressorParameters> pqFunction(MSource ms,
                                                                      int k,
                                                                      Boolean centerDataSpec,
                                                                      float anisotropicThreshold) {
        return ds -> {
            boolean centerData = (centerDataSpec != null)
                    ? centerDataSpec
                    : ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
            int m = ms.resolve(ds);
            return new CompressorParameters.PQParameters(m, k, centerData, anisotropicThreshold);
        };
    }

    /// Source of the {@code m} dimension: either an exact value or a divisor applied to
    /// the dataset dimensionality at resolve time.
    private static final class MSource {
        private final Integer exactM;
        private final Integer factor;

        private MSource(Integer exactM, Integer factor) {
            this.exactM = exactM;
            this.factor = factor;
        }

        static MSource exact(int m) { return new MSource(m, null); }
        static MSource factor(int f) { return new MSource(null, f); }

        int resolve(DataSet ds) {
            return (exactM != null) ? exactM : ds.getDimension() / factor;
        }
    }

    // ------------------------------------------------------------------------
    // YAML value coercion: scalar or list -> typed list
    // ------------------------------------------------------------------------

    private static List<Integer> asIntList(Object raw) {
        if (raw instanceof List<?>) {
            List<?> src = (List<?>) raw;
            List<Integer> out = new ArrayList<>(src.size());
            for (Object o : src) out.add(toInt(o));
            return out;
        }
        return List.of(toInt(raw));
    }

    private static List<Float> asFloatList(Object raw) {
        if (raw instanceof List<?>) {
            List<?> src = (List<?>) raw;
            List<Float> out = new ArrayList<>(src.size());
            for (Object o : src) out.add(toFloat(o));
            return out;
        }
        return List.of(toFloat(raw));
    }

    private static List<Boolean> asBooleanList(Object raw) {
        if (raw instanceof List<?>) {
            List<?> src = (List<?>) raw;
            List<Boolean> out = new ArrayList<>(src.size());
            for (Object o : src) out.add(toBoolean(o));
            return out;
        }
        return List.of(toBoolean(raw));
    }

    private static int toInt(Object o) {
        if (o instanceof Number) return ((Number) o).intValue();
        if (o instanceof String) return Integer.parseInt(((String) o).trim());
        throw new IllegalArgumentException("Cannot interpret as integer: " + o);
    }

    private static float toFloat(Object o) {
        if (o instanceof Number) return ((Number) o).floatValue();
        if (o instanceof String) return Float.parseFloat(((String) o).trim());
        throw new IllegalArgumentException("Cannot interpret as float: " + o);
    }

    /// Accepts YAML-native {@code Boolean} (from {@code true}/{@code false} or
    /// {@code Yes}/{@code No}) as well as the literal strings {@code "Yes"}/{@code "No"}/
    /// {@code "true"}/{@code "false"} for configs that quote the value.
    private static boolean toBoolean(Object o) {
        if (o instanceof Boolean) return (Boolean) o;
        if (o instanceof String) {
            String s = ((String) o).trim();
            if (s.equalsIgnoreCase("Yes") || s.equalsIgnoreCase("true")) return true;
            if (s.equalsIgnoreCase("No") || s.equalsIgnoreCase("false")) return false;
            throw new IllegalArgumentException("Cannot interpret as boolean: " + o);
        }
        throw new IllegalArgumentException("Cannot interpret as boolean: " + o);
    }
}
