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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import java.util.Locale;
import java.util.logging.Logger;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

/** Base-vector storage policy for benchmark harnesses, independent of quantization. */
public enum BaseVectorLoading {
    BUFFERED,
    PRELOAD;

    private static final Logger LOG = Logger.getLogger(BaseVectorLoading.class.getName());

    /** Existing VM option is an explicit override of the YAML flag. */
    public static BaseVectorLoading forBenchmark(boolean useDatasetPrefetching) {
        String value = System.getProperty("jvector.dataset_loader",
                useDatasetPrefetching ? "buffered" : "preload");
        try { return valueOf(value.toUpperCase(Locale.ROOT)); }
        catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("jvector.dataset_loader must be buffered or preload, got: " + value, e);
        }
    }

    public static BaseVectorLoading forBenchmark() { return forBenchmark(false); }

    /**
     * The VM option overrides the YAML flag. Configurations requiring resident raw
     * vectors retain preload so construction and query scoring stay unchanged.
     */
    public static BaseVectorLoading forBenchmark(MultiConfig config) {
        var loading = forBenchmark(config.useDatasetPrefetching);
        if (loading == PRELOAD) return loading;
        if (config.construction.compression.stream().anyMatch(c -> "None".equals(c.type)))
            return residentInput("uncompressed construction");
        boolean exactSearch = config.search.compression.stream().anyMatch(c -> "None".equals(c.type));
        boolean needsRawInput = exactSearch && config.construction.getFeatureSets().stream().anyMatch(features ->
                !features.contains(FeatureId.FUSED_PQ));
        return needsRawInput ? residentInput("raw-input exact search") : BUFFERED;
    }

    private static BaseVectorLoading residentInput(String reason) {
        if ("buffered".equalsIgnoreCase(System.getProperty("jvector.dataset_loader")))
            LOG.warning("Using preload instead of jvector.dataset_loader=buffered for " + reason
                    + "; retaining resident input to preserve scoring behavior.");
        return PRELOAD;
    }
}
