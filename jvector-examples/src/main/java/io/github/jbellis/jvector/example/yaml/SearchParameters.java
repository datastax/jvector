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

import java.util.List;
import java.util.Map;

public class SearchParameters extends CommonParameters {
    /** ASH query arithmetic; omitted settings preserve JVM properties and auto. */
    public String queryProjection;

    public static final String ASH_QUERY_PROJECTION_PROPERTY = "jvector.ash.queryProjection";

    /** Resolve one precision before loading models; cached views cannot change mid-JVM. */
    public static final class AshQueryProjectionSetting implements AutoCloseable {
        private final String original = System.getProperty(ASH_QUERY_PROJECTION_PROPERTY);
        private boolean applied;

        public void apply(List<MultiConfig> configs) {
            String selected = null;
            boolean explicitYaml = false;
            for (MultiConfig config : configs) {
                String yaml = config.search == null ? null : config.search.queryProjection;
                if (yaml != null) validatePrecision(yaml);
                if (!hasAsh(config.construction) && !hasAsh(config.search)) continue;
                String mode = original != null ? original : yaml == null ? "auto" : yaml;
                validatePrecision(mode);
                if (selected != null && !selected.equals(mode))
                    throw new IllegalArgumentException("Different ASH queryProjection settings require separate BenchYAML JVMs");
                selected = mode;
                explicitYaml |= yaml != null;
            }
            if (original == null && explicitYaml && selected != null) {
                System.setProperty(ASH_QUERY_PROJECTION_PROPERTY, selected);
                applied = true;
            }
        }

        private static boolean hasAsh(CommonParameters parameters) {
            return parameters != null && parameters.compression != null
                    && parameters.compression.stream().anyMatch(c -> c != null && "ASH".equals(c.type));
        }

        private static void validatePrecision(String value) {
            if (!List.of("auto", "fp32", "fp16", "int8").contains(value))
                throw new IllegalArgumentException("search.queryProjection must be auto, fp32, fp16 or int8: " + value);
        }

        @Override public void close() {
            if (applied) {
                System.clearProperty(ASH_QUERY_PROJECTION_PROPERTY);
                applied = false;
            }
        }
    }

    public Map<Integer, List<Double>> topKOverquery;
    public List<Boolean> useSearchPruning;

    // NOTE: benchmark compute + console/logging selection are now run-level (run-config.yml)
    // and are no longer recognized in dataset configs.
}
