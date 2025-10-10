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

/**
 * Configuration parameters for search operations in benchmarks.
 * Extends CommonParameters to include search-specific settings like topK, overquery ratios,
 * pruning options, and benchmark configurations.
 */
public class SearchParameters extends CommonParameters {
    /** Map of topK values to lists of overquery ratios. */
    public Map<Integer, List<Double>> topKOverquery;
    /** List of boolean flags indicating whether to use search pruning. */
    public List<Boolean> useSearchPruning;
    /** Map of benchmark names to their configuration options. */
    public Map<String, List<String>> benchmarks;

    /**
     * Constructs an empty SearchParameters instance.
     */
    public SearchParameters() {
    }
}