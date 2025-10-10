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
package io.github.jbellis.jvector.benchframe;

import java.util.Map;

/**
 * Result model for a single benchmark execution. Encapsulates the dataset identifier,
 * configuration parameters, and performance metrics from a benchmark run.
 * <p>
 * This class is designed for serialization to JSON and CSV formats through {@link ResultHandler}
 * implementations. All fields are public for compatibility with Jackson and other serialization
 * libraries.
 * <p>
 * Typical parameter keys include:
 * <ul>
 *   <li>{@code M} - max connections per node</li>
 *   <li>{@code efConstruction} - construction-time search depth</li>
 *   <li>{@code buildCompressor} - compression used during construction</li>
 *   <li>{@code searchCompressor} - compression used during search</li>
 *   <li>{@code featureSet} - enabled feature flags</li>
 * </ul>
 * <p>
 * Typical metric keys include:
 * <ul>
 *   <li>{@code recall} - search accuracy (0.0 to 1.0)</li>
 *   <li>{@code qps} - queries per second</li>
 *   <li>{@code latency} - average query latency in milliseconds</li>
 *   <li>{@code buildTimeMs} - index construction time in milliseconds</li>
 *   <li>{@code indexSizeBytes} - on-disk index size in bytes</li>
 * </ul>
 *
 * @see ResultHandler
 * @see BenchFrame
 */
public class BenchResult {
    /**
     * The name of the dataset this result is for.
     */
    public String dataset;

    /**
     * Map of configuration parameters used for this benchmark run.
     * Keys are parameter names, values are parameter values (typically String, Integer, Boolean, etc.).
     */
    public Map<String, Object> parameters;

    /**
     * Map of performance metrics measured during this benchmark run.
     * Keys are metric names, values are metric values (typically Double, Long, Integer, etc.).
     */
    public Map<String, Object> metrics;

    /**
     * Default constructor for deserialization.
     */
    public BenchResult() {}

    /**
     * Constructs a BenchResult with the specified dataset, parameters, and metrics.
     *
     * @param dataset the dataset name
     * @param parameters map of configuration parameters
     * @param metrics map of performance metrics
     */
    public BenchResult(String dataset, Map<String, Object> parameters, Map<String, Object> metrics) {
        this.dataset = dataset;
        this.parameters = parameters;
        this.metrics = metrics;
    }
}
