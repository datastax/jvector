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
package io.github.jbellis.jvector.example;

import java.util.Map;

/**
 * Benchmark result data container.
 */
public class BenchResult {
    /** The dataset name. */
    public String dataset;
    /** The benchmark parameters. */
    public Map<String, Object> parameters;
    /** The benchmark metrics. */
    public Map<String, Object> metrics;

    /**
     * Constructs a BenchResult.
     */
    public BenchResult() {}

    /**
     * Constructs a BenchResult with the specified values.
     * @param dataset the dataset name
     * @param parameters the benchmark parameters
     * @param metrics the benchmark metrics
     */
    public BenchResult(String dataset, Map<String, Object> parameters, Map<String, Object> metrics) {
        this.dataset = dataset;
        this.parameters = parameters;
        this.metrics = metrics;
    }
}
