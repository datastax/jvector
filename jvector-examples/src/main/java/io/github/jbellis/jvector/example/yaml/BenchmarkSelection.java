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
 * A sink-agnostic selection of benchmark stats and related metrics, using the same schema
 * as SearchParameters.benchmarks ({@code Map<String, List<String>>}).
 *
 * Used for console projections now, and logging (CSV/Parquet/etc) later.
 */
public class BenchmarkSelection {
    public Map<String, List<String>> benchmarks;
    public MetricSelection metrics;
}
