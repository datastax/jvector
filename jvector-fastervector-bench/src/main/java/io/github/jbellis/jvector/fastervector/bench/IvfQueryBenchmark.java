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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;

import java.util.List;

/** The IVF counterpart of jvector-examples' {@code QueryBenchmark}: one measurement of one search point. */
public interface IvfQueryBenchmark {
    String getBenchmarkName();

    List<Metric> runBenchmark(IvfSystem system, SearchPoint point);
}
