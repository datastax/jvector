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

import io.github.jbellis.jvector.example.benchmarks.Metric;

import java.util.List;

/** One query configuration's measurements, as reported to a {@link GridListener}. */
final class QueryRow {
    final boolean usePruning;
    final int topK;
    final double overquery;
    final int rerankK;
    /** Query benchmark results followed by the index's construction metrics. */
    final List<Metric> metrics;

    QueryRow(boolean usePruning, int topK, double overquery, int rerankK, List<Metric> metrics) {
        this.usePruning = usePruning;
        this.topK = topK;
        this.overquery = overquery;
        this.rerankK = rerankK;
        this.metrics = metrics;
    }
}
