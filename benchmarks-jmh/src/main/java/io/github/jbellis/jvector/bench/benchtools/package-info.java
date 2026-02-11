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

/**
 * Reusable benchmark infrastructure utilities.
 * <p>
 * This package provides general-purpose tools for JMH benchmarks including
 * git hash detection ({@link io.github.jbellis.jvector.bench.benchtools.GitInfo}),
 * JSONL result writing ({@link io.github.jbellis.jvector.bench.benchtools.JsonlWriter}),
 * JFR recording management ({@link io.github.jbellis.jvector.bench.benchtools.JfrRecorder}),
 * system stats collection ({@link io.github.jbellis.jvector.bench.benchtools.SystemStatsCollector}),
 * and {@code @Param} combination counting ({@link io.github.jbellis.jvector.bench.benchtools.BenchmarkParamCounter}).
 */
package io.github.jbellis.jvector.bench.benchtools;
