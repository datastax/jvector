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

/**
 * Search logging sink configuration.
 *
 * YAML shape:
 * search:
 *   logging:
 *     type: csv
 *     dir: logging
 *     benchmarks: ...
 *     metrics: ...
 */
public class SearchLoggingParameters extends BenchmarkSelection {
    /**
     * Logging sink type: "csv", "parquet", etc.
     */
    public String type;

    /**
     * Output directory for logs. If null, a default will be used.
     */
    public String dir;
}
