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

/** Base-vector storage policy for benchmark harnesses, independent of quantization. */
public enum BaseVectorLoading {
    BUFFERED,
    PRELOAD;

    /** BenchYAML/Grid default; an explicit property selects the resident baseline. */
    public static BaseVectorLoading forBenchmark() {
        String value = System.getProperty("jvector.dataset_loader", "buffered");
        try {
            return valueOf(value.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("jvector.dataset_loader must be buffered or preload, got: " + value, e);
        }
    }
}
