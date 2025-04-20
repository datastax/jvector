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

package io.github.jbellis.jvector.example.benchmarks;

import java.util.Map;
import java.util.function.Function;

/**
 * A single column in the table:
 *   - header: The column title
 *   - fmtSpec: The format specifier, e.g. ".3f" for floats or "s" for strings
 *   - extractor: How to pull the value from the summary‐map
 */
public class Metric {
    private final String header;
    private final String fmtSpec;
    private final Function<
            Map<Class<? extends BenchmarkSummary>, BenchmarkSummary>,
            Double
            > extractor;

    private Metric(String header,
                   String fmtSpec,
                   Function<
                           Map<Class<? extends BenchmarkSummary>,BenchmarkSummary>,
                           Double
                           > extractor) {
        this.header    = header;
        this.fmtSpec   = fmtSpec;
        this.extractor = extractor;
    }

    public String getHeader()   { return header; }
    public String getFmtSpec()  { return fmtSpec;  }
    public double extract(Map<Class<? extends BenchmarkSummary>,BenchmarkSummary> m) {
        return extractor.apply(m);
    }

    /**
     * Factory to create a Metric given:
     *   - Header string
     *   - The Summary.class to look up
     *   - A getter method reference from the Summary
     *   - The format specifier (e.g. ".2f", ".3f")
     */
    public static <T extends BenchmarkSummary> Metric of(
            String header,
            Class<T> summaryClass,
            Function<T,Double> getter,
            String fmtSpec
    ) {
        return new Metric(
                header,
                fmtSpec,
                summaries -> {
                    @SuppressWarnings("unchecked")
                    T s = (T) summaries.get(summaryClass);
                    return getter.apply(s);
                }
        );
    }
}

