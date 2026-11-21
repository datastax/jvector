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

import java.util.Locale;

/** The queries the throughput benchmark warms up with before its timed passes. */
public enum WarmupQueries {
    /** Normalized random vectors. */
    RANDOM,
    /** Base vectors sampled uniformly from the dataset: they follow the data without favoring measured queries. */
    BASE,
    /** No warmup. */
    NONE;

    /** Parses {@code random}, {@code base} or {@code none}, ignoring case. */
    public static WarmupQueries parse(String value) {
        try {
            return valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException unknown) {
            throw new IllegalArgumentException("Warmup must be random, base or none: " + value);
        }
    }
}
