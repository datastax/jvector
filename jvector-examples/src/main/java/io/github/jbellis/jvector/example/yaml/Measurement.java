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

import io.github.jbellis.jvector.example.benchmarks.WarmupQueries;

/**
 * Run YAML {@code measurement}: how benchmarks run, as opposed to which stats they compute. Every field is
 * optional; omitted fields keep the built-in behavior.
 */
public class Measurement {
    /** Drop each on-disk index file from the operating system page cache before its first search (Linux). */
    public Boolean evictIndex;
    /** Settings for the throughput benchmark. */
    public Throughput throughput;

    public static class Throughput {
        /** Warmup queries: random (default), base (vectors sampled from the dataset) or none. */
        public String warmup;
        /** Minimum warmup passes over the query count; default 3. */
        public Integer warmupPasses;
        /** Keep warming, up to 10 passes, until a pass compiles little new JIT code; default false. */
        public Boolean untilCompiled;
        /** Timed passes; default 3. */
        public Integer passes;
        /** Minimum seconds per timed pass, repeating the query set if needed; default 1.0. */
        public Double minSeconds;

        public WarmupQueries warmup() {
            return warmup == null ? WarmupQueries.RANDOM : WarmupQueries.parse(warmup);
        }

        public int warmupPasses() {
            return warmupPasses == null ? 3 : warmupPasses;
        }

        public boolean untilCompiled() {
            return Boolean.TRUE.equals(untilCompiled);
        }

        public int passes() {
            return passes == null ? 3 : passes;
        }

        public double minSeconds() {
            return minSeconds == null ? 1.0 : minSeconds;
        }

        void validate() {
            try {
                warmup();
            } catch (IllegalArgumentException invalid) {
                throw new IllegalArgumentException("measurement.throughput.warmup must be random, base or none: " + warmup, invalid);
            }
            if (warmupPasses != null && warmupPasses < 0) {
                throw new IllegalArgumentException("measurement.throughput.warmupPasses must be zero or more");
            }
            if (passes != null && passes < 1) {
                throw new IllegalArgumentException("measurement.throughput.passes must be at least 1");
            }
            if (minSeconds != null && (!Double.isFinite(minSeconds) || minSeconds < 0)) {
                throw new IllegalArgumentException("measurement.throughput.minSeconds must be finite and nonnegative");
            }
            if (warmup() == WarmupQueries.NONE && untilCompiled()) {
                throw new IllegalArgumentException("measurement.throughput.untilCompiled requires warmup queries");
            }
        }
    }

    public boolean evictIndex() {
        return Boolean.TRUE.equals(evictIndex);
    }

    void validate() {
        if (throughput != null) throughput.validate();
    }
}
