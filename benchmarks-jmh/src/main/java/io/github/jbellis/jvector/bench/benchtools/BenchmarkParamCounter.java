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

package io.github.jbellis.jvector.bench.benchtools;

import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.runner.options.CommandLineOptions;

/**
 * Counts the total number of {@code @Param} combinations for a JMH benchmark class.
 */
public final class BenchmarkParamCounter {
    private BenchmarkParamCounter() {}

    /**
     * Computes the total number of benchmark parameter combinations as the cartesian product
     * of all {@code @Param} value sets. When {@code cmdOptions} is provided, command-line
     * {@code -p} overrides take precedence over the annotation defaults.
     *
     * @param benchmarkClass the JMH benchmark class to inspect
     * @param cmdOptions     parsed command-line options, or {@code null} to use annotation defaults only
     * @return the total number of parameter combinations
     */
    public static int computeTotalTests(Class<?> benchmarkClass, CommandLineOptions cmdOptions) {
        int total = 1;
        for (var field : benchmarkClass.getDeclaredFields()) {
            var paramAnnotation = field.getAnnotation(Param.class);
            if (paramAnnotation != null) {
                if (cmdOptions != null) {
                    var cmdOverride = cmdOptions.getParameter(field.getName());
                    if (cmdOverride.hasValue() && !cmdOverride.get().isEmpty()) {
                        total *= cmdOverride.get().size();
                        continue;
                    }
                }
                total *= paramAnnotation.value().length;
            }
        }
        return total;
    }
}
