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

import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

/** Worker counts in index YAML: a positive integer, or {@code max} for the platform's physical cores. */
public final class Workers {
    private Workers() {}

    /** Resolves one YAML value; {@code field} names it in the error message. */
    public static int resolve(Object value, String field) {
        if ("max".equals(value)) return PhysicalCoreExecutor.getPhysicalCoreCount();
        if (value instanceof Integer && (Integer) value > 0) return (Integer) value;
        throw new IllegalArgumentException(field + ": " + value + " is not a worker count."
                + System.lineSeparator() + "Use a positive integer, or max for the platform's physical cores.");
    }
}
