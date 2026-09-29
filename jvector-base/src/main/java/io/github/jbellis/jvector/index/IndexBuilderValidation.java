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

package io.github.jbellis.jvector.index;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared "collect every problem, then report them all at once" bookkeeping for index builders'
 * {@code build()} methods, instead of failing on the first one: missing required values
 * ({@link #require}, {@link #requireCondition}), and invalid values or conflicting settings
 * ({@link #check}).
 * <p>
 * Each backing's builder still decides what is required, what ranges are valid, and which settings
 * conflict &mdash; this only replaces the boilerplate of accumulating them and formatting one
 * exception.
 */
public final class IndexBuilderValidation {
    private final List<String> missing = new ArrayList<>();
    private final List<String> invalid = new ArrayList<>();

    /**
     * Records {@code name} as missing if {@code value} is {@code null}.
     */
    public IndexBuilderValidation require(String name, Object value) {
        if (value == null) {
            missing.add(name);
        }
        return this;
    }

    /**
     * Records {@code name} as missing if {@code present} is {@code false}. Use this for
     * conditionally-required values whose presence can't be expressed as a single null check
     * (e.g. "required unless some other field is set").
     */
    public IndexBuilderValidation requireCondition(String name, boolean present) {
        if (!present) {
            missing.add(name);
        }
        return this;
    }

    /**
     * Records {@code problem} if {@code valid} is {@code false}. Use this for a value that is present
     * but out of range, or for settings that conflict with each other. {@code problem} should say
     * what is wrong in full, including the offending value, e.g. {@code "beamWidth must be positive
     * (was 0)"}.
     */
    public IndexBuilderValidation check(boolean valid, String problem) {
        if (!valid) {
            invalid.add(problem);
        }
        return this;
    }

    /**
     * Throws an {@link IllegalStateException} describing every problem recorded so far, if any: the
     * missing values first, then each invalid or conflicting setting. {@code builderDescription} is
     * prepended to the message, e.g. {@code "Cannot build GraphIndexBuilder"}.
     */
    public void throwIfAny(String builderDescription) {
        if (missing.isEmpty() && invalid.isEmpty()) {
            return;
        }
        StringBuilder message = new StringBuilder(builderDescription);
        if (!missing.isEmpty()) {
            message.append(", missing required value(s): ").append(String.join(", ", missing));
        }
        if (!invalid.isEmpty()) {
            message.append(missing.isEmpty() ? ": " : "; ").append(String.join("; ", invalid));
        }
        throw new IllegalStateException(message.toString());
    }
}
