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
 * Shared "collect every missing required value, then report them all at once" bookkeeping for
 * index builders' {@code build()} methods, instead of failing on the first missing value.
 * <p>
 * Each backing's builder still owns its own required/optional distinction and any coupled or
 * mutually-exclusive checks between fields &mdash; this only replaces the boilerplate of
 * accumulating names and formatting one exception.
 */
public final class IndexBuilderValidation {
    private final List<String> missing = new ArrayList<>();

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
     * Throws an {@link IllegalStateException} naming every value recorded as missing so far, if
     * any. {@code builderDescription} is prepended to the message, e.g.
     * {@code "Cannot build GraphIndexBuilder"}.
     */
    public void throwIfAny(String builderDescription) {
        if (!missing.isEmpty()) {
            throw new IllegalStateException(
                    builderDescription + ", missing required value(s): " + String.join(", ", missing));
        }
    }
}
