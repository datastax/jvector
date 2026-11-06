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

package io.github.jbellis.jvector.example;

import java.util.List;
import java.util.function.BiConsumer;

/**
 * One swept construction parameter: the values to try and how each one is set on a {@link BuildConfig}.
 *
 * @param <T> the parameter's type
 */
public final class SweepDimension<T> {
    private final String name;
    private final List<T> values;
    private final BiConsumer<BuildConfig.Builder, T> setter;

    private SweepDimension(String name, List<T> values, BiConsumer<BuildConfig.Builder, T> setter) {
        this.name = name;
        this.values = values;
        this.setter = setter;
    }

    /**
     * @param name the parameter name, used in error messages
     * @param values the values to sweep, in order
     * @param setter sets one value on a builder, typically a {@link BuildConfig.Builder} method reference
     * @throws IllegalArgumentException if {@code values} is null, e.g. a YAML field that was not set
     */
    public static <T> SweepDimension<T> of(String name, List<T> values, BiConsumer<BuildConfig.Builder, T> setter) {
        if (values == null) {
            throw new IllegalArgumentException("No values to sweep for " + name);
        }
        return new SweepDimension<>(name, values, setter);
    }

    public String name() {
        return name;
    }

    int size() {
        return values.size();
    }

    void apply(BuildConfig.Builder builder, int index) {
        setter.accept(builder, values.get(index));
    }
}
