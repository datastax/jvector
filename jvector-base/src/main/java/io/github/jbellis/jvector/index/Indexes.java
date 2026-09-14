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

import io.github.jbellis.jvector.graph.HnswIndexBuilder;
import io.github.jbellis.jvector.ivf.IvfIndexBuilder;

/**
 * Entry point for building a jvector {@link Index}: pick the backing type first, then only its
 * own parameters are available to set.
 * <p>
 * This lives here rather than as static methods on {@link Index} itself because {@link Index} is
 * part of jvector-api (the pure contract module, with no dependency on any concrete backing),
 * while the builders returned here construct concrete implementation objects and belong in
 * jvector-base alongside them.
 */
public final class Indexes {
    private Indexes() {
    }

    public static HnswIndexBuilder hnswBuilder() {
        return new HnswIndexBuilder();
    }

    public static IvfIndexBuilder ivfBuilder() {
        return new IvfIndexBuilder();
    }
}
