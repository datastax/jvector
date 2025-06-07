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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.Closeable;
import java.io.IOException;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * writes a graph index to a target
 */
public interface GraphIndexWriter extends Closeable {
    /**
     * Write the index header and completed edge lists to the given outputs.  Inline features given in
     * `featureStateSuppliers` will also be written.  (Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline).
     * <p>
     * Each supplier takes a node ordinal and returns a FeatureState suitable for Feature.writeInline.
     *
     * @param featureStateSuppliers a map of FeatureId to a function that returns a Feature.State
     */
    void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException;
}
