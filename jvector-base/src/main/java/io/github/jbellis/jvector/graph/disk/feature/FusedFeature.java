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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.util.Accountable;

import java.io.DataOutput;
import java.io.IOException;

/**
 * A fused feature is one that is computed from the neighbors of a node.
 * - writeInline writes the fused features based on the neighbors of the node
 * - writeSource writes the feature of the node itself
 * Implements Quick ADC-style scoring by fusing PQ-encoded neighbors into an OnDiskGraphIndex.
 */
public interface FusedFeature extends Feature {
    default boolean isFused() {
        return true;
    }

    void writeSourceFeature(DataOutput out, State state) throws IOException;

    interface InlineSource extends Accountable {}

    InlineSource loadSourceFeature(RandomAccessReader in) throws IOException;
}
