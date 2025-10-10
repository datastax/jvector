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

import java.io.DataOutput;
import java.io.IOException;
import java.util.EnumMap;
import java.util.function.IntFunction;

/**
 * A feature of an on-disk graph index. Information to use a feature is stored in the header on-disk.
 */
public interface Feature {
    /**
     * Returns the unique identifier for this feature.
     *
     * @return the FeatureId of this feature
     */
    FeatureId id();

    /**
     * Returns the size in bytes of this feature's header data.
     *
     * @return the header size in bytes
     */
    int headerSize();

    /**
     * Returns the size in bytes of this feature's per-node data.
     *
     * @return the feature size in bytes per node
     */
    int featureSize();

    /**
     * Writes this feature's header data to the output stream.
     *
     * @param out the output stream to write to
     * @throws IOException if an I/O error occurs
     */
    void writeHeader(DataOutput out) throws IOException;

    /**
     * Writes inline feature data for a node to the output stream.
     * Default implementation is a no-op for features that don't support inline storage.
     *
     * @param out the output stream to write to
     * @param state the state containing the data to write
     * @throws IOException if an I/O error occurs
     */
    default void writeInline(DataOutput out, State state) throws IOException {
        // default no-op
    }

    /**
     * Marker interface for feature-specific state used during writing.
     * Feature implementations should implement this interface for their specific state.
     */
    interface State {
    }

    /**
     * Creates a single-entry map associating a FeatureId with a state factory function.
     *
     * @param id the feature identifier
     * @param stateFactory the factory function to create state instances
     * @return an EnumMap containing the single mapping
     */
    static EnumMap<FeatureId, IntFunction<State>> singleStateFactory(FeatureId id, IntFunction<State> stateFactory) {
        EnumMap<FeatureId, IntFunction<State>> map = new EnumMap<>(FeatureId.class);
        map.put(id, stateFactory);
        return map;
    }

    /**
     * Creates a single-entry map associating a FeatureId with a state instance.
     *
     * @param id the feature identifier
     * @param state the state instance
     * @return an EnumMap containing the single mapping
     */
    static EnumMap<FeatureId, State> singleState(FeatureId id, State state) {
        EnumMap<FeatureId, State> map = new EnumMap<>(FeatureId.class);
        map.put(id, state);
        return map;
    }
}
