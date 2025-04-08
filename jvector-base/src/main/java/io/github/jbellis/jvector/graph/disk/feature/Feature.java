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
    FeatureId id();

    int headerSize();

    int featureSize();

    void writeHeader(DataOutput out) throws IOException;

    default void writeInline(DataOutput out, State state) throws IOException {
        // default no-op
    }

    // Feature implementations should implement a State as well for use with writeInline/writeSeparately
    interface State {
    }

    static EnumMap<FeatureId, IntFunction<State>> singleStateFactory(FeatureId id, IntFunction<State> stateFactory) {
        EnumMap<FeatureId, IntFunction<State>> map = new EnumMap<>(FeatureId.class);
        map.put(id, stateFactory);
        return map;
    }

    static EnumMap<FeatureId, State> singleState(FeatureId id, State state) {
        EnumMap<FeatureId, State> map = new EnumMap<>(FeatureId.class);
        map.put(id, state);
        return map;
    }
}
