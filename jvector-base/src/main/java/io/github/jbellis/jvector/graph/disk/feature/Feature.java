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

import io.github.jbellis.jvector.disk.IndexWriter;

import java.io.IOException;
import java.util.EnumMap;
import java.util.function.IntFunction;

/**
 * A feature of an on-disk graph index. Information to use a feature is stored in the header on-disk.
 */
public interface Feature extends Comparable<Feature> {
    FeatureId id();

    default boolean isFused() {
        return false;
    }

    int headerSize();

    /** The bytes {@link #writeHeader(IndexWriter, int)} writes at {@code version}. */
    default int headerSize(int version) {
        return headerSize();
    }

    int featureSize();

    /**
     * Writes this feature's header at {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex#CURRENT_VERSION}.
     *
     * @deprecated a header embedded in a graph must be written at the graph's version; use
     *             {@link #writeHeader(IndexWriter, int)}
     */
    @Deprecated
    void writeHeader(IndexWriter out) throws IOException;

    /**
     * Writes this feature's header for a graph written at {@code version}. Features that serialize a
     * compressor write it at this version, so a graph written at an older version is laid out as that
     * version wrote it. The default ignores the version, for headers that don't depend on it.
     *
     * @param out the output to write to
     * @param version the on-disk format version of the graph being written
     */
    default void writeHeader(IndexWriter out, int version) throws IOException {
        writeHeader(out);
    }

    default void writeInline(IndexWriter out, State state) throws IOException {
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
