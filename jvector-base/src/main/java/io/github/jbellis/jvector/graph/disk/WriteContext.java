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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.List;
import java.util.Map;

/**
 * Encapsulates the per-write-session context shared by all serializer operations.
 * Eliminates the repeated passing of the same 6-7 parameters through every method
 * in the serializer hierarchy.
 */
class WriteContext {
    /** The immutable graph being serialized. */
    final ImmutableGraphIndex graph;
    /** Maps between original (old) ordinals and the compacted (new) ordinals written to disk. */
    final OrdinalMapper ordinalMapper;
    /** All features configured for this index, keyed by their {@link FeatureId}. */
    final Map<FeatureId, Feature> featureMap;
    /** Ordered list of features that are written inline with each node record. */
    final List<Feature> inlineFeatures;
    /** Byte offset within the output stream at which this graph's data begins. */
    final long startOffset;
    /** Byte size of the header (or footer) written before the node records. */
    final long headerSize;
    /** Vector dimension, derived from the configured features rather than the graph. */
    final int dimension;

    /**
     * Constructs a {@code WriteContext} bundling all parameters required for a single serialization session.
     *
     * @param graph          the immutable graph to serialize
     * @param ordinalMapper  mapping between original and compacted ordinals
     * @param featureMap     all features configured for the index
     * @param inlineFeatures features written inline alongside each node record
     * @param startOffset    byte offset in the output stream at which the graph data begins
     * @param headerSize     byte size of the header written before node records
     * @param dimension      vector dimension, sourced from the configured features
     */
    WriteContext(ImmutableGraphIndex graph,
                 OrdinalMapper ordinalMapper,
                 Map<FeatureId, Feature> featureMap,
                 List<Feature> inlineFeatures,
                 long startOffset,
                 long headerSize,
                 int dimension) {
        this.graph = graph;
        this.ordinalMapper = ordinalMapper;
        this.featureMap = featureMap;
        this.inlineFeatures = inlineFeatures;
        this.startOffset = startOffset;
        this.headerSize = headerSize;
        this.dimension = dimension;
    }
}
