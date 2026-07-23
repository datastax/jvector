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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Strategy interface for version-specific serialization of graph indexes.
 * Each version of the on-disk format has its own implementation that encapsulates
 * all version-specific logic for reading and writing graph data.
 * <p>
 * This design eliminates scattered version checks throughout the codebase and
 * makes it easy to add new versions without modifying existing code.
 */
public interface GraphIndexFormat {
    /**
     * @return the version number this format handles
     */
    int getVersion();

    /**
     * Checks if this version supports a given feature.
     * @param feature the feature to check
     * @return true if the feature is supported in this version
     */
    boolean supportsFeature(FeatureId feature);

    /**
     * Checks if this version supports multi-layer (hierarchical) graphs.
     * @return true if multi-layer graphs are supported
     */
    boolean supportsMultiLayer();

    /**
     * Checks if this version uses a footer for metadata instead of a header.
     * @return true if footer-based metadata is used
     */
    boolean usesFooter();

    /**
     * Gets the feature ordering strategy for this version.
     * @return the feature ordering strategy
     */
    FeatureOrdering getFeatureOrdering();

    /**
     * Returns the complete set of {@link FeatureId}s that this format version is capable of storing.
     *
     * @return an unmodifiable set of supported feature identifiers
     */
    Set<FeatureId> getSupportedFeatures();

    /**
     * Callback for writing L0 (base layer) node records. Implemented by the writer so that
     * I/O strategy (sequential vs. parallel) stays in the writer while the format
     * handles orchestration without depending on the concrete writer type.
     */
    @FunctionalInterface
    interface L0RecordWriter {
        /**
         * Writes the base-layer (level-0) node records using the provided graph view and feature suppliers.
         *
         * @param view      a view over the graph used to iterate neighbors at level 0
         * @param suppliers per-feature functions that produce a {@link Feature.State} for a given original ordinal
         * @throws IOException if an I/O error occurs while writing
         */
        void write(ImmutableGraphIndex.View view, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException;
    }

    /**
     * Defines how features should be ordered when writing/reading.
     */
    enum FeatureOrdering {
        /** Features ordered by their FeatureId enum ordinal (versions &lt;= 5) */
        BY_FEATURE_ID,
        /** Features ordered with fused features last (version 6+) */
        FUSED_LAST
    }

    /**
     * Writes adjacency records for all graph levels above level 0 (the "sparse" upper layers).
     *
     * @param ctx       the write session context
     * @param out       the sequential output stream
     * @param suppliers per-feature state suppliers keyed by {@link FeatureId}
     * @throws IOException if an I/O error occurs while writing
     */
    void writeSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException;

    /**
     * Writes the format header (version, dimensions, entry node, layer info, and feature metadata).
     *
     * @param ctx the write session context
     * @param out the sequential output stream positioned at the header location
     * @throws IOException if an I/O error occurs while writing
     */
    void writeHeader(WriteContext ctx, IndexWriter out) throws IOException;

    /**
     * Returns the byte offset within the output stream at which the inline feature data
     * for the given (new) ordinal begins.
     *
     * @param ctx     the write session context
     * @param ordinal the compacted (new) node ordinal
     * @return the absolute byte offset for the node's inline feature section
     */
    long featureOffsetForOrdinal(WriteContext ctx, int ordinal);

    /**
     * Writes the format footer containing the header offset and magic number,
     * used by footer-based formats (V5+) to locate the header at read time.
     *
     * @param ctx          the write session context
     * @param headerOffset the byte position in the output stream where the header was written
     * @param out          the sequential output stream
     * @throws IOException if an I/O error occurs while writing
     */
    void writeFooter(WriteContext ctx, long headerOffset, IndexWriter out) throws IOException;

    /**
     * Writes all {@link io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature} data blocks
     * sequentially after the node records.
     *
     * @param ctx       the write session context
     * @param out       the sequential output stream
     * @param suppliers per-feature state suppliers keyed by {@link FeatureId}
     * @throws IOException if an I/O error occurs while writing
     */
    void writeSeparatedFeatures(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException;

    /**
     * Writes inline feature data for a single node directly into the random-access output stream
     * at the position determined by {@link #featureOffsetForOrdinal}.
     *
     * @param ctx      the write session context
     * @param ordinal  the compacted (new) node ordinal whose features are being written
     * @param stateMap feature states to write, keyed by {@link FeatureId}
     * @param out      the random-access output stream
     * @throws IOException if an I/O error occurs while writing
     */
    void writeFeaturesInline(WriteContext ctx, int ordinal, Map<FeatureId, Feature.State> stateMap, RandomAccessWriter out) throws IOException;

    /**
     * Writes the complete graph index sequentially: header, node records, sparse levels,
     * separated features, and (for V5+) footer.  All nodes must be present in the ordinal
     * mapper; gaps (OMITTED ordinals) are not permitted.
     *
     * @param ctx       the write session context
     * @param out       the sequential output stream
     * @param suppliers per-feature state suppliers keyed by {@link FeatureId}
     * @throws IOException if an I/O error occurs while writing
     */
    void writeOnDiskSequential(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException;

    /**
     * Writes the complete graph index using random-access I/O, delegating base-layer record
     * writing to the provided {@link L0RecordWriter} (which may use parallel I/O).
     * After all data is written the header is re-written at its original position so that
     * accurate offset information is recorded.
     *
     * @param ctx       the write session context
     * @param out       the random-access output stream
     * @param suppliers per-feature state suppliers keyed by {@link FeatureId}
     * @param l0Writer  callback responsible for writing the level-0 node records
     * @throws IOException if an I/O error occurs while writing
     */
    void writeRandomAccess(WriteContext ctx, RandomAccessWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers, L0RecordWriter l0Writer) throws IOException;
}
