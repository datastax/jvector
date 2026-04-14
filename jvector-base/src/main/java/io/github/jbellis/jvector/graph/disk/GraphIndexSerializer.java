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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.Map;

/**
 * Strategy interface for version-specific serialization of graph indexes.
 * Each version of the on-disk format has its own implementation that encapsulates
 * all version-specific logic for reading and writing graph data.
 * <p>
 * This design eliminates scattered version checks throughout the codebase and
 * makes it easy to add new versions without modifying existing code.
 */
public interface GraphIndexSerializer {
    /**
     * @return the version number this serializer handles
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
     * Writes the common header portion of the index.
     * @param out the output writer
     * @param metadata the metadata to write
     * @throws IOException if an I/O error occurs
     */
    void writeCommonHeader(IndexWriter out, GraphIndexMetadata metadata) throws IOException;

    /**
     * Reads the common header portion of the index.
     * @param in the input reader
     * @return the metadata read from the header
     * @throws IOException if an I/O error occurs
     */
    GraphIndexMetadata readCommonHeader(RandomAccessReader in) throws IOException;

    /**
     * Writes feature-specific header information.
     * @param out the output writer
     * @param features the features to write headers for
     * @throws IOException if an I/O error occurs
     */
    void writeFeatureHeaders(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException;

    /**
     * Reads feature-specific header information.
     * @param in the input reader
     * @param metadata the common metadata (needed for feature construction)
     * @return map of feature IDs to feature instances
     * @throws IOException if an I/O error occurs
     */
    Map<FeatureId, Feature> readFeatureHeaders(RandomAccessReader in, GraphIndexMetadata metadata) throws IOException;

    /**
     * Calculates the total size of the header in bytes.
     * @param metadata the metadata
     * @param features the features
     * @return the header size in bytes
     */
    int calculateHeaderSize(GraphIndexMetadata metadata, Map<FeatureId, ? extends Feature> features);

    /**
     * Defines how features should be ordered when writing/reading.
     */
    enum FeatureOrdering {
        /** Features ordered by their FeatureId enum ordinal (versions <= 5) */
        BY_FEATURE_ID,
        /** Features ordered with fused features last (version 6+) */
        FUSED_LAST
    }
}

// Made with Bob
