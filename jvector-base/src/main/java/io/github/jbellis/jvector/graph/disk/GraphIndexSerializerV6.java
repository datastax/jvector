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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Serializer for version 6 of the on-disk graph format.
 * Version 6 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata
 * - NEW: Changes feature ordering to place fused features last
 * - NEW: Writes feature count and ordinals explicitly instead of bitflags
 */
class GraphIndexSerializerV6 extends GraphIndexSerializerV4 {

    GraphIndexSerializerV6() {
        // Inherits most behavior from V4, but changes feature ordering
    }

    @Override
    public int getVersion() {
        return 6;
    }

    @Override
    public boolean usesFooter() {
        return true;
    }

    @Override
    public FeatureOrdering getFeatureOrdering() {
        return FeatureOrdering.FUSED_LAST;
    }

    @Override
    public void writeFeatureHeaders(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException {
        // V6 writes feature count and ordinals explicitly (preserving order)
        out.writeInt(features.size());
        
        for (var entry : features.entrySet()) {
            out.writeInt(entry.getKey().ordinal());
            entry.getValue().writeHeader(out);
        }
    }

    @Override
    public Map<FeatureId, Feature> readFeatureHeaders(RandomAccessReader in, GraphIndexMetadata metadata) throws IOException {
        // V6 reads features in order (LinkedHashMap preserves insertion order)
        Map<FeatureId, Feature> features = new LinkedHashMap<>();
        
        int nFeatures = in.readInt();
        
        // Create a CommonHeader from metadata for feature loading
        CommonHeader commonHeader = new CommonHeader(
            metadata.getVersion(),
            metadata.getDimension(),
            metadata.getEntryNode(),
            metadata.getLayerInfo(),
            metadata.getIdUpperBound()
        );
        
        for (int i = 0; i < nFeatures; i++) {
            FeatureId featureId = FeatureId.values()[in.readInt()];
            features.put(featureId, featureId.load(commonHeader, in));
        }
        
        return features;
    }

    @Override
    public int calculateHeaderSize(GraphIndexMetadata metadata, Map<FeatureId, ? extends Feature> features) {
        // V6: magic + version + 4 base ints + idUpperBound + numLayers + (32 * 2 layer info ints) 
        //     + feature count + (feature ordinals) + feature headers
        int size = 8 * Integer.BYTES; // magic, version, size, dimension, entryNode, maxDegree, idUpperBound, numLayers
        size += 2 * 32 * Integer.BYTES; // layer info (padded to 32 layers)
        size += Integer.BYTES; // feature count
        size += features.size() * Integer.BYTES; // feature ordinals
        size += features.values().stream().mapToInt(Feature::headerSize).sum();
        return size;
    }
}

// Made with Bob
