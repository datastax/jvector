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
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

/**
 * Serializer for version 4 of the on-disk graph format.
 * Version 4 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - No footer
 */
class GraphIndexSerializerV4 extends AbstractGraphIndexSerializer {
    private static final int V4_MAX_LAYERS = 32;

    GraphIndexSerializerV4() {
        super(4, allFeatures(), true, false, FeatureOrdering.BY_FEATURE_ID);
    }

    @Override
    public void writeCommonHeader(IndexWriter out, GraphIndexMetadata metadata) throws IOException {
        // V4 format: magic, version, size, dimension, entryNode, maxDegree, idUpperBound, numLayers, layer info
        out.writeInt(OnDiskGraphIndex.MAGIC);
        out.writeInt(4);
        out.writeInt(metadata.getBaseLayerSize());
        out.writeInt(metadata.getDimension());
        out.writeInt(metadata.getEntryNode());
        out.writeInt(metadata.getBaseLayerDegree());
        out.writeInt(metadata.getIdUpperBound());
        
        if (metadata.getLayerInfo().size() > V4_MAX_LAYERS) {
            throw new IllegalArgumentException(
                String.format("Number of layers %d exceeds maximum of %d", 
                    metadata.getLayerInfo().size(), V4_MAX_LAYERS));
        }
        
        out.writeInt(metadata.getLayerInfo().size());
        
        // Write actual layer info
        for (CommonHeader.LayerInfo info : metadata.getLayerInfo()) {
            out.writeInt(info.size);
            out.writeInt(info.degree);
        }
        
        // Pad remaining entries with zeros
        for (int i = metadata.getLayerInfo().size(); i < V4_MAX_LAYERS; i++) {
            out.writeInt(0); // size
            out.writeInt(0); // degree
        }
    }

    @Override
    public GraphIndexMetadata readCommonHeader(RandomAccessReader in) throws IOException {
        int magic = in.readInt();
        if (magic != OnDiskGraphIndex.MAGIC) {
            throw new IOException("Invalid magic number: " + magic);
        }
        int version = in.readInt();
        int size = in.readInt();
        int dimension = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();
        int idUpperBound = in.readInt();
        int numLayers = in.readInt();
        
        List<CommonHeader.LayerInfo> layerInfo = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            CommonHeader.LayerInfo info = new CommonHeader.LayerInfo(in.readInt(), in.readInt());
            layerInfo.add(info);
        }
        
        // Skip over remaining padding entries
        for (int i = numLayers; i < V4_MAX_LAYERS; i++) {
            in.readInt();
            in.readInt();
        }
        
        return new GraphIndexMetadata(version, dimension, entryNode, layerInfo, idUpperBound);
    }

    @Override
    public void writeFeatureHeaders(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException {
        // V4 writes feature set as bitflags
        out.writeInt(FeatureId.serialize(EnumSet.copyOf(features.keySet())));
        
        // Then write each feature's header
        for (Feature feature : features.values()) {
            feature.writeHeader(out);
        }
    }

    @Override
    public Map<FeatureId, Feature> readFeatureHeaders(RandomAccessReader in, GraphIndexMetadata metadata) throws IOException {
        Map<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        
        // Read feature set bitflags
        EnumSet<FeatureId> featureIds = FeatureId.deserialize(in.readInt());
        
        // Create a CommonHeader from metadata for feature loading
        CommonHeader commonHeader = new CommonHeader(
            metadata.getVersion(),
            metadata.getDimension(),
            metadata.getEntryNode(),
            metadata.getLayerInfo(),
            metadata.getIdUpperBound()
        );
        
        // Load each feature
        for (FeatureId featureId : featureIds) {
            features.put(featureId, featureId.load(commonHeader, in));
        }
        
        return features;
    }

    @Override
    public int calculateHeaderSize(GraphIndexMetadata metadata, Map<FeatureId, ? extends Feature> features) {
        // V4: magic + version + 4 base ints + idUpperBound + numLayers + (32 * 2 layer info ints) + feature bitflags + feature headers
        int size = 8 * Integer.BYTES; // magic, version, size, dimension, entryNode, maxDegree, idUpperBound, numLayers
        size += 2 * V4_MAX_LAYERS * Integer.BYTES; // layer info (padded to 32 layers)
        size += Integer.BYTES; // feature bitflags
        size += features.values().stream().mapToInt(Feature::headerSize).sum();
        return size;
    }
}

// Made with Bob
