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
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

/**
 * Serializer for version 3 of the on-disk graph format.
 * Version 3 characteristics:
 * - Has magic number
 * - Supports multiple features (feature set serialization)
 * - Single layer only
 * - No footer
 */
class GraphIndexSerializerV3 extends AbstractGraphIndexSerializer {

    GraphIndexSerializerV3() {
        super(3, allFeatures(), false, false, FeatureOrdering.BY_FEATURE_ID);
    }

    @Override
    public void writeCommonHeader(IndexWriter out, GraphIndexMetadata metadata) throws IOException {
        // V3 format: magic, version, size, dimension, entryNode, maxDegree
        out.writeInt(OnDiskGraphIndex.MAGIC);
        out.writeInt(3);
        out.writeInt(metadata.getBaseLayerSize());
        out.writeInt(metadata.getDimension());
        out.writeInt(metadata.getEntryNode());
        out.writeInt(metadata.getBaseLayerDegree());
    }

    @Override
    public GraphIndexMetadata readCommonHeader(RandomAccessReader in) throws IOException {
        // V3 format: magic, version, size, dimension, entryNode, maxDegree
        int magic = in.readInt();
        if (magic != OnDiskGraphIndex.MAGIC) {
            throw new IOException("Invalid magic number: " + magic);
        }
        int version = in.readInt();
        int size = in.readInt();
        int dimension = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();
        
        // V3 only supports single layer
        List<CommonHeader.LayerInfo> layerInfo = List.of(new CommonHeader.LayerInfo(size, maxDegree));
        int idUpperBound = size;
        
        return new GraphIndexMetadata(version, dimension, entryNode, layerInfo, idUpperBound);
    }

    @Override
    public void writeFeatureHeaders(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException {
        // V3 writes feature set as bitflags
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
        // V3: magic + version + 4 ints (size, dimension, entryNode, maxDegree) + feature bitflags + feature headers
        int size = 6 * Integer.BYTES; // magic, version, size, dimension, entryNode, maxDegree
        size += Integer.BYTES; // feature bitflags
        size += features.values().stream().mapToInt(Feature::headerSize).sum();
        return size;
    }
}

// Made with Bob
