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
 * Serializer for version 2 of the on-disk graph format.
 * Version 2 characteristics:
 * - No magic number
 * - Only supports INLINE_VECTORS feature
 * - Single layer only
 * - No footer
 */
class GraphIndexSerializerV2 extends AbstractGraphIndexSerializer {

    GraphIndexSerializerV2() {
        super(2, inlineVectorsOnly(), false, false, FeatureOrdering.BY_FEATURE_ID);
    }

    @Override
    public void writeCommonHeader(IndexWriter out, GraphIndexMetadata metadata) throws IOException {
        // V2 format: size, dimension, entryNode, maxDegree (no magic, no version)
        out.writeInt(metadata.getBaseLayerSize());
        out.writeInt(metadata.getDimension());
        out.writeInt(metadata.getEntryNode());
        out.writeInt(metadata.getBaseLayerDegree());
    }

    @Override
    public GraphIndexMetadata readCommonHeader(RandomAccessReader in) throws IOException {
        // V2 format: size, dimension, entryNode, maxDegree
        int size = in.readInt();
        int dimension = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();
        
        // V2 only supports single layer
        List<CommonHeader.LayerInfo> layerInfo = List.of(new CommonHeader.LayerInfo(size, maxDegree));
        int idUpperBound = size;
        
        return new GraphIndexMetadata(2, dimension, entryNode, layerInfo, idUpperBound);
    }

    @Override
    public void writeFeatureHeaders(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException {
        // V2 doesn't write feature set information, just the feature headers
        // Only INLINE_VECTORS is supported
        if (!features.containsKey(FeatureId.INLINE_VECTORS) || features.size() > 1) {
            throw new IllegalArgumentException("Version 2 only supports INLINE_VECTORS feature");
        }
        
        for (Feature feature : features.values()) {
            feature.writeHeader(out);
        }
    }

    @Override
    public Map<FeatureId, Feature> readFeatureHeaders(RandomAccessReader in, GraphIndexMetadata metadata) throws IOException {
        // V2 only has INLINE_VECTORS
        Map<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        EnumSet<FeatureId> featureIds = EnumSet.of(FeatureId.INLINE_VECTORS);
        
        // Create a CommonHeader from metadata for feature loading
        CommonHeader commonHeader = new CommonHeader(
            metadata.getVersion(),
            metadata.getDimension(),
            metadata.getEntryNode(),
            metadata.getLayerInfo(),
            metadata.getIdUpperBound()
        );
        
        for (FeatureId featureId : featureIds) {
            features.put(featureId, featureId.load(commonHeader, in));
        }
        
        return features;
    }

    @Override
    public int calculateHeaderSize(GraphIndexMetadata metadata, Map<FeatureId, ? extends Feature> features) {
        // V2: 4 ints (size, dimension, entryNode, maxDegree) + feature headers
        int size = 4 * Integer.BYTES;
        size += features.values().stream().mapToInt(Feature::headerSize).sum();
        return size;
    }
}

// Made with Bob
