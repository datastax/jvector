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
import java.util.*;

/**
 * Format for version 3 of the on-disk graph format.
 * Version 3 characteristics:
 * - Has magic number
 * - Supports multiple features (feature set serialization)
 * - Single layer only
 * - No footer
 */
class GraphIndexFormatV3 extends AbstractGraphIndexFormat {

    /**
     * Creates the singleton format for version 3.
     */
    GraphIndexFormatV3() {
        super(3, nonFusedFeatures(), false, false);
    }

    protected GraphIndexFormatV3(int version, Set<FeatureId> supportedFeatures, boolean supportsMultiLayer, boolean usesFooter) {
        super(version, supportedFeatures, supportsMultiLayer, usesFooter);
    }

    @Override
    public void writeCommonHeader(IndexWriter out, List<CommonHeader.LayerInfo> layerInfo, int dimension, int entryNode, int idUpperBound) throws IOException {
        out.writeInt(OnDiskGraphIndex.MAGIC);
        out.writeInt(this.getVersion());
        super.writeCommonHeader(out, layerInfo, dimension, entryNode, idUpperBound);
    }

    @Override
    public int commonHeaderSize() {
        return 6 * Integer.BYTES;
    }

    @Override
    public void writeHeaderFeatures(IndexWriter out, Map<FeatureId, ? extends Feature> features) throws IOException {
        out.writeInt(FeatureId.serialize(EnumSet.copyOf(features.keySet())));
        super.writeHeaderFeatures(out, features);
    }

    @Override
    public int headerSize(Map<FeatureId, ? extends Feature> features) {
        int size = super.headerSize(features);

        size += Integer.BYTES;

        return size;
    }

    @Override
    public Map<FeatureId, Feature> loadHeaderFeatures(RandomAccessReader reader, CommonHeader common) throws IOException {
        EnumSet<FeatureId> featureIds;
        Map<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        featureIds = FeatureId.deserialize(reader.readInt());
        for (FeatureId featureId : featureIds) {
            features.put(featureId, featureId.load(common, reader));
        }
        return features;
    }

}
