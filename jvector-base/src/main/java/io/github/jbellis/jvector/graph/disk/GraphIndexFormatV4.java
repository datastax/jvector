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
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static io.github.jbellis.jvector.graph.disk.CommonHeader.V4_MAX_LAYERS;

/**
 * Format for version 4 of the on-disk graph format.
 * Version 4 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - No footer
 */
class GraphIndexFormatV4 extends GraphIndexFormatV3 {
    private static final Logger logger = LoggerFactory.getLogger(GraphIndexFormatV4.class);

    /** Creates the singleton format for version 4. */
    GraphIndexFormatV4() {
        super(4, nonFusedFeatures(), true, false);
    }

    /**
     * Protected constructor allowing subclasses (V5, V6) to specify their own version,
     * feature set, and footer flag while sharing V4's wire format.
     */
    protected GraphIndexFormatV4(int version, Set<FeatureId> supportedFeatures, boolean usesFooter) {
        super(version, supportedFeatures, true, usesFooter);
    }

    @Override
    public void writeCommonHeader(IndexWriter out, List<CommonHeader.LayerInfo> layerInfo, int dimension, int entryNode, int idUpperBound) throws IOException {
        out.writeInt(OnDiskGraphIndex.MAGIC);
        out.writeInt(this.getVersion());
        out.writeInt(layerInfo.get(0).size);
        out.writeInt(dimension);
        out.writeInt(entryNode);
        out.writeInt(layerInfo.get(0).degree);
        out.writeInt(idUpperBound);

        if (layerInfo.size() > V4_MAX_LAYERS) {
            var msg = String.format("Number of layers %d exceeds maximum of %d", layerInfo.size(), V4_MAX_LAYERS);
            throw new IllegalArgumentException(msg);
        }
        logger.debug("Writing {} layers", layerInfo.size());
        out.writeInt(layerInfo.size());
        // Write actual layer info
        for (CommonHeader.LayerInfo info : layerInfo) {
            out.writeInt(info.size);
            out.writeInt(info.degree);
        }
        // Pad remaining entries with zeros
        for (int i = layerInfo.size(); i < V4_MAX_LAYERS; i++) {
            out.writeInt(0); // size
            out.writeInt(0); // degree
        }
    }

    @Override
    public CommonHeader readCommonHeader(RandomAccessReader in, int size) throws IOException {
        int dimension = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();

        List<CommonHeader.LayerInfo> layerInfo;
        int idUpperBound = in.readInt();
        int numLayers = in.readInt();
        logger.debug("{} layers", numLayers);
        layerInfo = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            CommonHeader.LayerInfo info = new CommonHeader.LayerInfo(in.readInt(), in.readInt());
            layerInfo.add(info);
        }
        // Skip over remaining padding entries
        for (int i = numLayers; i < V4_MAX_LAYERS; i++) {
            in.readInt();
            in.readInt();
        }
        logger.debug("Common header finished reading at position {}", in.getPosition());

        return new CommonHeader(getVersion(), dimension, entryNode, layerInfo, idUpperBound);
    }

    @Override
    public int commonHeaderSize() {
        // super.commonHeaderSize() (V3's) covers magic + version + the 4 shared base fields;
        // V4 adds idUpperBound, the layer count, and the fixed-size padded layer array.
        return super.commonHeaderSize() + (2 + 2 * V4_MAX_LAYERS) * Integer.BYTES;
    }

}
