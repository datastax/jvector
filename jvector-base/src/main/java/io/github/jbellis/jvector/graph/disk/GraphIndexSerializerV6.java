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
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.IntFunction;

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
        super(allFeatures());
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

    @Override
    public void writeSparseLevels(ImmutableGraphIndex graph, OrdinalMapper ordinalMapper, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                // node id
                final int newOrdinal = ordinalMapper.oldToNew(originalOrdinal);
                out.writeInt(newOrdinal);
                // neighbors
                var view = graph.getView();
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                out.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    out.writeInt(ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    out.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }
        // There should be only one fused feature per node. This is checked in the class constructor.
        // This is the only place where we explicitly need the fused feature. If there are more places in the
        // future, it may be worth having fusedFeature as class member.
        FusedFeature fusedFeature = null;
        for (var feature : inlineFeatures) {
            if (feature.isFused()) {
                fusedFeature = (FusedFeature) feature;
            }
        }
        if (fusedFeature != null) {
            var supplier = featureStateSuppliers.get(fusedFeature.id());
            if (supplier == null) {
                throw new IllegalStateException("Supplier for feature " + fusedFeature.id() + " not found");
            }

            if (graph.getMaxLevel() >= 1) {
                int level = 1;
                int layerSize = graph.size(level);
                int nodesWritten = 0;
                for (var it = graph.getNodes(level); it.hasNext(); ) {
                    int originalOrdinal = it.nextInt();

                    // We write the ordinal (node id) so that we can map it to the corresponding feature
                    final int newOrdinal = ordinalMapper.oldToNew(originalOrdinal);
                    out.writeInt(newOrdinal);
                    fusedFeature.writeSourceFeature(out, supplier.apply(originalOrdinal));
                    nodesWritten++;
                }
                if (nodesWritten != layerSize) {
                    throw new IllegalStateException("Mismatch between layer 1 size and features written");
                }
            } else {
                // Write the source feature of the entry node
                final int originalEntryNode = graph.getView().entryNode().node;
                final int entryNode = ordinalMapper.oldToNew(originalEntryNode);
                out.writeInt(entryNode);
                fusedFeature.writeSourceFeature(out, supplier.apply(originalEntryNode));
            }
        }
    }
}

