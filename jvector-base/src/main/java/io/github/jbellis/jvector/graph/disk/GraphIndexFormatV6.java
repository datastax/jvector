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
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;

import java.io.IOException;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Format for version 6 of the on-disk graph format.
 * Version 6 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata
 * - Changes feature ordering to place fused features last
 * - Writes feature count and ordinals explicitly instead of bitflags
 */
class GraphIndexFormatV6 extends GraphIndexFormatV5 {

    /** Creates the singleton format for version 6. */
    GraphIndexFormatV6() {
        super(6, allFeatures());
    }

    /** Places fused features last so non-fused inline features occupy a contiguous prefix. */
    @Override
    public Map<FeatureId, Feature> orderFeatures(EnumMap<FeatureId, Feature> features) {
        var sorted = features.values().stream().sorted().collect(Collectors.toList());
        var map = new LinkedHashMap<FeatureId, Feature>();
        for (var f : sorted) {
            map.put(f.id(), f);
        }
        return map;
    }

    /**
     * Appends fused feature data for level-1 nodes after the sparse levels are written.
     * At most one fused feature is permitted per graph (enforced at write time).
     */
    @Override
    protected void writeAfterSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {
        FusedFeature fusedFeature = null;
        for (var feature : ctx.inlineFeatures) {
            if (feature.isFused()) {
                fusedFeature = (FusedFeature) feature;
            }
        }
        if (fusedFeature == null) {
            return;
        }

        var supplier = suppliers.get(fusedFeature.id());
        if (supplier == null) {
            throw new IllegalStateException("Supplier for feature " + fusedFeature.id() + " not found");
        }

        if (ctx.graph.getMaxLevel() >= 1) {
            int layerSize = ctx.graph.size(1);
            int nodesWritten = 0;
            for (var it = ctx.graph.getNodes(1); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                out.writeInt(ctx.ordinalMapper.oldToNew(originalOrdinal));
                fusedFeature.writeSourceFeature(out, supplier.apply(originalOrdinal));
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer 1 size and features written");
            }
        } else {
            try (var view = ctx.graph.getView()) {
                var en = view.entryNode();
                if (en == null) {
                    return;
                }
                final int originalEntryNode = en.node;
                out.writeInt(ctx.ordinalMapper.oldToNew(originalEntryNode));
                fusedFeature.writeSourceFeature(out, supplier.apply(originalEntryNode));
            }
        }
    }
}
