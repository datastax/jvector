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

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Format for version 5 of the on-disk graph format.
 * Version 5 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata (major change from V4)
 *
 * The wire format is identical to V4; the only behavioral difference is that
 * a footer is written after the graph data.
 */
class GraphIndexFormatV5 extends GraphIndexFormatV4 {

    /** Creates the singleton format for version 5. */
    GraphIndexFormatV5() {
        super(5, nonFusedFeatures(), true, FeatureOrdering.BY_FEATURE_ID);
    }

    /**
     * Protected constructor for subclasses (V6) to specify their own version,
     * feature set, and feature ordering while inheriting V5's footer-writing behavior.
     * Footer is always true for V5 and later.
     */
    protected GraphIndexFormatV5(int version, Set<FeatureId> supportedFeatures, FeatureOrdering featureOrdering) {
        super(version, supportedFeatures, true, featureOrdering);
    }

    /**
     * Writes the complete graph index using random-access I/O, and additionally appends a
     * footer (header offset + magic number) after the graph data — the key behavioral difference
     * from the V4 implementation, which does not write a footer.
     *
     * {@inheritDoc}
     */
    @Override
    public void writeRandomAccess(WriteContext ctx, RandomAccessWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers, GraphIndexFormat.L0RecordWriter l0Writer) throws IOException {
        if (ctx.graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) ctx.graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : suppliers.keySet()) {
            if (!ctx.featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ctx.ordinalMapper.maxOrdinal() < ctx.graph.size(0) - 1) {
            throw new IllegalStateException(String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ctx.ordinalMapper.maxOrdinal(), ctx.graph.size(0)));
        }

        var view = ctx.graph.getView();

        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        l0Writer.write(view, suppliers);
        writeSparseLevels(ctx, out, suppliers);
        writeSeparatedFeatures(ctx, out, suppliers);
        writeFooter(ctx, out.position(), out);

        final var endOfGraphPosition = out.position();
        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        out.seek(endOfGraphPosition);
        out.flush();
        view.close();
    }
}
