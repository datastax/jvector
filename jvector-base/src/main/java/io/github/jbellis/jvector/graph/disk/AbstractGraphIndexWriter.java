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
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;
import org.agrona.collections.Int2IntHashMap;

import java.io.IOException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public abstract class AbstractGraphIndexWriter implements  GraphIndexWriter {
    public static final int FOOTER_MAGIC = 0x4a564244; // "EOF magic"
    public static final int FOOTER_OFFSET_SIZE = Long.BYTES; // The size of the offset in the footer
    public static final int FOOTER_MAGIC_SIZE = Integer.BYTES; // The size of the magic number in the footer
    public static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE; // The total size of the footer
    final int version;
    final GraphIndex graph;
    final GraphIndex.View view;
    final OrdinalMapper ordinalMapper;
    final int dimension;
    // we don't use Map features but EnumMap is the best way to make sure we don't
    // accidentally introduce an ordering bug in the future
    final EnumMap<FeatureId, Feature> featureMap;
    final IndexWriter out; /* output for graph nodes and inline features */
    final int headerSize;
    volatile int maxOrdinalWritten = -1;
    final List<Feature> inlineFeatures;

    AbstractGraphIndexWriter(IndexWriter out,
                                     int version,
                                     GraphIndex graph,
                                     OrdinalMapper oldToNewOrdinals,
                                     int dimension,
                                     EnumMap<FeatureId, Feature> features)
    {
        if (graph.getMaxLevel() > 0 && version < 4) {
            throw new IllegalArgumentException("Multilayer graphs must be written with version 4 or higher");
        }
        this.version = version;
        this.graph = graph;
        this.view = graph instanceof OnHeapGraphIndex ? ((OnHeapGraphIndex) graph).getFrozenView() : graph.getView();
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.featureMap = features;
        this.inlineFeatures = features.values().stream().filter(f -> !(f instanceof SeparatedFeature)).collect(Collectors.toList());
        this.out = out;

        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    /**
     * @return the maximum ordinal written so far, or -1 if no ordinals have been written yet
     */
    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    public Set<FeatureId> getFeatureSet() {
        return featureMap.keySet();
    }

    long featureOffsetForOrdinal(long startOffset, int ordinal) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset
                + headerSize
                + inlineBytes // previous nodes
                + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.  "Holes" left by
     * deleted nodes are filled in by shifting down the new ordinals.
     */
    public static Map<Integer, Integer> sequentialRenumbering(GraphIndex graph) {
        try (var view = graph.getView()) {
            Int2IntHashMap oldToNewMap = new Int2IntHashMap(-1);
            int nextOrdinal = 0;
            for (int i = 0; i < view.getIdUpperBound(); i++) {
                if (graph.containsNode(i)) {
                    oldToNewMap.put(i, nextOrdinal++);
                }
            }
            return oldToNewMap;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Write the {@link Header} as a footer for the graph index.
     * <p>
     * To read the graph later, we will perform the following steps:
     * <ol>
     *     <li> Find the magic number at the end of the slice
     *     <li> Read the header offset from the end of the slice
     *     <li> Read the header
     *     <li> Read the neighbors offsets and graph metadata
     * </ol>
     * @param headerOffset the offset of the header in the slice
     * @throws IOException IOException
     */
    void writeFooter(long headerOffset) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out); // write the header
        out.writeLong(headerOffset); // We write the offset of the header at the end of the file
        out.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert out.position() == expectedPosition : String.format("%d != %d", out.position(), expectedPosition);
    }

    /**
     * Writes the index header, including the graph size, so that OnDiskGraphIndex can open it.
     * The output IS flushed.
     * <p>
     * Public so that you can write the index size (and thus usefully open an OnDiskGraphIndex against the index)
     * to read Features from it before writing the edges.
     */
    public synchronized void writeHeader(long startOffset) throws IOException {
        // graph-level properties
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out);
        assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
    }
}
