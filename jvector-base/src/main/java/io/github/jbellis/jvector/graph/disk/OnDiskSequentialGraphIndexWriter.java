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
import io.github.jbellis.jvector.graph.disk.feature.*;

import java.io.IOException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Writes a graph index to disk in a format that can be loaded as an OnDiskGraphIndex.
 * <p>
 * Unlike {@link OnDiskGraphIndexWriter}, this class always writes in a sequential order and the header metadata is written as the footer.
 * <p>
 * Assumptions:
 * <ul>
 * <li> The graph already exists and is not modified as it is being written, and therefore can be written sequentially in a single pass.
 * <li> This assumption is valid for two common cases in Log Structure Merge Tree-based systems such as Cassandra and Lucene:
 *   <ol>
 *   <li> The graph is being written as part of compaction
 *   <li> The graph is being written for addition of a small immutable segment.
 *   </ol>
 * </ul>
 * <p>
 * Goals:
 * <ul>
 * <li> Immutability: Every byte written to the index file is immutable. This allows for running calculation of checksums without needing to re-read the file.
 * <li> Performance: We can take advantage of sequential writes for performance.
 * </ul>
 * <p>
 * The above goals are driven by the following motivations:
 * <ul>
 * <li> When we work with either cloud object storage where random writes are not supported on a single stream
 * <li> When we embed jVector in frameworks such as Lucene that rely on sequential writes for performance and correctness
 * </ul>
 */
public class OnDiskSequentialGraphIndexWriter extends AbstractGraphIndexWriter<IndexWriter> {
    public static final int FOOTER_MAGIC = 0x4a564244; // "EOF magic"
    public static final int FOOTER_OFFSET_SIZE = Long.BYTES; // The size of the offset in the footer
    public static final int FOOTER_MAGIC_SIZE = Integer.BYTES; // The size of the magic number in the footer
    public static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE; // The total size of the footer

    OnDiskSequentialGraphIndexWriter(IndexWriter out,
                                             int version,
                                             GraphIndex graph,
                                             OrdinalMapper oldToNewOrdinals,
                                             int dimension,
                                             EnumMap<FeatureId, Feature> features)
    {
        super(out, version, graph, oldToNewOrdinals, dimension, features);
    }

    @Override
    public synchronized void close() throws IOException {
        view.close();
        // Note: we don't close the output streams since we don't own them in this writer
    }

    /**
     * Note: There are several limitations you should be aware of when using:
     * <ul>
     * <li> This method doesn't persist (e.g. flush) the output streams.  The caller is responsible for doing so.
     * <li> This method does not support writing to "holes" in the ordinal space.  If your ordinal mapper
     *      maps a new ordinal to an old ordinal that does not exist in the graph, an exception will be thrown.
     * </ul>
     */
    @Override
    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
        final var startOffset = out.position();
        writeHeader(startOffset);
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : featureStateSuppliers.keySet()) {
            if (!featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ordinalMapper.maxOrdinal() < graph.size(0) - 1) {
            var msg = String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ordinalMapper.maxOrdinal(), graph.size(0));
            throw new IllegalStateException(msg);
        }

        // for each graph node, write the associated features, followed by its neighbors at L0
        for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            // if no node exists with the given ordinal, write a placeholder
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                throw new IllegalStateException("Ordinal mapper mapped new ordinal" + newOrdinal + " to non-existing node. This behavior is not supported on OnDiskSequentialGraphIndexWriter. Use OnDiskGraphIndexWriter instead.");
            }

            if (!graph.containsNode(originalOrdinal)) {
                var msg = String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal);
                throw new IllegalStateException(msg);
            }
            out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            assert out.position() == featureOffsetForOrdinal(startOffset, newOrdinal) : String.format("%d != %d", out.position(), featureOffsetForOrdinal(startOffset, newOrdinal));
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + feature.id() + " not found");
                } else {
                    feature.writeInline(out, supplier.apply(originalOrdinal));
                }
            }

            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                var msg = String.format("Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                                        originalOrdinal, neighbors.size(), graph.getDegree(0));
                throw new IllegalStateException(msg);
            }
            // write neighbors list
            out.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal > ordinalMapper.maxOrdinal()) {
                    var msg = String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, ordinalMapper.maxOrdinal());
                    throw new IllegalStateException(msg);
                }
                out.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.getDegree(0); n++) {
                out.writeInt(-1);
            }
        }

        writeSparseLevels();

        writeSeparatedFeatures(featureStateSuppliers);

        // Write the footer with all the metadata info about the graph
        writeFooter(out.position());
        // Note: flushing the data output is the responsibility of the caller we are not going to make assumptions about further uses of the data outputs
    }

    /**
     * Builder for {@link OnDiskSequentialGraphIndexWriter}, with optional features.
     */
    public static class Builder extends AbstractGraphIndexWriter.Builder<OnDiskSequentialGraphIndexWriter, IndexWriter> {
        public Builder(GraphIndex graphIndex, IndexWriter out) {
            super(graphIndex, out);
        }

        @Override
        protected OnDiskSequentialGraphIndexWriter reallyBuild(int dimension) {
            return new OnDiskSequentialGraphIndexWriter(out, version, graphIndex, ordinalMapper, dimension, features);

        }
    }
}