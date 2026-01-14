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
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Base class for graph index writers that use RandomAccessWriter for output.
 * <p>
 * This class provides common functionality for writers that need random access
 * capabilities, such as seeking to specific positions and rewriting headers.
 * <p>
 * Subclasses include:
 * <ul>
 *   <li>{@link OnDiskGraphIndexWriter} - Sequential writing with header updates</li>
 *   <li>{@link OnDiskParallelGraphIndexWriter} - Parallel writing with async I/O</li>
 * </ul>
 */
public abstract class RandomAccessOnDiskGraphIndexWriter extends AbstractGraphIndexWriter<RandomAccessWriter> {
    protected final long startOffset;

    /**
     * Constructs a RandomAccessOnDiskGraphIndexWriter.
     *
     * @param randomAccessWriter the writer to use for output
     * @param version the format version to write
     * @param startOffset the starting offset in the file
     * @param graph the graph to write
     * @param oldToNewOrdinals mapper for ordinal renumbering
     * @param dimension the vector dimension
     * @param features the features to include
     */
    protected RandomAccessOnDiskGraphIndexWriter(RandomAccessWriter randomAccessWriter,
                                                  int version,
                                                  long startOffset,
                                                  ImmutableGraphIndex graph,
                                                  OrdinalMapper oldToNewOrdinals,
                                                  int dimension,
                                                  EnumMap<FeatureId, Feature> features)
    {
        super(randomAccessWriter, version, graph, oldToNewOrdinals, dimension, features);
        this.startOffset = startOffset;
    }

    /**
     * Close the view and the output stream. For backwards compatibility reasons,
     * this method assumes ownership of the output stream.
     */
    @Override
    public synchronized void close() throws IOException {
        out.close();
    }

    /**
     * Caller should synchronize on this writer instance if mixing usage of the
     * output with calls to any of the synchronized methods in this class.
     * <p>
     * Provided for callers (like Cassandra) that want to add their own header/footer to the output.
     */
    public RandomAccessWriter getOutput() {
        return out;
    }

    /**
     * This method is deprecated. Please use {@link #writeFeaturesInline(int, Map) writeFeaturesInline} instead
     */
    @Deprecated
    public synchronized void writeInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException
    {
        writeFeaturesInline(ordinal, stateMap);
    }

    /**
     * Write the inline features of the given ordinal to the output at the correct offset.
     * Nothing else is written (no headers, no edges).  The output IS NOT flushed.
     * <p>
     * Note: the ordinal given is implicitly a "new" ordinal in the sense of the OrdinalMapper,
     * but since no nodes or edges are involved (we just write the given State to the index file),
     * the mapper is not invoked.
     */
    public synchronized void writeFeaturesInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException {
        for (var featureId : stateMap.keySet()) {
            if (!featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }

        out.seek(featureOffsetForOrdinal(ordinal));

        for (var feature : inlineFeatures) {
            var state = stateMap.get(feature.id());
            if (state == null) {
                out.seek(out.position() + feature.featureSize());
            } else {
                feature.writeInline(out, state);
            }
        }

        maxOrdinalWritten = Math.max(maxOrdinalWritten, ordinal);
    }

    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
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

        var view = graph.getView();

        writeHeader(view); // sets position to start writing features

        writeL0Records(view, featureStateSuppliers);

        // We will use the abstract method because no random access is needed
        writeSparseLevels(view, featureStateSuppliers);

        // We will use the abstract method because no random access is needed
        writeSeparatedFeatures(featureStateSuppliers);

        if (version >= 5) {
            writeFooter(view, out.position());
        }
        final var endOfGraphPosition = out.position();

        // Write the header again with updated offsets
        writeHeader(view);
        out.seek(endOfGraphPosition);
        out.flush();
        view.close();
    }

    protected abstract void writeL0Records(ImmutableGraphIndex.View view,
                                           Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException;

    /**
     * Computes the file offset for the inline features of a given ordinal.
     *
     * @param ordinal the ordinal to compute the offset for
     * @return the file offset
     */
    protected long featureOffsetForOrdinal(int ordinal) {
        return super.featureOffsetForOrdinal(startOffset, ordinal);
    }

    /**
     * Write the index header and completed edge lists to the given output.
     * This method flushes the output and seeks to the startOffset to rewrite the header.
     *
     * @param view the graph index view
     * @throws IOException if there is an error writing the header
     */
    public synchronized void writeHeader(ImmutableGraphIndex.View view) throws IOException {
        out.seek(startOffset);
        super.writeHeader(view, startOffset);
        out.flush();
    }

    /**
     * CRC32 checksum of bytes written since the starting offset.
     *
     * @return the checksum value
     * @throws IOException if an I/O error occurs
     */
    public synchronized long checksum() throws IOException {
        long endOffset = out.position();
        return out.checksum(startOffset, endOffset);
    }
}
