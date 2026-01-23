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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Writes a graph index to disk in a format that can be loaded as an OnDiskGraphIndex.
 * <p>
 * The serialization process follows these steps:
 * <p>
 * 1. File Layout:
 *    - CommonHeader: Contains version, dimension, entry node, and layer information
 *    - Header with Features: Contains feature-specific headers
 *    - Layer 0 data: Contains node ordinals, inline features, and edges for all nodes
 *    - Higher layer data (levels 1..N): Contains sparse node ordinals and edges
 *    - Separated features: Contains feature data stored separately from nodes
 * <p>
 * 2. Serialization Process:
 *    - First, a placeholder header is written to reserve space
 *    - For each node in layer 0:
 *      - Write node ordinal
 *      - Write inline features (vectors, quantized data, etc.)
 *      - Write neighbor count and neighbor ordinals
 *    - For each higher layer (1..N):
 *      - Write only nodes that exist in that layer
 *      - For each node: write ordinal, neighbor count, and neighbor ordinals
 *    - For each separated feature:
 *      - Write feature data for all nodes sequentially
 *    - Finally, rewrite the header with correct offsets
 * <p>
 * 3. Ordinal Mapping:
 *    - The writer uses an OrdinalMapper to map between original node IDs and 
 *      the sequential IDs used in the on-disk format
 *    - This allows for compaction (removing "holes" from deleted nodes)
 *    - It also enables custom ID mapping schemes for specific use cases
 * <p>
 * The class supports incremental writing through the writeInline method, which
 * allows writing features for individual nodes without writing the entire graph.
 */
public class OnDiskParallelGraphIndexWriter extends RandomAccessOnDiskGraphIndexWriter {
    private final Path filePath;
    private final int parallelWorkerThreads;
    private final boolean parallelUseDirectBuffers;

    /**
     * Constructs an OnDiskParallelGraphIndexWriter with all parameters including optional file path
     * and parallel write configuration.
     *
     * @param randomAccessWriter the writer to use for output
     * @param version the format version to write
     * @param startOffset the starting offset in the file
     * @param graph the graph to write
     * @param oldToNewOrdinals mapper for ordinal renumbering
     * @param dimension the vector dimension
     * @param features the features to include
     * @param filePath file path required for parallel writes (can be null for sequential writes)
     * @param parallelWorkerThreads number of worker threads for parallel writes (0 = use available processors)
     * @param parallelUseDirectBuffers whether to use direct ByteBuffers for parallel writes
     */
    OnDiskParallelGraphIndexWriter(RandomAccessWriter randomAccessWriter,
                                   int version,
                                   long startOffset,
                                   ImmutableGraphIndex graph,
                                   OrdinalMapper oldToNewOrdinals,
                                   int dimension,
                                   EnumMap<FeatureId, Feature> features,
                                   Path filePath,
                                   int parallelWorkerThreads,
                                   boolean parallelUseDirectBuffers)
    {
        super(randomAccessWriter, version, startOffset, graph, oldToNewOrdinals, dimension, features);
        this.filePath = filePath;
        this.parallelWorkerThreads = parallelWorkerThreads;
        this.parallelUseDirectBuffers = parallelUseDirectBuffers;
    }

    /**
     * Writes L0 records using parallel workers with asynchronous file I/O.
     * <p>
     * Records are written asynchronously using AsynchronousFileChannel for improved throughput
     * while maintaining correct ordering. This method parallelizes record building across
     * multiple threads and writes results in sequential order.
     * <p>
     * Requires filePath to have been provided during construction.
     *
     * @param featureStateSuppliers suppliers for feature state data
     * @throws IOException if an I/O error occurs
     */
    @Experimental
    @Override
    protected void writeL0Records(ImmutableGraphIndex.View view,
                                Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        if (filePath == null) {
            throw new IllegalStateException("Parallel writes require a file path. Use Builder(ImmutableGraphIndex, Path) constructor.");
        }

        // Flush writer before async writes to ensure buffered data is on disk
        // This is critical when using AsynchronousFileChannel in parallel with BufferedRandomAccessWriter
        out.flush();
        long baseOffset = out.position();

        var config = new ParallelGraphWriter.Config(
            parallelWorkerThreads,
            parallelUseDirectBuffers,
            4  // Default task multiplier (4x cores)
        );

        try (var parallelWriter = new ParallelGraphWriter(
                out,
                graph,
                inlineFeatures,
                config,
                filePath)) {

            parallelWriter.writeL0Records(
                ordinalMapper,
                inlineFeatures,
                featureStateSuppliers,
                baseOffset
            );

            // Update maxOrdinalWritten
            maxOrdinalWritten = ordinalMapper.maxOrdinal();

            // Seek to end of L0 region
            long endOffset = baseOffset + (long) (ordinalMapper.maxOrdinal() + 1) * parallelWriter.getRecordSize();
            out.seek(endOffset);
        }
    }


    /**
     * Builder for {@link OnDiskParallelGraphIndexWriter}, with optional features.
     */
    public static class Builder extends AbstractGraphIndexWriter.Builder<OnDiskParallelGraphIndexWriter, RandomAccessWriter> {
        private long startOffset = 0L;
        private final Path filePath;
        private int parallelWorkerThreads = 0;
        private boolean parallelUseDirectBuffers = false;

        public Builder(ImmutableGraphIndex graphIndex, Path outPath) throws FileNotFoundException {
            super(graphIndex, new BufferedRandomAccessWriter(outPath));
            this.filePath = outPath;
        }

        /**
         * Set the starting offset for the graph index in the output file.  This is useful if you want to
         * append the index to an existing file.
         */
        public Builder withStartOffset(long startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        /**
         * Set the number of worker threads for parallel writes.
         *
         * @param workerThreads number of worker threads (0 = use available processors)
         * @return this builder
         */
        public Builder withParallelWorkerThreads(int workerThreads) {
            this.parallelWorkerThreads = workerThreads;
            return this;
        }

        /**
         * Set whether to use direct ByteBuffers for parallel writes.
         * Direct buffers can provide better performance for large records but use off-heap memory.
         *
         * @param useDirectBuffers whether to use direct ByteBuffers
         * @return this builder
         */
        public Builder withParallelDirectBuffers(boolean useDirectBuffers) {
            this.parallelUseDirectBuffers = useDirectBuffers;
            return this;
        }

        @Override
        protected OnDiskParallelGraphIndexWriter reallyBuild(int dimension) {
            return new OnDiskParallelGraphIndexWriter(
                out, version, startOffset, graphIndex, ordinalMapper, dimension, features, filePath,
                parallelWorkerThreads, parallelUseDirectBuffers
            );
        }
    }
}