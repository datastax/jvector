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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * An {@link ImmutableGraphIndex} that can be written to disk.
 * <p>
 * Both {@code OnHeapGraphIndex} (in-memory, potentially still under construction) and
 * {@code OnDiskGraphIndex} (already on disk) implement this interface. Writing is supported
 * via {@link #writer(Path)} (parallel/random-access) and {@link #writer(IndexWriter)}
 * (sequential, e.g. for Cassandra/Lucene integration that requires or prefers sequential I/O).
 * <p>
 * {@link MutableGraphIndex} extends this interface, so an on-heap graph is persistable at any
 * point during construction.
 */
public interface PersistableGraphIndex extends ImmutableGraphIndex {

    /**
     * Returns a {@link WriteBuilder} configured to write this graph to {@code path} using the
     * default (parallel/random-access) writer strategy for this index type.
     * <p>
     * Call configuration methods on the returned builder before invoking {@link WriteBuilder#write}.
     */
    default WriteBuilder writer(Path path) throws FileNotFoundException {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support path-based writing");
    }

    /**
     * Returns a {@link WriteBuilder} that writes this graph sequentially to {@code out}.
     * <p>
     * Sequential writing is suitable for cloud object storage and frameworks such as Lucene or
     * Cassandra that require or prefer sequential I/O. The header is written as a footer; the
     * caller is responsible for flushing {@code out}.
     */
    default WriteBuilder writer(IndexWriter out) throws FileNotFoundException {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support sequential writing");
    }

    /**
     * Fluent builder for persisting a {@link PersistableGraphIndex} to disk.
     * <p>
     * Obtain an instance via {@link PersistableGraphIndex#writer(Path)} or
     * {@link PersistableGraphIndex#writer(IndexWriter)}. Configuration methods must be called
     * before the first call to {@link #writeFeaturesInline} or {@link #write}.
     */
    interface WriteBuilder extends Closeable {
        /** Adds a feature to be written with this graph. */
        WriteBuilder with(Feature feature);

        /** Sets the ordinal mapper used to renumber node ids on write. */
        WriteBuilder withMapper(OrdinalMapper mapper);

        /** Convenience for {@link #withMapper} using a pre-computed old-to-new mapping. */
        WriteBuilder withMap(Map<Integer, Integer> oldToNew);

        /** Sets the on-disk format version (defaults to the current version). */
        WriteBuilder withVersion(int version);

        /**
         * Sets the byte offset at which writing begins in the output file.
         * Useful when appending a graph to an existing file.
         */
        WriteBuilder withStartOffset(long offset);

        /**
         * Sets the number of worker threads for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         *
         * @param n number of threads; negative means use all available processors; 0 (default) disables parallel writes
         */
        WriteBuilder withParallelWorkerThreads(int n);

        /**
         * Whether to use direct ByteBuffers for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         */
        WriteBuilder withParallelDirectBuffers(boolean useDirectBuffers);

        /**
         * Writes the inline features for a single node ordinal without writing graph structure.
         * Used for incremental (node-at-a-time) construction patterns.
         * Must be called after all configuration methods and before {@link #write}.
         */
        WriteBuilder writeFeaturesInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException;

        /**
         * Writes the complete graph header, edge lists, and any remaining features to the
         * configured output. Closes the underlying output stream when done.
         *
         * @param featureStateSuppliers per-node suppliers for each configured feature; features
         *                              already written via {@link #writeFeaturesInline} can be
         *                              omitted from this map
         */
        void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException;

        /** Writes the graph header/metadata at the current position. */
        void writeHeader(ImmutableGraphIndex.View view) throws IOException;

        /** @return the CRC32 checksum of all bytes written since the start offset */
        long checksum() throws IOException;

        @Override
        void close() throws IOException;
    }

    /** @deprecated use {@link ImmutableGraphIndex#prettyPrint(ImmutableGraphIndex)} */
    @Deprecated
    static String prettyPrint(PersistableGraphIndex graph) {
        return ImmutableGraphIndex.prettyPrint(graph);
    }
}
