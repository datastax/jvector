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

package io.github.jbellis.jvector.index.graph;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.index.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.index.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.index.graph.disk.feature.Feature;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * A {@link GraphIndex} that can be written to disk.
 * <p>
 * Both {@code OnHeapGraphIndex} (in-memory, potentially still under construction) and
 * {@code OnDiskGraphIndex} (already on disk) implement this interface. Writing is supported
 * via {@link #getWriterBuilder(Path)} (parallel/random-access) and {@link #getWriterBuilder(IndexWriter)}
 * (sequential, e.g. for Cassandra/Lucene integration that requires or prefers sequential I/O).
 * <p>
 * {@link MutableGraphIndex} extends this interface, so an on-heap graph is persistable at any
 * point during construction.
 */
public interface PersistableGraphIndex extends GraphIndex {

    /**
     * Returns a {@link GraphIndexWriterBuilder} configured to write this graph to {@code path} using the
     * parallel writer strategy
     */
    GraphIndexWriterBuilder getParallelWriterBuilder(Path path) throws FileNotFoundException;

    /**
     * Returns a {@link GraphIndexWriterBuilder} configured to write this graph to {@code path} using the
     * sequential writer strategy
     */
    GraphIndexWriterBuilder getWriterBuilder(Path path) throws FileNotFoundException;

    /**
     * Returns a {@link GraphIndexWriterBuilder} that writes this graph sequentially to {@code out}.
     * <p>
     * Sequential writing is suitable for cloud object storage and frameworks such as Lucene
     * that require or prefer sequential I/O. The header is written as a footer; the
     * caller is responsible for flushing {@code out}.
     */
    GraphIndexWriterBuilder getWriterBuilder(IndexWriter out) throws FileNotFoundException;

    /**
     * Fluent builder for persisting a {@link PersistableGraphIndex} to disk.
     * <p>
     * Obtain an instance via {@link PersistableGraphIndex#getWriterBuilder(Path)} or
     * {@link PersistableGraphIndex#getWriterBuilder(IndexWriter)}.
     */
    interface GraphIndexWriterBuilder {
        /** Adds a feature to be written with this graph. */
        GraphIndexWriterBuilder with(Feature feature);

        /** Sets the ordinal mapper used to renumber node ids on write. */
        GraphIndexWriterBuilder withMapper(OrdinalMapper mapper);

        /** Convenience for {@link #withMapper} using a pre-computed old-to-new mapping. */
        GraphIndexWriterBuilder withMap(Map<Integer, Integer> oldToNew);

        /** Sets the on-disk format version (defaults to the current version). */
        GraphIndexWriterBuilder withVersion(int version);

        /**
         * Sets the byte offset at which writing begins in the output file.
         * Useful when appending a graph to an existing file.
         */
        GraphIndexWriterBuilder withStartOffset(long offset);

        /**
         * Sets the number of worker threads for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         *
         * @param n number of threads; negative means use all available processors; 0 (default) disables parallel writes
         */
        GraphIndexWriterBuilder withParallelWorkerThreads(int n);

        /**
         * Whether to use direct ByteBuffers for parallel writes.
         * Ignored (with a WARN log) when the underlying graph is not an in-memory graph.
         */
        GraphIndexWriterBuilder withParallelDirectBuffers(boolean useDirectBuffers);

        /** Builds the graph index writer. */
        GraphIndexWriter build() throws IOException;
    }

    static String prettyPrint(PersistableGraphIndex graph) {
        return GraphIndex.prettyPrint(graph);
    }
}