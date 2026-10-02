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
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * A {@link GraphIndex} that can be written to disk.
 * <p>
 * Both {@code OnHeapGraphIndex} (in-memory, potentially still under construction) and
 * {@code OnDiskGraphIndex} (already on disk) implement this interface. All three accessors produce the
 * same on-disk format, loadable with {@code OnDiskGraphIndex.load}; they differ in what output they
 * need and how they write it:
 * <ul>
 *     <li>{@link #getWriterBuilder(Path)}: random-access writer ({@code OnDiskGraphIndexWriter},
 *     {@code GraphIndexWriterTypes.RANDOM_ACCESS}). Writes node records in order on a single thread,
 *     then seeks back to fill in the header.</li>
 *     <li>{@link #getParallelWriterBuilder(Path)}: parallel random-access writer
 *     ({@code OnDiskParallelGraphIndexWriter}, {@code GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL}).
 *     Encodes node records on worker threads and writes them with an {@code AsynchronousFileChannel}.</li>
 *     <li>{@link #getWriterBuilder(IndexWriter)}: sequential writer
 *     ({@code OnDiskSequentialGraphIndexWriter}, {@code GraphIndexWriterTypes.ON_DISK_SEQUENTIAL}).
 *     One forward pass that never seeks, for append-only outputs such as a Lucene {@code IndexOutput}
 *     or cloud object storage.</li>
 * </ul>
 * <p>
 * The graph a {@link GraphIndexBuilder} or {@link HnswIndexBuilder} builds implements this interface,
 * so it is persistable at any point during construction.
 * <p>
 * The {@code Path} accessors don't open the file: the writer builder's {@code build()} opens it, and
 * the {@link GraphIndexWriter} it returns closes it. A writer builder that is never built, or whose
 * {@code build()} fails, leaves no file open.
 */
public interface PersistableGraphIndex extends GraphIndex {

    /**
     * Returns a {@link GraphIndexWriterBuilder} that writes this graph to {@code path} with the parallel
     * random-access writer ({@code OnDiskParallelGraphIndexWriter}): node records are encoded on worker
     * threads and written asynchronously. Supports {@link GraphIndexWriterBuilder#withStartOffset},
     * {@link GraphIndexWriterBuilder#withParallelWorkerThreads} and
     * {@link GraphIndexWriterBuilder#withParallelDirectBuffers}.
     */
    GraphIndexWriterBuilder getParallelWriterBuilder(Path path) throws FileNotFoundException;

    /**
     * Returns a {@link GraphIndexWriterBuilder} that writes this graph to {@code path} with the
     * single-threaded random-access writer ({@code OnDiskGraphIndexWriter}): node records are written in
     * order, then the writer seeks back to fill in the header. Supports
     * {@link GraphIndexWriterBuilder#withStartOffset}; the parallel options throw
     * {@link UnsupportedOperationException}. For a writer that never seeks, use
     * {@link #getWriterBuilder(IndexWriter)}.
     */
    GraphIndexWriterBuilder getWriterBuilder(Path path) throws FileNotFoundException;

    /**
     * Returns a {@link GraphIndexWriterBuilder} that writes this graph sequentially to {@code out}.
     * <p>
     * Sequential writing is suitable for cloud object storage and frameworks such as Lucene
     * that require or prefer sequential I/O. The header is written as a footer; the
     * caller owns {@code out} and is responsible for flushing and closing it. Writing starts at
     * {@code out}'s current position: {@link GraphIndexWriterBuilder#withStartOffset} and the parallel
     * options throw {@link UnsupportedOperationException}.
     */
    GraphIndexWriterBuilder getWriterBuilder(IndexWriter out);

    /**
     * Fluent builder for persisting a {@link PersistableGraphIndex} to disk.
     * <p>
     * Obtain an instance via {@link PersistableGraphIndex#getWriterBuilder(Path)},
     * {@link PersistableGraphIndex#getParallelWriterBuilder(Path)} or
     * {@link PersistableGraphIndex#getWriterBuilder(IndexWriter)}. Not every option applies to every
     * writer; unsupported options throw {@link UnsupportedOperationException} rather than being
     * silently ignored.
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
         *
         * @throws UnsupportedOperationException for the sequential writer, which always starts at the
         *         output's current position
         */
        GraphIndexWriterBuilder withStartOffset(long offset);

        /**
         * Sets the number of worker threads for parallel writes. Only supported by the parallel writer
         * ({@link PersistableGraphIndex#getParallelWriterBuilder(Path)}).
         *
         * @param n number of threads; 0 (the default) or negative means use all available processors
         * @throws UnsupportedOperationException for the random-access and sequential writers
         */
        GraphIndexWriterBuilder withParallelWorkerThreads(int n);

        /**
         * Whether to use direct (off-heap) ByteBuffers for parallel writes. Only supported by the
         * parallel writer ({@link PersistableGraphIndex#getParallelWriterBuilder(Path)}).
         *
         * @throws UnsupportedOperationException for the random-access and sequential writers
         */
        GraphIndexWriterBuilder withParallelDirectBuffers(boolean useDirectBuffers);

        /** Builds the graph index writer. */
        GraphIndexWriter build() throws IOException;
    }
}