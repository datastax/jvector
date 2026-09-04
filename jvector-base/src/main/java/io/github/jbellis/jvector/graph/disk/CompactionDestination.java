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
import io.github.jbellis.jvector.disk.SeekableSink;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Embedding extension point: tells {@link OnDiskGraphIndexCompactor} WHERE to write its compacted
 * graph, so the body lands directly inside the embedder's container (after a header the embedder
 * reserves) — eliminating the temp-file-and-copy. The destination itself is stateless
 * configuration an embedder can build once and hold; each {@code compact(...)} call
 * {@link #reserve() reserves} its own {@link OutputReservation}, commits it on success, and always
 * closes it.
 *
 * <pre>{@code
 *   try (CompactionDestination.OutputReservation r = destination.reserve()) {
 *       // ...compactor writes the graph into r.file() at r.startOffset()...
 *       r.commit(bodyLength);   // success: body written & durable; embedder finalizes its footer
 *   }                           // close() always runs; no commit() => reservation released, partial output discarded
 * }</pre>
 *
 * <p>The compactor needs a real file (it uses a memory-mapped read-back during refinement and a
 * random-access writer), so an {@link OutputReservation} is expressed as a container {@link Path}
 * plus a base offset rather than an opaque stream. The generic {@link SeekableSink} primitive
 * addresses the same window in region-relative coordinates and is what an embedder uses to read the
 * committed body back for its checksum, e.g. {@code SeekableSink.over(channel, r.startOffset())}.
 */
@FunctionalInterface
@Experimental
public interface CompactionDestination {

    /** Reserve a fresh output region for one compaction. */
    OutputReservation reserve() throws IOException;

    /**
     * One compaction's reserved output region plus its commit/abort lifecycle: live, single-use
     * state made against the destination, fulfilled by {@link #commit} or released by a
     * {@link #close} without one.
     */
    interface OutputReservation extends AutoCloseable {

        /** The container file the graph body is written into. */
        Path file();

        /** The byte offset within {@link #file()} at which the graph body begins ({@code >= 0}). */
        long startOffset();

        /**
         * Fulfills the reservation. Signalled exactly once, after the body has been fully written
         * and forced, reporting its length ({@code file() size - startOffset()}). The embedder
         * finalizes its container here (e.g. writes a footer/checksum). MUST be called before
         * {@link #close()} on the success path.
         */
        void commit(long bodyLength) throws IOException;

        /**
         * Always runs (try-with-resources). If reached without a prior {@link #commit}, the
         * compaction failed: the reservation is released, the embedder discards the partial
         * output, and embedder resources are freed.
         */
        @Override
        void close() throws IOException;
    }

    /**
     * Default standalone destination: writes to its own file at offset {@code 0} (today's
     * {@code compact(Path)} behaviour). {@code commit} is a no-op marker; a {@code close} without a
     * prior commit deletes the partial file.
     */
    static CompactionDestination toFile(Path path) {
        return new FileCompactionDestination(path);
    }
}
