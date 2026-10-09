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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.annotations.Experimental;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * A seekable region that supports positional reads and writes, addressed in coordinates relative
 * to the region's start (0-based). An embedder uses it to hand a compactor (or other writer) a
 * bounded window inside a larger container file: positions are region-relative and the
 * implementation adds the container's base offset, so the writer never needs to know the absolute
 * offset.
 *
 * <p>Implementations must support concurrent positional writes and reads to disjoint ranges (a
 * {@link FileChannel} does). This is a generic IO primitive; the compaction extension point that
 * hands one out is {@code io.github.jbellis.jvector.graph.disk.CompactionDestination}.
 */
@Experimental
public interface SeekableSink extends AutoCloseable {

    /** Write {@code src} fully at region-relative {@code position} (must be {@code >= 0}). */
    void writeAt(long position, ByteBuffer src) throws IOException;

    /**
     * Read up to {@code dst.remaining()} bytes at region-relative {@code position} (must be
     * {@code >= 0}); returns the number of bytes read, or {@code -1} at end of region.
     */
    int readAt(long position, ByteBuffer dst) throws IOException;

    /** Force written bytes to durable storage. */
    void force() throws IOException;

    @Override
    void close() throws IOException;

    /**
     * Reference implementation over a {@link FileChannel} region. Every region-relative position is
     * translated by {@code baseOffset}. The channel's lifecycle is owned by the caller — this
     * {@link #close()} does <b>not</b> close the channel.
     *
     * @param channel    the backing channel, opened for read and write
     * @param baseOffset the absolute offset of the region's start within {@code channel} ({@code >= 0})
     */
    static SeekableSink over(FileChannel channel, long baseOffset) {
        return new FileChannelSeekableSink(channel, baseOffset);
    }
}
