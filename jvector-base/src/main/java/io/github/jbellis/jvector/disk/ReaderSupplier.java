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

import java.io.IOException;

/**
 * A supplier of RandomAccessReaders.
 */
public interface ReaderSupplier extends AutoCloseable {
    /**
     * Creates a new reader.
     * @return a new reader.  It is up to the caller to re-use these readers or close them,
     * the ReaderSupplier is not responsible for caching them.
     * @throws IOException if an I/O error occurs
     */
    RandomAccessReader get() throws IOException;

    /**
     * Streams the byte range {@code [offset, offset + length)} of the underlying storage into
     * the page cache, if the implementation supports it. Synchronous and best-effort; a no-op
     * by default. Sized for repeated windowed calls — a caller that knows which records a bulk
     * phase is about to read (e.g. graph compaction, whose readers otherwise advise
     * {@code MADV_RANDOM} and forgo kernel readahead) can warm exactly those, keeping transient
     * cache demand proportional to the window rather than the file.
     */
    default void prefetch(long offset, long length) {
    }

    /**
     * Releases the supplier's underlying resource. Two implementation families exist, with very
     * different safety under concurrency:
     * <ul>
     * <li><b>Coordinated</b> (e.g. the jvector-native {@code MemorySegmentReader.Supplier}, whose
     * shared-Arena close performs a liveness handshake): closing while vended readers are still
     * in use degrades to {@code IllegalStateException} on those readers.</li>
     * <li><b>Raw-release</b> (e.g. {@link SimpleMappedReader.Supplier}, which unmaps
     * immediately): closing while any vended reader is mid-read invalidates the mapped pages
     * underneath it, and the JVM fails with a native fault (SIGSEGV) rather than an
     * exception.</li>
     * </ul>
     * Callers must not close a supplier until every reader vended by {@link #get()} is provably
     * quiescent; implementations should document which family they belong to.
     *
     * @throws IOException if an I/O error occurs
     */
    default void close() throws IOException {
    }
}
