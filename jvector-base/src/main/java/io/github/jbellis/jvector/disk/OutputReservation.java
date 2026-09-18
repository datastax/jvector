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

/**
 * One artifact's reserved output plus its completion lifecycle: live, single-use state made
 * against an {@link OutputSession}, fulfilled by {@link #complete()} or released by a
 * {@link #close()} without one.
 * <p>
 * The artifact is written strictly sequentially through {@link #stream()}. After
 * {@link #complete} or {@link #close}, every other call fails with {@link IllegalStateException}.
 */
@Experimental
public interface OutputReservation extends AutoCloseable {

    /** The artifact this reservation holds. */
    OutputArtifact artifact();

    /**
     * The append-only output for this artifact; the same instance on every call. Its
     * {@code position()} counts the bytes written through it, starting at {@code 0}. Every offset
     * jvector stores inside the artifact (the footer's header offset, separated-feature offsets)
     * is relative to that origin, so a host reads the artifact back with a reader whose origin is
     * the same place. Single-threaded. Closing it flushes and does not close the reservation, so
     * a jvector writer that takes ownership of its output may close it freely.
     */
    IndexWriter stream() throws IOException;

    /**
     * Marks the artifact final: flushes anything still buffered in {@link #stream()} and hands
     * the artifact to the host, which finalizes it here (footer, checksum, durability). At most
     * once; a failure leaves the reservation not completed, which prevents the session from
     * committing.
     *
     * @throws IllegalStateException if already completed or closed
     */
    void complete() throws IOException;

    /**
     * Idempotent. Releases the stream. Without a prior successful {@link #complete}, the artifact
     * is discarded and the owning session can no longer commit.
     */
    @Override
    void close() throws IOException;
}
