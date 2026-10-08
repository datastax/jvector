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
 * One write operation against an {@link IndexDestination}: reserves the output for each
 * artifact it produces and ends in exactly one of {@link #commit()} or an abort (a
 * {@link #close()} without a prior successful commit).
 * <p>
 * Lifecycle calls ({@code reserve}, {@code commit}, {@code close}) are made by the thread that
 * owns the operation, after any worker threads using the vended streams have finished. Streams
 * vended by this session are released by their reservation's {@code close()} or, at the latest,
 * by this session's {@code close()}.
 * <p>
 * State machine: {@code OPEN} until {@code commit()} succeeds ({@code COMMITTED}) or fails
 * ({@code ABORTED}) or {@code close()} runs first ({@code ABORTED}); {@code close()} is idempotent
 * and every other call on a non-open session fails with {@link IllegalStateException}.
 */
@Experimental
public interface OutputSession extends AutoCloseable {

    /**
     * Reserves the output for one artifact. Each artifact may be reserved at most once per
     * session; a second request fails with {@link IllegalStateException}.
     *
     * @throws IllegalArgumentException if this destination has no placement for {@code artifact}
     */
    OutputReservation reserve(OutputArtifact artifact) throws IOException;

    /**
     * Publishes every completed artifact as one consistent unit, as atomically as the host can
     * (rename into place, register components, mark a segment's file set). Precondition: every
     * reservation made in this session has been completed and closed. At most once; if it
     * throws, the session is aborted.
     *
     * @throws IllegalStateException if a reservation is still open, or a reserved artifact was
     *                               never completed
     */
    void commit() throws IOException;

    /**
     * Idempotent. Without a prior successful {@link #commit()} this is an abort: every stream
     * still open is released and every partial artifact is discarded. Must succeed from any
     * state, including before the first reservation and after a failed commit. An
     * {@link IOException} raised here during an abort is attached as suppressed to the failure
     * that caused the abort.
     */
    @Override
    void close() throws IOException;
}
