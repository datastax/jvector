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

package io.github.jbellis.jvector.util.work;

import io.github.jbellis.jvector.annotations.Experimental;

/**
 * Admission contract: blocks until an amount of work may proceed, returning a {@link Grant} that
 * the caller closes once the admitted work has completed.
 *
 * <p>The unit of {@code amount} is <b>defined by the consumer</b> (e.g. bytes for IO, or rows,
 * nodes, items); jvector fixes only the blocking-grant mechanism, never the meaning of the
 * quantity. Implementations must be thread-safe and reentrant. {@code acquire} may block but must
 * not throw for ordinary back-pressure.
 */
@Experimental
@FunctionalInterface
public interface WorkLimiter {
    /**
     * Blocks until {@code amount} units of work may proceed.
     *
     * @param amount the amount of work about to be performed, in consumer-defined units
     * @return a non-null grant to {@link Grant#close() close} once that work has completed
     * @throws InterruptedException if the calling thread is interrupted while blocked, which
     *                              aborts the operation
     */
    Grant acquire(long amount) throws InterruptedException;

    /**
     * A handle released by the consumer once the admitted work has completed. For a rate-limiter
     * realization (cost paid at {@link WorkLimiter#acquire}) {@link #close()} is a no-op; for a
     * semaphore-style in-flight-amount realization it releases the permits taken by {@code acquire}.
     */
    interface Grant extends AutoCloseable {
        /** Releases the grant. Never throws. */
        @Override
        void close();

        /** A grant that holds nothing and releases nothing. */
        Grant NOOP = () -> { };
    }

    /** A limiter that admits everything immediately. */
    WorkLimiter UNLIMITED = amount -> Grant.NOOP;
}
