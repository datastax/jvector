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

import java.util.concurrent.TimeUnit;

/**
 * A leaky-bucket rate meter realizing the {@link WorkLimiter} facet: {@link #acquire} paces the
 * aggregate admitted amount to a fixed {@code unitsPerSecond}, blocking the caller when the rate
 * would be exceeded. The bucket drains during idle gaps (a burst after a quiet period is not
 * charged for the idle time), and the first request after an idle period is admitted without
 * delay — the cost of each request is paid by the <i>next</i> one, which is the standard smooth
 * shaping behaviour. {@link #onProgress} is inherited as a no-op: this limiter only throttles.
 *
 * <p>Thread-safe and reentrant: the emission clock is advanced under a short lock, then the caller
 * sleeps <i>outside</i> the lock, so concurrent callers serialize their reservations but wait
 * independently. The returned grant is a no-op — the cost is paid entirely at {@code acquire}.
 *
 * <p>Obtain instances via {@link ProgressLimiter#rateLimited(double)}.
 */
final class LeakyBucketLimiter implements ProgressLimiter {
    private final double nanosPerUnit;
    private final Object lock = new Object();
    // Earliest nanoTime at which the next reservation may start. Long.MIN_VALUE until the first
    // acquire, so Math.max(now, nextFreeNanos) == now (a fully drained bucket) on the first call.
    private long nextFreeNanos = Long.MIN_VALUE;

    LeakyBucketLimiter(double unitsPerSecond) {
        if (!(unitsPerSecond > 0) || Double.isInfinite(unitsPerSecond)) {
            throw new IllegalArgumentException("unitsPerSecond must be finite and > 0, got " + unitsPerSecond);
        }
        this.nanosPerUnit = 1_000_000_000.0 / unitsPerSecond;
    }

    @Override
    public Grant acquire(long amount) throws InterruptedException {
        if (amount <= 0) {
            return Grant.NOOP;
        }
        long startAt;
        synchronized (lock) {
            long now = System.nanoTime();
            startAt = Math.max(now, nextFreeNanos);              // drain if idle, else queue behind backlog
            long cost = (long) Math.min((double) Long.MAX_VALUE, amount * nanosPerUnit);
            nextFreeNanos = startAt + cost;
        }
        // Sleep (interruptibly, so cancellation aborts) until this request's slot opens.
        for (long remaining = startAt - System.nanoTime(); remaining > 0; remaining = startAt - System.nanoTime()) {
            TimeUnit.NANOSECONDS.sleep(remaining);
        }
        return Grant.NOOP;
    }
}
