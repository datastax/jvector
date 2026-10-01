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

import java.util.Objects;
import java.util.function.Consumer;

/**
 * The {@link ProgressTracker tracker} and the {@link WorkLimiter throttle} melded into one control
 * surface. A long-running jvector operation accepts a single {@code ProgressLimiter} and uses both
 * facets; an embedder may override only the facet it needs — the other defaults to a no-op, so
 * {@link #UNLIMITED} behaves exactly as if no SPI were installed.
 *
 * <p>Both methods default to no-ops here. A consumer that wants only one facet can still accept a
 * lambda via the single-method parents ({@link ProgressTracker}, {@link WorkLimiter}); a consumer
 * that wants both accepts a {@code ProgressLimiter}.
 */
@Experimental
public interface ProgressLimiter extends ProgressTracker, WorkLimiter {

    @Override
    default PhaseScope startPhase(WorkStage stage) { return PhaseScope.NOOP; }

    @Override
    default Grant acquire(long amount) throws InterruptedException { return Grant.NOOP; }

    /** Observes nothing and limits nothing — behaviour identical to no SPI installed. */
    ProgressLimiter UNLIMITED = new ProgressLimiter() { };

    /**
     * A leaky-bucket rate meter realizing the throttle facet: {@link #acquire} paces the aggregate
     * admitted amount to {@code unitsPerSecond} (bytes/sec for the compaction consumer), blocking
     * the caller when the rate would be exceeded and draining during idle gaps. {@link #startPhase}
     * returns a no-op scope and the returned grant is a no-op (cost is paid at {@code acquire}).
     * Compose with {@link #logging(ProgressLimiter, Consumer)} to also log.
     *
     * @param unitsPerSecond the sustained admission rate; must be finite and {@code > 0}
     * @throws IllegalArgumentException if {@code unitsPerSecond} is not finite and positive
     */
    static ProgressLimiter rateLimited(double unitsPerSecond) {
        return new LeakyBucketLimiter(unitsPerSecond);
    }

    /**
     * Wraps {@code delegate}, emitting a one-line message to {@code sink} on each phase start,
     * each {@link PhaseScope#onProgress} report, each phase completion, and each {@link #acquire}
     * that actually blocked, then delegating both facets to {@code delegate}. Composes over any
     * limiter — e.g. {@code logging(rateLimited(bytesPerSecond), log::info)} logs a rate-limited
     * operation. The delegate's grant is returned unchanged, so a semaphore delegate still
     * releases on close.
     *
     * @param delegate the limiter to observe and delegate to; {@code null} means {@link #UNLIMITED}
     * @param sink     receives formatted log lines (e.g. {@code msg -> logger.info(msg)})
     */
    static ProgressLimiter logging(ProgressLimiter delegate, Consumer<String> sink) {
        Objects.requireNonNull(sink, "sink");
        final ProgressLimiter d = (delegate == null) ? UNLIMITED : delegate;
        return new ProgressLimiter() {
            @Override
            public PhaseScope startPhase(WorkStage stage) {
                sink.accept("phase[" + stage.name() + "] started");
                PhaseScope scope = d.startPhase(stage);
                return new PhaseScope() {
                    @Override
                    public void onProgress(long completed, long total) {
                        sink.accept("progress[" + stage.name() + "] " + completed + "/" + (total < 0 ? "?" : Long.toString(total)));
                        scope.onProgress(completed, total);
                    }

                    @Override
                    public void close() {
                        try {
                            scope.close();
                        } finally {
                            sink.accept("phase[" + stage.name() + "] completed");
                        }
                    }
                };
            }

            @Override
            public Grant acquire(long amount) throws InterruptedException {
                long startNanos = System.nanoTime();
                Grant g = d.acquire(amount);
                long waitedMs = (System.nanoTime() - startNanos) / 1_000_000L;
                if (waitedMs > 0) {
                    sink.accept("acquire " + amount + " units - throttled " + waitedMs + "ms");
                }
                return g;
            }
        };
    }

    /** Logging over no throttle: equivalent to {@code logging(UNLIMITED, sink)}. */
    static ProgressLimiter logging(Consumer<String> sink) {
        return logging(UNLIMITED, sink);
    }
}
