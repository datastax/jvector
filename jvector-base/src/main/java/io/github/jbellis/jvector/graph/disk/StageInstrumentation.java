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

import io.github.jbellis.jvector.util.work.ProgressLimiter;
import io.github.jbellis.jvector.util.work.WorkStage;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Uniform stage instrumentation for one compaction run, wrapped around the embedder's
 * {@link ProgressLimiter} at the one point every phase passes through.
 *
 * <p>Every stage of a compaction — the compactor's own and the strategies' — opens its phase
 * through {@code progress.startPhase}, reports through the returned scope, and admits output
 * through {@code progress.acquire}. Decorating that object therefore instruments every stage the
 * same way without a log line at each site:
 * <ul>
 *   <li>a line when a stage starts, one when its total becomes known, one at each 10% of
 *       progress, and one on completion with the units done and the elapsed time;</li>
 *   <li>per-stage elapsed time, accumulated across repeated phases of the same stage (the upper
 *       layers open one phase per level), for the end-of-run {@link #summary()};</li>
 *   <li>time spent blocked in {@link #acquire}, so a run paced by the host's throttle shows how
 *       much of its wall-clock was the throttle rather than the work.</li>
 * </ul>
 *
 * <p>The delegate sees exactly the calls it would have seen: one {@code startPhase} per phase,
 * every progress report in order, one {@code close}, and every {@code acquire} with its grant
 * returned unchanged. Progress reports are serialized per scope, which the compactor's callers
 * already do by locking the scope; locking here as well keeps the counters coherent for callers
 * that do not.
 */
final class StageInstrumentation implements ProgressLimiter {
    private final ProgressLimiter delegate;
    private final Consumer<String> sink;
    // stage name -> {nanos, phases, completed, total}; insertion order is first-start order
    private final Map<String, long[]> stages = new LinkedHashMap<>();
    private final AtomicLong throttleNanos = new AtomicLong();
    private final AtomicLong throttleBlocked = new AtomicLong();
    private long runStartNanos = System.nanoTime();

    StageInstrumentation(ProgressLimiter delegate, Consumer<String> sink) {
        this.delegate = delegate == null ? ProgressLimiter.UNLIMITED : delegate;
        this.sink = Objects.requireNonNull(sink, "sink");
    }

    /** The limiter this instruments. */
    ProgressLimiter delegate() {
        return delegate;
    }

    /** Resets the run clock and the per-stage totals; call at the start of a compaction. */
    synchronized void beginRun() {
        stages.clear();
        throttleNanos.set(0);
        throttleBlocked.set(0);
        runStartNanos = System.nanoTime();
    }

    @Override
    public PhaseScope startPhase(WorkStage stage) {
        final String name = stage.name();
        final long t0 = System.nanoTime();
        sink.accept("Stage " + name + " started");
        final PhaseScope inner = delegate.startPhase(stage);
        return new PhaseScope() {
            private long completed = -1;
            private long total = -1;
            private int bucket = 0;
            private boolean closed;

            @Override
            public synchronized void onProgress(long c, long t) {
                if (total <= 0 && t > 0) {
                    sink.accept("Stage " + name + ": " + t + " units");
                }
                completed = c;
                total = t;
                if (t > 0) {
                    int b = (int) Math.max(0, Math.min(10, c * 10 / t));
                    if (b > bucket) {
                        bucket = b;
                        if (b < 10) {
                            sink.accept("Stage " + name + " progress: " + c + "/" + t + " (" + (b * 10) + "%)");
                        }
                    }
                }
                inner.onProgress(c, t);
            }

            @Override
            public synchronized void close() {
                if (closed) {
                    return; // the contract is exactly once; a second close must not double-count
                }
                closed = true;
                try {
                    inner.close();
                } finally {
                    long nanos = System.nanoTime() - t0;
                    record(name, nanos, completed, total);
                    String units = total > 0 ? completed + "/" + total + " units in "
                                 : completed >= 0 ? completed + " units in " : "";
                    sink.accept("Stage " + name + " completed: " + units + (nanos / 1_000_000L) + " ms");
                }
            }
        };
    }

    @Override
    public Grant acquire(long amount) throws InterruptedException {
        long t0 = System.nanoTime();
        Grant g = delegate.acquire(amount);
        long waited = System.nanoTime() - t0;
        throttleNanos.addAndGet(waited);
        if (waited >= 1_000_000L) {
            throttleBlocked.incrementAndGet();
        }
        return g;
    }

    private synchronized void record(String name, long nanos, long completed, long total) {
        long[] v = stages.computeIfAbsent(name, k -> new long[4]);
        v[0] += nanos;
        v[1] += 1;
        v[2] = completed;
        v[3] = total;
    }

    /** Elapsed milliseconds per stage, in first-start order. */
    synchronized Map<String, Long> stageMillis() {
        Map<String, Long> out = new LinkedHashMap<>();
        for (Map.Entry<String, long[]> e : stages.entrySet()) {
            out.put(e.getKey(), e.getValue()[0] / 1_000_000L);
        }
        return out;
    }

    /** Milliseconds spent inside {@link #acquire}. */
    long throttleMillis() {
        return throttleNanos.get() / 1_000_000L;
    }

    /**
     * One line: every stage's elapsed time in first-start order, the run's wall-clock, and the
     * throttle wait. This is the "where the time goes" table of a merge, emitted by the merge.
     */
    synchronized String summary() {
        StringBuilder sb = new StringBuilder("Compaction stage times:");
        long inStages = 0;
        for (Map.Entry<String, long[]> e : stages.entrySet()) {
            long[] v = e.getValue();
            sb.append(' ').append(e.getKey()).append('=').append(v[0] / 1_000_000L).append("ms");
            if (v[1] > 1) {
                sb.append("(x").append(v[1]).append(')');
            }
            inStages += v[0];
        }
        long wall = System.nanoTime() - runStartNanos;
        sb.append(" | wall ").append(wall / 1_000_000L).append(" ms, ")
          .append(inStages / 1_000_000L).append(" ms in stages, throttle wait ")
          .append(throttleMillis()).append(" ms (").append(throttleBlocked.get()).append(" blocked admissions)");
        return sb.toString();
    }
}
