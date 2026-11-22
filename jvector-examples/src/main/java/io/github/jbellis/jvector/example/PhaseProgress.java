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

package io.github.jbellis.jvector.example;

import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/**
 * Console progress for a long construction phase. A daemon thread wakes every
 * {@code jvector.bench.progressSeconds} (default 10) to print elapsed time and, when the phase
 * has a counter, items done and an estimate of the time left. It reads the counter only, so it
 * does not slow the phase. Set the property to 0 to print only the start and finish lines.
 */
final class PhaseProgress implements AutoCloseable {
    private static final long INTERVAL_SECONDS = Long.getLong("jvector.bench.progressSeconds", 10);

    private final String phase;
    private final long total;
    private final LongSupplier done;
    private final long start = System.nanoTime();
    private final ScheduledExecutorService ticker;
    private boolean reportedComplete;

    private PhaseProgress(String phase, long total, LongSupplier done) {
        this.phase = phase;
        this.total = total;
        this.done = done;
        System.out.println(phase + " started");
        if (INTERVAL_SECONDS <= 0) {
            ticker = null;
            return;
        }
        ticker = Executors.newSingleThreadScheduledExecutor(r -> {
            var thread = new Thread(r, "jvector-bench-progress");
            thread.setDaemon(true);
            return thread;
        });
        ticker.scheduleAtFixedRate(this::report, INTERVAL_SECONDS, INTERVAL_SECONDS, TimeUnit.SECONDS);
    }

    /** A phase that reports {@code done} of {@code total} items. */
    static PhaseProgress counting(String phase, long total, LongSupplier done) {
        return new PhaseProgress(phase, total, done);
    }

    /** A phase without a counter; reports elapsed time only. */
    static PhaseProgress timed(String phase) {
        return new PhaseProgress(phase, 0, null);
    }

    private double elapsedSeconds() {
        return (System.nanoTime() - start) / 1e9;
    }

    private synchronized void report() {
        double elapsed = elapsedSeconds();
        if (done == null || total <= 0) {
            System.out.printf(Locale.ROOT, "%s: %.0f s elapsed%n", phase, elapsed);
            return;
        }
        long count = Math.min(done.getAsLong(), total);
        if (count >= total) {
            if (!reportedComplete) System.out.printf(Locale.ROOT, "%s: all %,d done, finishing (%.0f s elapsed)%n", phase, total, elapsed);
            else System.out.printf(Locale.ROOT, "%s: finishing (%.0f s elapsed)%n", phase, elapsed);
            reportedComplete = true;
            return;
        }
        String left = count == 0 ? "" : String.format(Locale.ROOT, ", about %.0f s left", elapsed * (total - count) / count);
        System.out.printf(Locale.ROOT, "%s: %,d of %,d (%.0f%%), %.0f s elapsed%s%n",
                phase, count, total, 100.0 * count / total, elapsed, left);
    }

    @Override
    public void close() {
        if (ticker != null) ticker.shutdownNow();
        synchronized (this) {
            System.out.printf(Locale.ROOT, "%s finished in %.1f s%n", phase, elapsedSeconds());
        }
    }
}
