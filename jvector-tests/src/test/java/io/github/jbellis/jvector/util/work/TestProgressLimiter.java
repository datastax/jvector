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

import io.github.jbellis.jvector.util.work.WorkLimiter.Grant;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class TestProgressLimiter {

    private static final WorkStage STAGE = () -> "TEST";

    private static long millisFor(ThrowingRunnable r) throws Exception {
        long t0 = System.nanoTime();
        r.run();
        return (System.nanoTime() - t0) / 1_000_000L;
    }

    private interface ThrowingRunnable { void run() throws Exception; }

    // ---- rateLimited (leaky bucket) ----

    @Test
    public void rateLimitedRejectsNonPositiveOrNonFiniteRate() {
        for (double bad : new double[]{0.0, -1.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY}) {
            try {
                ProgressLimiter.rateLimited(bad);
                fail("expected IllegalArgumentException for rate " + bad);
            } catch (IllegalArgumentException expected) {
                // ok
            }
        }
    }

    @Test
    public void rateLimitedAdmitsZeroOrNegativeAmountImmediately() throws Exception {
        ProgressLimiter limiter = ProgressLimiter.rateLimited(1.0); // 1 unit/sec: any real wait would be seconds
        long ms = millisFor(() -> {
            try (Grant g = limiter.acquire(0)) { assertNotNull(g); }
            try (Grant g = limiter.acquire(-100)) { assertNotNull(g); }
        });
        assertTrue("zero/negative amount must not block, waited " + ms + "ms", ms < 500);
    }

    @Test
    public void rateLimitedPacesSubsequentAcquire() throws Exception {
        ProgressLimiter limiter = ProgressLimiter.rateLimited(1000.0); // 1 unit/ms
        limiter.acquire(200).close(); // warmup: drained bucket admits the first request immediately

        long ms = millisFor(() -> limiter.acquire(200).close()); // must wait ~200ms behind the warmup reservation
        assertTrue("expected pacing >= ~100ms at 1000 units/s after a 200-unit warmup, got " + ms + "ms", ms >= 100);
    }

    @Test
    public void rateLimitedFirstAcquireIsNotDelayed() throws Exception {
        ProgressLimiter limiter = ProgressLimiter.rateLimited(10.0); // slow: a delayed first call would be seconds
        long ms = millisFor(() -> limiter.acquire(1000).close());
        assertTrue("first acquire on a drained bucket must not block, waited " + ms + "ms", ms < 500);
    }

    @Test
    public void rateLimitedIsInterruptible() throws Exception {
        ProgressLimiter limiter = ProgressLimiter.rateLimited(100.0); // 100 units/sec
        limiter.acquire(100).close(); // warmup reserves ~1s of future emission time

        AtomicReference<Throwable> caught = new AtomicReference<>();
        AtomicInteger returnedNormally = new AtomicInteger();
        Thread t = new Thread(() -> {
            try {
                limiter.acquire(1).close(); // blocks ~1s behind the warmup reservation
                returnedNormally.incrementAndGet();
            } catch (Throwable e) {
                caught.set(e);
            }
        }, "rate-limited-blocked");
        t.start();
        Thread.sleep(150); // let it reach the interruptible sleep
        t.interrupt();
        t.join(5_000);

        assertFalse("interrupted acquire should not hang", t.isAlive());
        assertEquals("acquire should not have returned normally", 0, returnedNormally.get());
        assertTrue("expected InterruptedException, got " + caught.get(),
                caught.get() instanceof InterruptedException);
    }

    @Test
    public void rateLimitedGrantIsNoopAndProgressIsNoop() throws Exception {
        ProgressLimiter limiter = ProgressLimiter.rateLimited(1_000_000.0);
        Grant g = limiter.acquire(10);
        assertNotNull(g);
        g.close();
        g.close(); // idempotent no-op
        limiter.onProgress(STAGE, 1, 2); // rate limiter does not track progress; must not throw
    }

    // ---- logging wrapper ----

    @Test(expected = NullPointerException.class)
    public void loggingRejectsNullSinkWithDelegate() {
        ProgressLimiter.logging(ProgressLimiter.UNLIMITED, null);
    }

    @Test(expected = NullPointerException.class)
    public void loggingRejectsNullSink() {
        ProgressLimiter.logging((java.util.function.Consumer<String>) null);
    }

    @Test
    public void loggingNullDelegateBehavesAsUnlimited() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        ProgressLimiter limiter = ProgressLimiter.logging(null, log::add);
        long ms = millisFor(() -> limiter.acquire(Long.MAX_VALUE).close()); // UNLIMITED: instant
        assertTrue("null delegate should not throttle, waited " + ms + "ms", ms < 500);
    }

    @Test
    public void loggingDelegatesBothFacets() {
        RecordingLimiter delegate = new RecordingLimiter();
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        ProgressLimiter limiter = ProgressLimiter.logging(delegate, log::add);

        limiter.onProgress(STAGE, 3, 10);
        assertEquals("onProgress must be delegated", 1, delegate.progressCalls.get());
        assertEquals(3, delegate.lastCompleted);
        assertEquals(10, delegate.lastTotal);
        assertTrue("onProgress should have been logged",
                log.stream().anyMatch(s -> s.contains("TEST") && s.contains("3/10")));
    }

    @Test
    public void loggingPreservesDelegateGrant() throws Exception {
        RecordingLimiter delegate = new RecordingLimiter();
        ProgressLimiter limiter = ProgressLimiter.logging(delegate, s -> { });

        Grant g = limiter.acquire(1234);
        assertEquals("acquire must be delegated", 1, delegate.acquireCalls.get());
        assertEquals(1234, delegate.lastAmount);
        assertEquals("grant must not be closed yet", 0, delegate.grantCloses.get());
        g.close();
        assertEquals("closing the wrapper grant must close the delegate's grant", 1, delegate.grantCloses.get());
    }

    @Test
    public void loggingLogsAcquireOnlyWhenItBlocks() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());

        // Instant delegate (UNLIMITED): no throttled line expected.
        ProgressLimiter fast = ProgressLimiter.logging(ProgressLimiter.UNLIMITED, log::add);
        fast.acquire(500).close();
        assertTrue("unblocked acquire should not log a throttle line",
                log.stream().noneMatch(s -> s.contains("throttled")));

        // Blocking delegate: a throttled line is expected.
        log.clear();
        ProgressLimiter slow = ProgressLimiter.logging(new SleepingLimiter(60), log::add);
        slow.acquire(500).close();
        assertTrue("blocked acquire should log a throttle line",
                log.stream().anyMatch(s -> s.contains("throttled") && s.contains("500")));
    }

    // ---- composition ----

    @Test
    public void loggingComposesWithRateLimited() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        ProgressLimiter limiter = ProgressLimiter.logging(ProgressLimiter.rateLimited(1000.0), log::add);

        limiter.acquire(200).close(); // warmup
        long ms = millisFor(() -> limiter.acquire(200).close());

        assertTrue("composed limiter should still pace, got " + ms + "ms", ms >= 100);
        assertTrue("composed limiter should log the throttled acquire",
                log.stream().anyMatch(s -> s.contains("throttled")));
        limiter.onProgress(STAGE, 5, 5);
        assertTrue("composed limiter should log progress",
                log.stream().anyMatch(s -> s.contains("TEST") && s.contains("5/5")));
    }

    // ---- melded SPI defaults ----

    @Test
    public void unlimitedIsFullyNoop() throws Exception {
        long ms = millisFor(() -> {
            try (Grant g = ProgressLimiter.UNLIMITED.acquire(Long.MAX_VALUE)) {
                assertNotNull(g);
            }
        });
        assertTrue("UNLIMITED.acquire must not block, waited " + ms + "ms", ms < 500);
        ProgressLimiter.UNLIMITED.onProgress(STAGE, 7, -1); // no-op, must not throw

        // Facet no-op constants exist and are safe.
        WorkLimiter.Grant.NOOP.close();
        ProgressTracker.NOOP.onProgress(STAGE, 1, 1);
        try (Grant g = WorkLimiter.UNLIMITED.acquire(99)) {
            assertNotNull(g);
        }
    }

    @Test
    public void facetsAreIndependentlyOverridable() throws Exception {
        // Tracker-only: overrides onProgress, inherits no-op acquire.
        AtomicInteger progressSeen = new AtomicInteger();
        ProgressLimiter trackerOnly = new ProgressLimiter() {
            @Override public void onProgress(WorkStage stage, long completed, long total) {
                progressSeen.incrementAndGet();
            }
        };
        try (Grant g = trackerOnly.acquire(1_000_000)) { // inherited no-op: must not block
            assertNotNull(g);
        }
        trackerOnly.onProgress(STAGE, 1, 1);
        assertEquals(1, progressSeen.get());

        // Throttle-only: overrides acquire, inherits no-op onProgress.
        AtomicInteger acquireSeen = new AtomicInteger();
        ProgressLimiter throttleOnly = new ProgressLimiter() {
            @Override public Grant acquire(long amount) {
                acquireSeen.incrementAndGet();
                return Grant.NOOP;
            }
        };
        throttleOnly.onProgress(STAGE, 1, 1); // inherited no-op: must not throw
        throttleOnly.acquire(5).close();
        assertEquals(1, acquireSeen.get());
    }

    // ---- test doubles ----

    /** Records both facets and hands out a grant whose close is counted. */
    private static final class RecordingLimiter implements ProgressLimiter {
        final AtomicInteger progressCalls = new AtomicInteger();
        final AtomicInteger acquireCalls = new AtomicInteger();
        final AtomicInteger grantCloses = new AtomicInteger();
        volatile long lastCompleted, lastTotal, lastAmount;

        @Override
        public void onProgress(WorkStage stage, long completed, long total) {
            progressCalls.incrementAndGet();
            lastCompleted = completed;
            lastTotal = total;
        }

        @Override
        public Grant acquire(long amount) {
            acquireCalls.incrementAndGet();
            lastAmount = amount;
            return grantCloses::incrementAndGet;
        }
    }

    /** A throttle that always blocks for a fixed number of milliseconds. */
    private static final class SleepingLimiter implements ProgressLimiter {
        private final long sleepMillis;

        SleepingLimiter(long sleepMillis) { this.sleepMillis = sleepMillis; }

        @Override
        public Grant acquire(long amount) throws InterruptedException {
            Thread.sleep(sleepMillis);
            return Grant.NOOP;
        }
    }
}
