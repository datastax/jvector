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

package io.github.jbellis.jvector.fastervector;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link IvfAshBatchingSearcher} over a fake search, so the batching runs without the native
 * libraries. The fake answers each query with one result whose label is the query's first
 * component.
 */
class IvfAshBatchingSearcherTest {
    private static final int DIM = 2;

    private static SearchResult echo(Vecs queries) {
        List<List<NodeScore>> rows = new ArrayList<>();
        VecsCursor cursor = queries.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) queries.dim() * Float.BYTES);
            while (cursor.nextVectorInto(buf)) {
                rows.add(List.of(new NodeScore((long) buf.get(ValueLayout.JAVA_FLOAT, 0), 1f)));
            }
        }
        return new SearchResult(rows);
    }

    /** A fake search whose first call blocks until {@link #release()}, so queries queue behind it. */
    private static final class Gate implements Function<Vecs, SearchResult> {
        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch released = new CountDownLatch(1);
        final List<Integer> batchSizes = new ArrayList<>();

        @Override
        public synchronized SearchResult apply(Vecs queries) {
            batchSizes.add((int) queries.nvecs());
            entered.countDown();
            boolean interrupted = false;
            while (released.getCount() > 0) {
                try {
                    wait(10);
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
            return echo(queries);
        }

        void awaitEntered() throws InterruptedException {
            assertTrue(entered.await(10, TimeUnit.SECONDS));
        }

        void release() {
            released.countDown();
        }
    }

    private static float[] query(int id) {
        return new float[]{id, 0};
    }

    @Test
    void eachCallerGetsItsOwnResult() throws Exception {
        try (var batcher = new IvfAshBatchingSearcher(IvfAshBatchingSearcherTest::echo, DIM, 64, Duration.ZERO)) {
            ExecutorService pool = Executors.newFixedThreadPool(16);
            try {
                List<Future<?>> callers = new ArrayList<>();
                for (int t = 0; t < 16; t++) {
                    int base = t * 1000;
                    callers.add(pool.submit(() -> {
                        for (int i = 0; i < 200; i++) {
                            assertEquals(base + i, batcher.search(query(base + i)).get(0).label());
                        }
                    }));
                }
                for (Future<?> caller : callers) {
                    caller.get(30, TimeUnit.SECONDS);
                }
            } finally {
                pool.shutdownNow();
            }
            assertEquals(16 * 200, batcher.queryCount());
            assertTrue(batcher.batchCount() <= batcher.queryCount());
        }
    }

    @Test
    void queriesThatArriveDuringASearchFormTheNextBatch() throws Exception {
        Gate gate = new Gate();
        try (var batcher = new IvfAshBatchingSearcher(gate, DIM, 64, Duration.ZERO)) {
            var first = batcher.searchAsync(query(0));
            gate.awaitEntered();
            List<CompletableFuture<List<NodeScore>>> rest = new ArrayList<>();
            for (int i = 1; i <= 10; i++) {
                rest.add(batcher.searchAsync(query(i)));
            }
            gate.release();
            assertEquals(0, first.get(10, TimeUnit.SECONDS).get(0).label());
            for (int i = 0; i < rest.size(); i++) {
                assertEquals(i + 1, rest.get(i).get(10, TimeUnit.SECONDS).get(0).label());
            }
            assertEquals(List.of(1, 10), gate.batchSizes);
            assertEquals(2, batcher.batchCount());
            assertEquals(11, batcher.queryCount());
        }
    }

    @Test
    void batchesAreCappedAtMaxBatch() throws Exception {
        Gate gate = new Gate();
        try (var batcher = new IvfAshBatchingSearcher(gate, DIM, 4, Duration.ZERO)) {
            List<CompletableFuture<List<NodeScore>>> all = new ArrayList<>();
            all.add(batcher.searchAsync(query(0)));
            gate.awaitEntered();
            for (int i = 1; i <= 10; i++) {
                all.add(batcher.searchAsync(query(i)));
            }
            gate.release();
            CompletableFuture.allOf(all.toArray(CompletableFuture[]::new)).get(10, TimeUnit.SECONDS);
            assertEquals(List.of(1, 4, 4, 2), gate.batchSizes);
        }
    }

    @Test
    void maxDelayWaitsForMoreQueries() throws Exception {
        List<Integer> sizes = new ArrayList<>();
        Function<Vecs, SearchResult> search = queries -> {
            synchronized (sizes) {
                sizes.add((int) queries.nvecs());
            }
            return echo(queries);
        };
        try (var batcher = new IvfAshBatchingSearcher(search, DIM, 2, Duration.ofSeconds(10))) {
            var a = batcher.searchAsync(query(1));
            Thread.sleep(50);
            var b = batcher.searchAsync(query(2));
            assertEquals(1, a.get(10, TimeUnit.SECONDS).get(0).label());
            assertEquals(2, b.get(10, TimeUnit.SECONDS).get(0).label());
            assertEquals(List.of(2), sizes, "the first query waits for the second, up to the full batch");
        }
    }

    @Test
    void wrongDimIsRejectedWhenTheDimIsKnown() {
        try (var batcher = new IvfAshBatchingSearcher(IvfAshBatchingSearcherTest::echo, DIM, 64, Duration.ZERO)) {
            assertThrows(IllegalArgumentException.class, () -> batcher.search(new float[]{1, 2, 3}));
        }
    }

    @Test
    void withoutAKnownDimOnlyTheOddQueryFails() throws Exception {
        Gate gate = new Gate();
        try (var batcher = new IvfAshBatchingSearcher(gate, 0, 64, Duration.ZERO)) {
            var first = batcher.searchAsync(query(0));
            gate.awaitEntered();
            var good = batcher.searchAsync(query(1));
            var odd = batcher.searchAsync(new float[]{2, 0, 0});
            gate.release();
            first.get(10, TimeUnit.SECONDS);
            assertEquals(1, good.get(10, TimeUnit.SECONDS).get(0).label());
            var e = assertThrows(CompletionException.class, odd::join);
            assertInstanceOf(IllegalArgumentException.class, e.getCause());
        }
    }

    @Test
    void searchFailuresReachEveryCallerInTheBatch() {
        Function<Vecs, SearchResult> failing = queries -> {
            throw new FvecException("boom");
        };
        try (var batcher = new IvfAshBatchingSearcher(failing, DIM, 64, Duration.ZERO)) {
            assertThrows(FvecException.class, () -> batcher.search(query(1)));
            // The dispatcher survives a failed batch
            assertThrows(FvecException.class, () -> batcher.search(query(2)));
        }
    }

    @Test
    void closeFinishesTheCurrentBatchAndFailsTheRest() throws Exception {
        Gate gate = new Gate();
        var batcher = new IvfAshBatchingSearcher(gate, DIM, 64, Duration.ZERO);
        var inFlight = batcher.searchAsync(query(1));
        gate.awaitEntered();
        var queued = batcher.searchAsync(query(2));
        Thread closer = new Thread(batcher::close);
        closer.start();
        Thread.sleep(50);
        gate.release();
        closer.join(10_000);
        assertFalse(closer.isAlive());

        assertEquals(1, inFlight.get(10, TimeUnit.SECONDS).get(0).label());
        var e = assertThrows(CompletionException.class, queued::join);
        assertInstanceOf(IllegalStateException.class, e.getCause());
        assertThrows(IllegalStateException.class, () -> batcher.searchAsync(query(3)));
        batcher.close(); // idempotent
    }

    @Test
    void invalidSettingsAreRejected() {
        assertThrows(IllegalArgumentException.class,
                     () -> new IvfAshBatchingSearcher(IvfAshBatchingSearcherTest::echo, DIM, 0, Duration.ZERO));
        assertThrows(IllegalArgumentException.class,
                     () -> new IvfAshBatchingSearcher(IvfAshBatchingSearcherTest::echo, DIM, 1, Duration.ofMillis(-1)));
    }
}
