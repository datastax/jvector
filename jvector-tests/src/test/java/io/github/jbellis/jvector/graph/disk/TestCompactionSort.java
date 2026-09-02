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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.ParallelExecutor;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CompactionSort} replaces {@code Arrays.parallelSort}, whose decomposition always lands on
 * the common pool and so escapes the embedder's thread budget. These verify the replacement sorts
 * identically to the JDK across the shapes the merge tree has to handle, and that it does its work
 * only on the executor it was handed.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestCompactionSort extends RandomizedTest {

    /** Records which threads ran work, so "stayed on the given executor" is checkable. */
    private static class WatchedExecutor implements ParallelExecutor {
        private final ParallelExecutor delegate;
        final Set<Thread> threads = ConcurrentHashMap.newKeySet();
        volatile int rounds;

        WatchedExecutor(ParallelExecutor delegate) {
            this.delegate = delegate;
        }

        @Override
        public void forEachInt(int upperBound, IntConsumer body) {
            rounds++;
            delegate.forEachInt(upperBound, i -> {
                threads.add(Thread.currentThread());
                body.accept(i);
            });
        }

        @Override
        public void forEach(IntStream source, IntConsumer body) {
            rounds++;
            delegate.forEach(source, i -> {
                threads.add(Thread.currentThread());
                body.accept(i);
            });
        }

        @Override
        public <T> void forEach(Stream<T> source, Consumer<T> body) {
            rounds++;
            delegate.forEach(source, t -> {
                threads.add(Thread.currentThread());
                body.accept(t);
            });
        }
    }

    private static long[] randomLongs(Random rnd, int n) {
        long[] a = new long[n];
        for (int i = 0; i < n; i++) {
            // Full 64-bit spread, so the negative half of the signed order is exercised: callers
            // pack a key into the high 32 bits and a key with bit 31 set sorts as negative.
            a[i] = rnd.nextLong();
        }
        return a;
    }

    /**
     * Matches {@code Arrays.sort} across lengths that straddle the chunking floor and produce
     * both even and odd numbers of merge runs, at several stated parallelisms.
     */
    @Test
    public void testMatchesJdkSortAcrossShapes() {
        Random rnd = new Random(20260821L);
        int[] lengths = {0, 1, 2, 7, 1000, CompactionSort.MIN_CHUNK - 1, CompactionSort.MIN_CHUNK,
                         2 * CompactionSort.MIN_CHUNK, 3 * CompactionSort.MIN_CHUNK + 17,
                         5 * CompactionSort.MIN_CHUNK, 7 * CompactionSort.MIN_CHUNK - 3,
                         64 * CompactionSort.MIN_CHUNK + 1};
        int[] parallelisms = {1, 2, 3, 4, 5, 8, 13};

        for (int length : lengths) {
            for (int parallelism : parallelisms) {
                long[] a = randomLongs(rnd, length);
                long[] expected = a.clone();
                Arrays.sort(expected);
                CompactionSort.sort(a, length, ParallelExecutor.callerRuns(), parallelism);
                assertArrayEquals("length=" + length + " parallelism=" + parallelism, expected, a);
            }
        }
    }

    /** Only the leading {@code length} elements are touched; the tail must be left alone. */
    @Test
    public void testSortsOnlyThePrefix() {
        Random rnd = new Random(7L);
        int length = 4 * CompactionSort.MIN_CHUNK + 11;
        int tail = 257;
        long[] a = randomLongs(rnd, length + tail);
        long[] expectedHead = Arrays.copyOf(a, length);
        Arrays.sort(expectedHead);
        long[] expectedTail = Arrays.copyOfRange(a, length, a.length);

        CompactionSort.sort(a, length, ParallelExecutor.callerRuns(), 8);

        assertArrayEquals(expectedHead, Arrays.copyOf(a, length));
        assertArrayEquals("elements past `length` must be untouched",
                expectedTail, Arrays.copyOfRange(a, length, a.length));
    }

    /** Duplicates and already-sorted / reversed input are the degenerate merge cases. */
    @Test
    public void testDegenerateInputs() {
        int length = 6 * CompactionSort.MIN_CHUNK;

        long[] allSame = new long[length];
        Arrays.fill(allSame, 42L);
        CompactionSort.sort(allSame, length, ParallelExecutor.callerRuns(), 8);
        for (long v : allSame) {
            assertEquals(42L, v);
        }

        long[] ascending = new long[length];
        for (int i = 0; i < length; i++) {
            ascending[i] = i;
        }
        long[] expected = ascending.clone();
        CompactionSort.sort(ascending, length, ParallelExecutor.callerRuns(), 8);
        assertArrayEquals(expected, ascending);

        long[] descending = new long[length];
        for (int i = 0; i < length; i++) {
            descending[i] = length - i;
        }
        long[] expectedDesc = descending.clone();
        Arrays.sort(expectedDesc);
        CompactionSort.sort(descending, length, ParallelExecutor.callerRuns(), 8);
        assertArrayEquals(expectedDesc, descending);
    }

    /**
     * The whole point of the class: work runs on the executor it was given, and on nothing else.
     * With callerRuns that means it never leaves the calling thread.
     */
    @Test
    public void testStaysOnCallerRunsThread() {
        long[] a = randomLongs(new Random(3L), 32 * CompactionSort.MIN_CHUNK);
        WatchedExecutor watched = new WatchedExecutor(ParallelExecutor.callerRuns());

        CompactionSort.sort(a, a.length, watched, 16);

        assertEquals("callerRuns must not leave the calling thread",
                Set.of(Thread.currentThread()), watched.threads);
        assertTrue("a multi-chunk sort should use more than one executor round", watched.rounds > 1);
    }

    /** With a real bounded executor, the work must land on that executor's threads only. */
    @Test
    public void testUsesOnlyTheSuppliedExecutorThreads() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(3, r -> {
            Thread t = new Thread(r, "compaction-sort-test");
            t.setDaemon(true);
            return t;
        });
        try {
            long[] a = randomLongs(new Random(11L), 40 * CompactionSort.MIN_CHUNK);
            long[] expected = a.clone();
            Arrays.sort(expected);

            WatchedExecutor watched = new WatchedExecutor(ParallelExecutor.over(pool, 3));
            CompactionSort.sort(a, a.length, watched, 3);

            assertArrayEquals(expected, a);
            for (Thread t : watched.threads) {
                assertTrue("work escaped onto " + t.getName(),
                        t.getName().equals("compaction-sort-test") || t == Thread.currentThread());
            }
        } finally {
            pool.shutdown();
            pool.awaitTermination(30, TimeUnit.SECONDS);
        }
    }
}
