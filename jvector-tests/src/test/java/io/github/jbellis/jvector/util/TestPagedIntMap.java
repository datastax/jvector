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
package io.github.jbellis.jvector.util;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** PagedIntMap-specific behavior: page boundaries, growth, and no lost updates while growing. */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPagedIntMap extends RandomizedTest {

    @Test
    public void testPageBoundariesAndSparseGrowth() {
        var map = new PagedIntMap<String>(1);
        int page = PagedIntMap.PAGE_SIZE;
        int[] keys = {0, page - 1, page, page + 1, 3 * page - 1, 10 * page, 1_000_003};
        for (int k : keys) {
            Assert.assertNull(map.get(k));
            Assert.assertTrue(map.compareAndPut(k, null, "v" + k));
        }
        Assert.assertEquals(keys.length, map.size());
        for (int k : keys) Assert.assertEquals("v" + k, map.get(k));
        // Gaps read as null, including inside and beyond installed pages.
        Assert.assertNull(map.get(page + 2));
        Assert.assertNull(map.get(5 * page));
        Assert.assertNull(map.get(Integer.MAX_VALUE));
        Assert.assertNull(map.remove(Integer.MAX_VALUE));
    }

    @Test
    public void testLargestKey() {
        var map = new PagedIntMap<String>(0);
        Assert.assertTrue(map.compareAndPut(Integer.MAX_VALUE, null, "max"));
        Assert.assertEquals("max", map.get(Integer.MAX_VALUE));
        Assert.assertEquals(1, map.size());
    }

    @Test
    public void testArgumentChecks() {
        var map = new PagedIntMap<String>(16);
        Assert.assertThrows(IllegalArgumentException.class, () -> map.compareAndPut(1, null, null));
        Assert.assertThrows(IndexOutOfBoundsException.class, () -> map.compareAndPut(-1, null, "x"));
        Assert.assertThrows(IndexOutOfBoundsException.class, () -> map.get(-1));
        Assert.assertThrows(IllegalArgumentException.class, () -> new PagedIntMap<String>(-1));
    }

    @Test
    public void testCompareAndPutSemantics() {
        var map = new PagedIntMap<String>(4);
        Assert.assertTrue(map.compareAndPut(7, null, "a"));
        Assert.assertFalse("insert over existing must fail", map.compareAndPut(7, null, "b"));
        Assert.assertFalse("stale expected value must fail", map.compareAndPut(7, "zzz", "b"));
        Assert.assertTrue(map.compareAndPut(7, "a", "b"));
        Assert.assertEquals(1, map.size());
        Assert.assertEquals("b", map.remove(7));
        Assert.assertNull(map.remove(7));
        Assert.assertEquals(0, map.size());
    }

    @Test
    public void testForEachInKeyOrder() {
        var map = new PagedIntMap<Integer>(1);
        int[] keys = {5, 4096, 4097, 9000, 70_000};
        for (int i = keys.length - 1; i >= 0; i--) map.compareAndPut(keys[i], null, keys[i]);
        List<Integer> seen = new ArrayList<>();
        map.forEach((k, v) -> { Assert.assertEquals(Integer.valueOf(k), v); seen.add(k); });
        Assert.assertEquals(List.of(5, 4096, 4097, 9000, 70_000), seen);
    }

    /**
     * The race DenseIntMap's lock prevents: a CAS landing while the storage grows. Threads keep
     * incrementing counters on low keys with CAS loops while other threads force growth by
     * inserting ever-higher keys. Any lost update shows up as a counter below its expected total.
     */
    @Test
    public void testNoLostUpdatesWhileGrowing() throws Exception {
        for (int round = 0; round < 5; round++) {
            var map = new PagedIntMap<Integer>(1);
            int counters = 64, incrementers = 8, incrementsEach = 20_000, growers = 4, growKeys = 300_000;
            for (int k = 0; k < counters; k++) map.compareAndPut(k, null, 0);
            var start = new CountDownLatch(1);
            var failure = new AtomicReference<Throwable>();
            List<Thread> threads = new ArrayList<>();
            for (int t = 0; t < incrementers; t++) {
                final int seed = t;
                threads.add(new Thread(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < incrementsEach; i++) {
                            int key = (seed * 31 + i) % counters;
                            while (true) {
                                Integer old = map.get(key);
                                if (map.compareAndPut(key, old, old + 1)) break;
                            }
                        }
                    } catch (Throwable e) { failure.compareAndSet(null, e); }
                }));
            }
            for (int t = 0; t < growers; t++) {
                final int offset = t;
                threads.add(new Thread(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < growKeys; i++) {
                            int key = counters + i * growers + offset;
                            Assert.assertTrue(map.compareAndPut(key, null, -key));
                        }
                    } catch (Throwable e) { failure.compareAndSet(null, e); }
                }));
            }
            threads.forEach(Thread::start);
            start.countDown();
            for (Thread t : threads) t.join();
            if (failure.get() != null) throw new AssertionError(failure.get());

            long total = 0;
            for (int k = 0; k < counters; k++) total += map.get(k);
            Assert.assertEquals((long) incrementers * incrementsEach, total);
            Assert.assertEquals(counters + growers * growKeys, map.size());
            for (int i = 0; i < growers * growKeys; i++) {
                int key = counters + i;
                Assert.assertEquals(Integer.valueOf(-key), map.get(key));
            }
        }
    }

    /** Readers running during growth never see a value disappear once observed. */
    @Test
    public void testReadersDuringGrowth() throws Exception {
        var map = new PagedIntMap<Integer>(1);
        int keys = 400_000;
        var done = new AtomicBoolean();
        var failure = new AtomicReference<Throwable>();
        var writer = new Thread(() -> {
            for (int k = 0; k < keys; k++) map.compareAndPut(k, null, k);
            done.set(true);
        });
        List<Thread> readers = new ArrayList<>();
        for (int r = 0; r < 4; r++) {
            readers.add(new Thread(() -> {
                try {
                    int highestSeen = -1;
                    while (!done.get()) {
                        // Values are inserted in key order, so everything at or below a seen key must be present.
                        int probe = Math.max(0, highestSeen);
                        Integer v = map.get(probe);
                        if (v != null) {
                            Assert.assertEquals(Integer.valueOf(probe), v);
                            highestSeen = probe + 1;
                        }
                        if (highestSeen > 0) Assert.assertNotNull(map.get(highestSeen - 1));
                    }
                } catch (Throwable e) { failure.compareAndSet(null, e); }
            }));
        }
        readers.forEach(Thread::start);
        writer.start();
        writer.join();
        for (Thread t : readers) t.join();
        if (failure.get() != null) throw new AssertionError(failure.get());
        Assert.assertEquals(keys, map.size());
    }
}
