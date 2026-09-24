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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static io.github.jbellis.jvector.example.util.MappedFvecsRandomAccessVectorValuesTest.assertVectorEquals;
import static io.github.jbellis.jvector.example.util.MappedFvecsRandomAccessVectorValuesTest.randomVectors;
import static io.github.jbellis.jvector.example.util.MappedFvecsRandomAccessVectorValuesTest.writeFvecs;
import static org.junit.jupiter.api.Assertions.*;

/// Tests for {@link LruGrainRandomAccessVectorValues}: correctness across grain and slab boundaries,
/// eviction bookkeeping, concurrent access, and recovery from a failed grain load.
public class LruGrainRandomAccessVectorValuesTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private MappedFvecsRandomAccessVectorValues mapped(float[][] vectors) throws IOException {
        Path file = tempFolder.newFile("base.fvecs").toPath();
        writeFvecs(file, vectors, null);
        int stride = Integer.BYTES + vectors[0].length * Float.BYTES;
        return new MappedFvecsRandomAccessVectorValues(file, 5L * stride); // slabs of 5, deliberately misaligned with grains
    }

    @Test
    public void servesCorrectVectorsWithinTheGrainBound() throws IOException {
        float[][] expected = randomVectors(100, 3, 1);
        var cache = new LruGrainRandomAccessVectorValues(mapped(expected), 7, 3);
        assertEquals(15, cache.grainCount());
        assertEquals(100, cache.size());
        assertEquals(3, cache.dimension());
        assertFalse(cache.isValueShared());
        assertSame(cache, cache.copy());

        for (int i = 0; i < 100; i++) {
            assertVectorEquals(expected[i], cache.getVector(i), 0);
        }
        assertEquals(15, cache.misses());
        assertEquals(100 - 15, cache.hits());
        assertEquals(12, cache.evictions());
        assertTrue(cache.residentGrains() <= 3, cache.stats());

        // reading backwards re-faults grains the forward pass evicted
        for (int i = 99; i >= 0; i--) {
            assertVectorEquals(expected[i], cache.getVector(i), 0);
        }
        assertTrue(cache.misses() > 15, cache.stats());
        assertTrue(cache.residentGrains() <= 3, cache.stats());
        assertThrows(IndexOutOfBoundsException.class, () -> cache.getVector(100));
    }

    @Test
    public void handedOutVectorsSurviveEviction() throws IOException {
        float[][] expected = randomVectors(40, 4, 2);
        var cache = new LruGrainRandomAccessVectorValues(mapped(expected), 4, 2);
        VectorFloat<?> first = cache.getVector(0);
        for (int i = 0; i < 40; i++) {
            cache.getVector(i);
        }
        assertTrue(cache.evictions() > 0);
        assertVectorEquals(expected[0], first, 0);
        // a re-faulted grain yields fresh objects, not the evicted ones
        assertNotSame(first, cache.getVector(0));
        assertVectorEquals(expected[0], cache.getVector(0), 0);
    }

    @Test
    public void rangeViewsAndGetVectorIntoWork() throws IOException {
        float[][] expected = randomVectors(30, 2, 3);
        var cache = new LruGrainRandomAccessVectorValues(mapped(expected), 8, 2);
        RandomAccessVectorValues view = cache.range(10, 25);
        assertFalse(view.isValueShared());
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int i = 0; i < view.size(); i++) {
            assertVectorEquals(expected[10 + i], view.getVector(i), 0);
            var dest = vts.createFloatVector(4);
            view.getVectorInto(i, dest, 2);
            assertVectorEquals(expected[10 + i], dest, 2);
        }
    }

    @Test
    public void wholeDatasetFitsWhenCapacityAllows() throws IOException {
        float[][] expected = randomVectors(50, 2, 4);
        var cache = new LruGrainRandomAccessVectorValues(mapped(expected), 10, 100);
        for (int pass = 0; pass < 3; pass++) {
            for (int i = 0; i < 50; i++) {
                assertVectorEquals(expected[i], cache.getVector(i), 0);
            }
        }
        assertEquals(5, cache.misses());
        assertEquals(0, cache.evictions());
        assertEquals(5, cache.residentGrains());
    }

    @Test
    public void concurrentReadersSeeConsistentValues() throws Exception {
        float[][] expected = randomVectors(5000, 8, 5);
        var cache = new LruGrainRandomAccessVectorValues(mapped(expected), 64, 6);
        int threads = 16;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                long seed = t;
                futures.add(pool.submit(() -> {
                    var random = new Random(seed);
                    for (int k = 0; k < 20_000; k++) {
                        int i = random.nextInt(expected.length);
                        assertVectorEquals(expected[i], cache.getVector(i), 0);
                    }
                }));
            }
            for (Future<?> f : futures) {
                f.get(2, TimeUnit.MINUTES);
            }
        } finally {
            pool.shutdownNow();
        }
        assertTrue(cache.evictions() > 0, cache.stats());
        assertTrue(cache.residentGrains() <= 6 + threads, cache.stats());
        // every read is counted exactly once, as a hit or as the load that satisfied it
        assertEquals(threads * 20_000L, cache.hits() + cache.misses(), cache.stats());
    }

    @Test
    public void failedLoadIsRetriedByTheNextReader() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> vectors = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            vectors.add(vts.createFloatVector(new float[] {i, i}));
        }
        var backing = new ListRandomAccessVectorValues(vectors, 2);
        var failOnce = new AtomicBoolean(true);
        RandomAccessVectorValues flaky = new RandomAccessVectorValues() {
            @Override public int size() { return backing.size(); }
            @Override public int dimension() { return backing.dimension(); }
            @Override public VectorFloat<?> getVector(int nodeId) {
                if (nodeId >= 5 && failOnce.compareAndSet(true, false)) {
                    throw new IllegalStateException("transient read failure");
                }
                return backing.getVector(nodeId);
            }
            @Override public boolean isValueShared() { return false; }
            @Override public RandomAccessVectorValues copy() { return this; }
        };

        var cache = new LruGrainRandomAccessVectorValues(flaky, 5, 2);
        assertEquals(1f, cache.getVector(1).get(0), 0f);
        assertThrows(IllegalStateException.class, () -> cache.getVector(7));
        assertEquals(1, cache.residentGrains());
        assertEquals(7f, cache.getVector(7).get(0), 0f);
        assertEquals(2, cache.residentGrains());
        assertEquals(2, cache.misses());
    }

    @Test
    public void rejectsBadParameters() throws IOException {
        var origin = mapped(randomVectors(3, 2, 6));
        assertThrows(IllegalArgumentException.class, () -> new LruGrainRandomAccessVectorValues(origin, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new LruGrainRandomAccessVectorValues(origin, 1, 0));
    }
}
