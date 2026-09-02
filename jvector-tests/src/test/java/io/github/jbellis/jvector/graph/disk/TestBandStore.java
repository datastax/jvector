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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Records spilled by band under a permutation come back by old ordinal, exactly, from any thread; the spill is removed on close. */
public class TestBandStore {
    private Path dir;

    @Before
    public void setup() throws IOException {
        dir = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(dir);
    }

    @Test
    public void testRoundTripAcrossBands() throws Exception {
        final int size = 3000, dim = 16, degree = 6, start = 1000;
        var rnd = new Random(1);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        // live nodes: 9 of 10; new ordinals: a random permutation of [start, start + live)
        boolean[] live = new boolean[size];
        int liveCount = 0;
        for (int i = 0; i < size; i++) {
            live[i] = rnd.nextInt(10) != 0;
            if (live[i]) liveCount++;
        }
        List<Integer> slots = new ArrayList<>();
        for (int i = 0; i < liveCount; i++) slots.add(start + i);
        Collections.shuffle(slots, rnd);
        int[] oldToNew = new int[size];
        Arrays.fill(oldToNew, -1);
        int k = 0;
        for (int i = 0; i < size; i++) if (live[i]) oldToNew[i] = slots.get(k++);
        VectorFloat<?>[] vectors = new VectorFloat<?>[size];
        int[][] edges = new int[size][];
        for (int i = 0; i < size; i++) {
            vectors[i] = TestUtil.randomVector(rnd, dim);
            edges[i] = new int[rnd.nextInt(degree + 1)];
            for (int e = 0; e < edges[i].length; e++) edges[i][e] = rnd.nextInt(size);
        }

        var store = new BandStore(dir, 7, start, liveCount, dim, degree, 256, old -> oldToNew[old]);
        assertEquals(256, store.bandNodes);
        assertEquals((liveCount + 255) / 256, store.numBands);
        // distribute from several threads, each a window of old ordinals
        int window = 500;
        List<Thread> threads = new ArrayList<>();
        for (int lo = 0; lo < size; lo += window) {
            final int from = lo, to = Math.min(size, lo + window);
            Thread t = new Thread(() -> {
                try {
                    for (int old = from; old < to; old++) {
                        if (!live[old]) continue;
                        store.put(old, vectors[old], new NodesIterator.ArrayNodesIterator(edges[old], edges[old].length));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
            threads.add(t);
            t.start();
        }
        for (Thread t : threads) t.join();
        store.finishDistribute();
        assertEquals(liveCount, store.records());
        assertEquals((long) liveCount * store.recordBytes, store.bytes());

        VectorFloat<?> vec = vts.createFloatVector(dim);
        int[] nbs = new int[degree];
        for (int old = 0; old < size; old++) {
            assertEquals(live[old], store.has(old));
            if (!live[old]) continue;
            store.vectorInto(old, vec);
            for (int d = 0; d < dim; d++) assertEquals(vectors[old].get(d), vec.get(d), 0f);
            int count = store.neighbors(old, nbs);
            assertArrayEquals("edges of " + old, edges[old], Arrays.copyOf(nbs, count));
        }
        assertEquals(liveCount, store.vectorsServed());
        assertTrue(store.bandsMapped() > 1);
        Path bandDir = store.directory();
        assertTrue(Files.isDirectory(bandDir));
        store.close();
        assertFalse("spill removed on close", Files.exists(bandDir));
        try (Stream<Path> left = Files.list(dir)) {
            assertEquals(0, left.count());
        }
    }

    @Test
    public void testBandWidthIsClampedToTheFileCap() throws Exception {
        // a record this wide would put 1<<18 of them past MAX_BAND_BYTES: the width shrinks to fit
        int dim = 8192, degree = 64;
        var store = new BandStore(dir, 0, 0, 10, dim, degree, 1 << 18, old -> old);
        assertTrue((long) store.bandNodes * store.recordBytes <= BandStore.MAX_BAND_BYTES);
        assertTrue(store.bandNodes < (1 << 18));
        store.close();
    }
}
