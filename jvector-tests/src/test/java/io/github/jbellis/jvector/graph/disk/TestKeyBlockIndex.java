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

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Block extents, overlap, the unclustered signal, and run merging with a budget. */
public class TestKeyBlockIndex {
    @Test
    public void testSortedSourceHasTightBlocks() {
        // 100 nodes, keys ascending with the ordinal: block b spans [b*1000, b*1000+900]
        var idx = new KeyBlockIndex(100, 10);
        for (int n = 0; n < 100; n++) {
            idx.add(n, (n / 10) * 1000 + (n % 10) * 100);
        }
        assertEquals(10, idx.blocks());
        assertTrue(idx.overlaps(3, 3100, 3200));
        assertFalse(idx.overlaps(4, 3100, 3200));
        assertEquals(0.1, idx.overlapFraction(3100, 3200), 1e-9);
        assertEquals(0.3, idx.overlapFraction(2500, 4500), 1e-9);
        // three consecutive blocks merge into one run
        List<int[]> runs = idx.runsFor(2500, 4500, 100, 1 << 20);
        assertEquals(1, runs.size());
        assertArrayEquals(new int[] {20, 49}, runs.get(0));
        // a budget truncates the last run
        runs = idx.runsFor(2500, 4500, 100, 15);
        assertEquals(1, runs.size());
        assertArrayEquals(new int[] {20, 34}, runs.get(0));
        // non-adjacent blocks stay separate runs
        var sparse = new KeyBlockIndex(40, 10);
        for (int n = 0; n < 40; n++) {
            sparse.add(n, (n / 10) % 2 == 0 ? 5 : 500);
        }
        runs = sparse.runsFor(0, 10, 40, 1 << 20);
        assertEquals(2, runs.size());
        assertArrayEquals(new int[] {0, 9}, runs.get(0));
        assertArrayEquals(new int[] {20, 29}, runs.get(1));
    }

    @Test
    public void testArrivalOrderedSourceOverlapsEverywhere() {
        var idx = new KeyBlockIndex(100, 10);
        var rnd = new java.util.Random(4);
        for (int n = 0; n < 100; n++) {
            idx.add(n, rnd.nextInt());
        }
        // ten random keys per block span most of the unsigned range, so a modest range hits
        // nearly every block (0.9 under this seed) — far above the 0.5 that marks a source
        // as unclustered
        assertTrue(idx.overlapFraction(1L << 30, (1L << 30) + (1L << 29)) >= 0.8);
        // empty blocks (dead nodes) never overlap and do not count
        var holes = new KeyBlockIndex(30, 10);
        holes.add(0, 1);
        holes.add(25, 1);
        assertFalse(holes.overlaps(1, 0, 10));
        assertEquals(1.0, holes.overlapFraction(0, 10), 1e-9);
        // unsigned keys: a negative int is a large key
        var unsigned = new KeyBlockIndex(2, 1);
        unsigned.add(0, -1);
        assertTrue(unsigned.overlaps(0, 0xFFFFFFFFL, 0xFFFFFFFFL));
        assertFalse(unsigned.overlaps(0, 0, 10));
    }
}
