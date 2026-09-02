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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Per-block similarity-key extents of one source, built from its token stream: for every run of
 * {@code blockNodes} ordinals, the smallest and largest {@link SimilarityKey} (unsigned) of its
 * live nodes. On a source whose disk order is a similarity order — a previous merge's output —
 * the blocks that can hold a band's neighbours are the few whose extent overlaps the band's key
 * range, and that is the window worth warming before the band's searches. On a source in
 * arrival order nearly every block overlaps every range, which {@link #overlapFraction} exposes.
 */
final class KeyBlockIndex {
    final int blockNodes;
    private final long[] min;
    private final long[] max;
    private final int[] live;

    KeyBlockIndex(int nodeCount, int blockNodes) {
        this.blockNodes = Math.max(1, blockNodes);
        int blocks = (nodeCount + this.blockNodes - 1) / this.blockNodes;
        min = new long[blocks];
        max = new long[blocks];
        live = new int[blocks];
        Arrays.fill(min, Long.MAX_VALUE);
        Arrays.fill(max, Long.MIN_VALUE);
    }

    /** Records one live node's key (unsigned 32-bit). */
    void add(int node, int key) {
        int b = node / blockNodes;
        long k = key & 0xFFFFFFFFL;
        if (k < min[b]) min[b] = k;
        if (k > max[b]) max[b] = k;
        live[b]++;
    }

    int blocks() {
        return min.length;
    }

    boolean overlaps(int block, long keyMin, long keyMax) {
        return live[block] > 0 && min[block] <= keyMax && max[block] >= keyMin;
    }

    /** Share of non-empty blocks whose extent overlaps the range. */
    double overlapFraction(long keyMin, long keyMax) {
        int nonEmpty = 0, hit = 0;
        for (int b = 0; b < min.length; b++) {
            if (live[b] == 0) continue;
            nonEmpty++;
            if (overlaps(b, keyMin, keyMax)) hit++;
        }
        return nonEmpty == 0 ? 1.0 : (double) hit / nonEmpty;
    }

    /**
     * Ordinal runs {@code [lo, hi]} covering the blocks that overlap the range, consecutive blocks
     * merged, truncated once {@code maxNodes} would be exceeded.
     */
    List<int[]> runsFor(long keyMin, long keyMax, int nodeCount, int maxNodes) {
        List<int[]> runs = new ArrayList<>();
        long budget = maxNodes;
        int runLo = -1, runHi = -1;
        for (int b = 0; b < min.length && budget > 0; b++) {
            if (!overlaps(b, keyMin, keyMax)) {
                if (runLo >= 0) {
                    runs.add(new int[] {runLo, runHi});
                    runLo = -1;
                }
                continue;
            }
            int lo = b * blockNodes;
            int hi = Math.min(nodeCount - 1, lo + blockNodes - 1);
            int take = (int) Math.min(budget, hi - lo + 1);
            hi = lo + take - 1;
            budget -= take;
            if (runLo >= 0 && lo == runHi + 1) {
                runHi = hi;
            } else {
                if (runLo >= 0) runs.add(new int[] {runLo, runHi});
                runLo = lo;
                runHi = hi;
            }
        }
        if (runLo >= 0) runs.add(new int[] {runLo, runHi});
        return runs;
    }
}
