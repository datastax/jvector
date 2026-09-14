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

import java.util.*;
import java.util.concurrent.*;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.*;
import static java.lang.Math.*;

/**
 * Scan results of a full-precision merge for one range of a searching source's new ordinals:
 * per larger target and node, the best {@code k} scanned ordinals with their exact scores, as
 * packed {@code (score bits << 32 | ordinal)} min-heaps in one flat array.
 */
final class ExactJoin {
    final int src, lo, n, k;
    final int[] targetIndex;           // source -> index among this range's targets, or -1
    final int targets;
    final long[] heaps;                // [(targetIndex * n + i) * k + j]
    final int[] sizes;                 // [targetIndex * n + i]
    final float[] thresholds;          // score of the k-th best so far, -inf until the heap is full
    final Object[] locks = new Object[1024];

    private final int numSources;
    private final int[] sizeRank;

    ExactJoin(int src, int lo, int hi, int k, int numSources, int[] sizeRank) {
        this.numSources = numSources;
        this.sizeRank = sizeRank;
        this.src = src;
        this.lo = lo;
        this.n = hi - lo;
        this.k = k;
        targetIndex = new int[numSources];
        Arrays.fill(targetIndex, -1);
        int t = 0;
        for (int i = 0; i < numSources; i++) {
            if (i != src && sizeRank[i] > sizeRank[src]) targetIndex[i] = t++;
        }
        targets = t;
        heaps = new long[Math.multiplyExact(Math.multiplyExact(targets, n), k)];
        sizes = new int[targets * n];
        thresholds = new float[targets * n];
        Arrays.fill(thresholds, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < locks.length; i++) locks[i] = new Object();
    }

    int slot(int target, int newOrdinal) {
        return targetIndex[target] * n + (newOrdinal - lo);
    }

    /** Offers (newOrd, score) to the slot's heap; scores are positive, so their bits order like the floats. */
    void offer(int slot, int newOrd, float score) {
        long key = ((long) Float.floatToRawIntBits(score) << 32) | (newOrd & 0xFFFFFFFFL);
        synchronized (locks[slot & (locks.length - 1)]) {
            int base = slot * k, size = sizes[slot];
            long[] h = heaps;
            if (size < k) {
                int i = size;
                while (i > 0) {
                    int parent = (i - 1) >>> 1;
                    if (h[base + parent] <= key) break;
                    h[base + i] = h[base + parent];
                    i = parent;
                }
                h[base + i] = key;
                sizes[slot] = ++size;
            } else if (key > h[base]) {
                int i = 0;
                while (true) {
                    int l = 2 * i + 1;
                    if (l >= size) break;
                    int r = l + 1;
                    int c = r < size && h[base + r] < h[base + l] ? r : l;
                    if (h[base + c] >= key) break;
                    h[base + i] = h[base + c];
                    i = c;
                }
                h[base + i] = key;
            } else {
                return;
            }
            if (size == k) {
                thresholds[slot] = Float.intBitsToFloat((int) (h[base] >>> 32));
            }
        }
    }
}
