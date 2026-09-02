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

import io.github.jbellis.jvector.graph.ParallelExecutor;

import java.util.Arrays;

/**
 * Ascending sort of a {@code long[]} that runs entirely on a caller-supplied
 * {@link ParallelExecutor}.
 *
 * <p>Exists because {@link Arrays#parallelSort(long[])} takes no pool and always decomposes onto
 * {@link java.util.concurrent.ForkJoinPool#commonPool()}. That is an escape hatch out of the
 * embedder's thread budget: a host that bound a compaction to one thread would still see the
 * ordinal-assignment sorts saturate every core. Here concurrency, parallelism, and the chunk
 * split are all decided by the executor and the parallelism the embedder stated.
 *
 * <p>Bottom-up merge sort: the range is cut into {@code chunks} contiguous pieces sorted
 * independently, then merged pairwise, one executor round per merge level. Working space is one
 * additional {@code long[length]} — the same bound {@code Arrays.parallelSort} documents for
 * itself, so this is not a memory regression. Sorting is by signed 64-bit value, identical to
 * {@code Arrays.sort}/{@code Arrays.parallelSort}, which matters because callers pack a key into
 * the high 32 bits and keys with bit 31 set therefore sort as negative.
 */
final class CompactionSort {

    /**
     * Below this many elements a range is sorted in one piece. Splitting finer costs more in
     * merge passes and executor round-trips than it recovers, and it is the same floor
     * {@code Arrays.parallelSort} uses before it stops decomposing.
     */
    static final int MIN_CHUNK = 1 << 13;

    private CompactionSort() {
    }

    /**
     * Sorts {@code a[0, length)} ascending.
     *
     * @param a           the array to sort in place
     * @param length      number of leading elements to sort
     * @param executor    runs each chunk-sort and merge round to completion
     * @param parallelism the executor's stated width; caps the chunk count
     */
    static void sort(long[] a, int length, ParallelExecutor executor, int parallelism) {
        if (length < 0 || length > a.length) {
            throw new IllegalArgumentException("length " + length + " out of range for array of " + a.length);
        }
        final int chunks = chunkCount(length, parallelism);
        if (chunks <= 1) {
            Arrays.sort(a, 0, length);
            return;
        }

        // Chunk boundaries computed once and shared: every round indexes the same split, and
        // clamping to `chunks` is what lets an odd trailing run merge against an empty range.
        final int[] bounds = new int[chunks + 1];
        for (int c = 0; c <= chunks; c++) {
            bounds[c] = (int) ((long) length * c / chunks);
        }

        executor.forEachInt(chunks, c -> Arrays.sort(a, bounds[c], bounds[c + 1]));

        long[] src = a;
        long[] dst = new long[length];
        for (int width = 1; width < chunks; width <<= 1) {
            final long[] from = src;
            final long[] to = dst;
            final int w = width;
            final int runs = (chunks + 2 * w - 1) / (2 * w);
            executor.forEachInt(runs, r -> {
                int lo = bounds[Math.min(r * 2 * w, chunks)];
                int mid = bounds[Math.min(r * 2 * w + w, chunks)];
                int hi = bounds[Math.min(r * 2 * w + 2 * w, chunks)];
                merge(from, lo, mid, hi, to);
            });
            long[] swap = src;
            src = dst;
            dst = swap;
        }

        // An odd number of merge rounds leaves the result in the scratch buffer.
        if (src != a) {
            System.arraycopy(src, 0, a, 0, length);
        }
    }

    /**
     * How many pieces to cut {@code length} into: never more than the executor can run at once,
     * and never so many that a piece drops below {@link #MIN_CHUNK}.
     */
    private static int chunkCount(int length, int parallelism) {
        if (parallelism <= 1 || length < 2 * MIN_CHUNK) {
            return 1;
        }
        return Math.max(1, Math.min(parallelism, length / MIN_CHUNK));
    }

    /** Merges the sorted runs {@code src[lo, mid)} and {@code src[mid, hi)} into {@code dst[lo, hi)}. */
    private static void merge(long[] src, int lo, int mid, int hi, long[] dst) {
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi) {
            dst[k++] = src[i] <= src[j] ? src[i++] : src[j++];
        }
        while (i < mid) {
            dst[k++] = src[i++];
        }
        while (j < hi) {
            dst[k++] = src[j++];
        }
    }
}
