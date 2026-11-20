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

package io.github.jbellis.jvector.gpu;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Neighbor lists from candidates, as JVector's incremental build would make them:
 * <ol>
 * <li>forward: each node's candidates are diversity-pruned to at most {@code maxDegree} (GPU);</li>
 * <li>reverse: every kept edge {@code i -> j} is offered to {@code j}; each list is merged with its reverse edges,
 * deduplicated, sorted and capped at {@link PruneKernels#SLOTS} (CPU);</li>
 * <li>the lists that gained reverse edges are pruned again with the same rule (GPU).</li>
 * </ol>
 */
final class GpuGraphBuilder {

    /** {@link PruneKernels#SLOTS} entries per node, {@code sizes[i]} valid ones, sorted by decreasing score. */
    record Adjacency(int[] ids, float[] scores, int[] sizes) {

        /** {@code n} nodes without neighbors. */
        static Adjacency empty(int n) {
            int[] ids = new int[n * PruneKernels.SLOTS];
            Arrays.fill(ids, -1);
            return new Adjacency(ids, new float[n * PruneKernels.SLOTS], new int[n]);
        }
    }

    private GpuGraphBuilder() {
    }

    static Adjacency build(PackedVectors vectors, int n, Candidates.TopK candidates, int maxDegree, float alpha) {
        try (GpuPrune pruner = new GpuPrune(vectors, maxDegree, alpha)) {
            return build(pruner, n, candidates);
        }
    }

    /** As {@link #build(PackedVectors, int, Candidates.TopK, int, float)}, with a pruner prepared by the caller. */
    static Adjacency build(GpuPrune pruner, int n, Candidates.TopK candidates) {
        return build(pruner, n, candidates, candidates.k());
    }

    /** As {@link #build(GpuPrune, int, Candidates.TopK)}, from the first {@code width} candidates of every list. */
    static Adjacency build(GpuPrune pruner, int n, Candidates.TopK candidates, int width) {
        int slots = PruneKernels.SLOTS;
        int[] ids = new int[n * slots];
        float[] scores = new float[n * slots];
        int[] sizes = new int[n];
        long t0 = System.nanoTime();
        fill(candidates, Math.min(width, candidates.k()), n, ids, scores, sizes);
        long t1 = System.nanoTime();
        pruner.prune(ids, scores, sizes, IntStream.range(0, n).toArray());
        long t2 = System.nanoTime();
        int[] grown = mergeReverseEdges(n, ids, scores, sizes, 0);
        long t3 = System.nanoTime();
        pruner.prune(ids, scores, sizes, grown);
        if (Boolean.getBoolean("jvector.gpu.trace")) {
            System.err.printf("graph build of %d: fill %.3f s, prune %.3f s, reverse merge %.3f s, prune %d grown %.3f s%n", n, (t1 - t0) / 1e9,
                    (t2 - t1) / 1e9, (t3 - t2) / 1e9, grown.length, (System.nanoTime() - t3) / 1e9);
        }
        return new Adjacency(ids, scores, sizes);
    }

    /** Copies the first {@code width} candidates of every list into {@code SLOTS}-wide rows, {@code -1} padded. */
    private static void fill(Candidates.TopK candidates, int width, int n, int[] ids, float[] scores, int[] sizes) {
        int slots = PruneKernels.SLOTS;
        int k = candidates.k();
        Arrays.fill(ids, -1);
        IntStream.range(0, n).parallel().forEach(i -> {
            int c = 0;
            for (int j = 0; j < width && c < slots; j++) {
                int id = candidates.ids()[i * k + j];
                if (id >= 0) {
                    ids[i * slots + c] = id;
                    scores[i * slots + c] = candidates.scores()[i * k + j];
                    c++;
                }
            }
            sizes[i] = c;
        });
    }

    /** @return the nodes whose lists gained reverse edges, from the lists of nodes {@code firstSource} and up */
    private static int[] mergeReverseEdges(int n, int[] ids, float[] scores, int[] sizes, int firstSource) {
        int slots = PruneKernels.SLOTS;
        // in-degrees and the reverse lists, built in parallel: a slot per edge is claimed with an atomic increment,
        // then each node's reverse list is sorted by source so the result does not depend on thread timing
        var inDegreeCount = new java.util.concurrent.atomic.AtomicIntegerArray(n);
        IntStream.range(firstSource, n).parallel().forEach(i -> {
            for (int s = 0; s < sizes[i]; s++) {
                inDegreeCount.incrementAndGet(ids[i * slots + s]);
            }
        });
        int[] inDegree = new int[n];
        int[] start = new int[n + 1];
        for (int i = 0; i < n; i++) {
            inDegree[i] = inDegreeCount.get(i);
            start[i + 1] = start[i] + inDegree[i];
        }
        // each reverse edge as (source << 32 | slot in the source's list), sorted by source per target: the order
        // the sequential loop produced, whatever the thread timing
        long[] rev = new long[start[n]];
        var fill = new java.util.concurrent.atomic.AtomicIntegerArray(Arrays.copyOf(start, n));
        IntStream.range(firstSource, n).parallel().forEach(i -> {
            for (int s = 0; s < sizes[i]; s++) {
                rev[fill.getAndIncrement(ids[i * slots + s])] = ((long) i << 32) | s;
            }
        });
        int[] revIds = new int[start[n]];
        float[] revScores = new float[start[n]];
        IntStream.range(0, n).parallel().forEach(j -> {
            Arrays.sort(rev, start[j], start[j + 1]);
            for (int r = start[j]; r < start[j + 1]; r++) {
                int i = (int) (rev[r] >>> 32);
                revIds[r] = i;
                revScores[r] = scores[i * slots + (int) rev[r]];
            }
        });
        return IntStream.range(0, n).parallel().filter(i -> {
            int forward = sizes[i];
            int[] mIds = new int[forward + inDegree[i]];
            float[] mScores = new float[mIds.length];
            int m = 0;
            for (int s = 0; s < forward; s++) {
                mIds[m] = ids[i * slots + s];
                mScores[m++] = scores[i * slots + s];
            }
            outer:
            for (int r = start[i]; r < start[i + 1]; r++) {
                for (int s = 0; s < forward; s++) {
                    if (ids[i * slots + s] == revIds[r]) {
                        continue outer;
                    }
                }
                mIds[m] = revIds[r];
                mScores[m++] = revScores[r];
            }
            if (m == forward) {
                return false;
            }
            // sort by decreasing score: (order-preserving score key, position) packed in a long, ascending
            long[] order = new long[m];
            for (int x = 0; x < m; x++) {
                order[x] = ((long) descendingKey(mScores[x]) << 32) | x;
            }
            Arrays.sort(order);
            int keep = Math.min(m, slots);
            for (int x = 0; x < slots; x++) {
                int at = x < keep ? (int) order[x] : -1;
                ids[i * slots + x] = at >= 0 ? mIds[at] : -1;
                scores[i * slots + x] = at >= 0 ? mScores[at] : -1;
            }
            sizes[i] = keep;
            return true;
        }).toArray();
    }

    /** An int that sorts ascending as the float sorts descending. */
    static int descendingKey(float score) {
        int bits = Float.floatToIntBits(score);
        return ~(bits ^ ((bits >> 31) & 0x7fffffff));
    }
}
