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
        int slots = PruneKernels.SLOTS;
        int[] ids = new int[n * slots];
        float[] scores = new float[n * slots];
        int[] sizes = new int[n];
        fill(candidates, n, ids, scores, sizes);
        pruner.prune(ids, scores, sizes, IntStream.range(0, n).toArray());
        int[] grown = mergeReverseEdges(n, ids, scores, sizes, 0);
        pruner.prune(ids, scores, sizes, grown);
        return new Adjacency(ids, scores, sizes);
    }

    /** Copies the candidate lists into {@code SLOTS}-wide rows, {@code -1} padded. */
    private static void fill(Candidates.TopK candidates, int n, int[] ids, float[] scores, int[] sizes) {
        int slots = PruneKernels.SLOTS;
        int k = candidates.k();
        Arrays.fill(ids, -1);
        IntStream.range(0, n).parallel().forEach(i -> {
            int c = 0;
            for (int j = 0; j < k && c < slots; j++) {
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
        int[] inDegree = new int[n];
        for (int i = firstSource; i < n; i++) {
            for (int s = 0; s < sizes[i]; s++) {
                inDegree[ids[i * slots + s]]++;
            }
        }
        int[] start = new int[n + 1];
        for (int i = 0; i < n; i++) {
            start[i + 1] = start[i] + inDegree[i];
        }
        int[] revIds = new int[start[n]];
        float[] revScores = new float[start[n]];
        int[] fill = Arrays.copyOf(start, n);
        for (int i = firstSource; i < n; i++) {
            for (int s = 0; s < sizes[i]; s++) {
                int j = ids[i * slots + s];
                revIds[fill[j]] = i;
                revScores[fill[j]++] = scores[i * slots + s];
            }
        }
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
