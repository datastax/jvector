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
import java.util.Random;

/**
 * Decides how many candidates per node the diversity pruning chooses from.
 * <p>
 * Pruning costs grow with the number of candidates. The pruning keeps up to {@code maxDegree} neighbors per node and
 * drops candidates that are redundant (closer to an already kept neighbor than to the node). When most nodes still
 * end up with a full list from the first {@value #NARROW} candidates, more candidates would only be dropped, and the
 * short list is enough; this is the case for typical embeddings. When many lists come out short, the pruning ran out
 * of useful candidates and gets the whole list.
 * <p>
 * The test is the pruning itself, on a random sample of nodes.
 */
final class CandidateWidth {

    /** The short list. */
    static final int NARROW = 48;
    /** Nodes in the sample: one batch of the pruner. */
    static final int SAMPLES = GpuPrune.BATCH;
    /** The fraction of sampled nodes whose pruned short list must reach {@code maxDegree}. */
    static final double MIN_FULL = 0.95;

    /** The chosen width, and the fraction of sampled nodes the short list filled. */
    record Choice(int width, double full) {
    }

    private CandidateWidth() {
    }

    /** A width set by the user: all {@code k} candidates. */
    static Choice fixed(int k) {
        return new Choice(k, Double.NaN);
    }

    /**
     * @param pruner     the pruner of the build
     * @param candidates every node's candidates, nearest first
     * @param maxDegree  the neighbors the pruning keeps per node
     */
    static Choice choose(GpuPrune pruner, Candidates.TopK candidates, int maxDegree) {
        int k = candidates.k();
        if (k <= NARROW) {
            return fixed(k);
        }
        int n = candidates.ids().length / k;
        int[] sample = new Random(42).ints(0, n).distinct().limit(Math.min(SAMPLES, n)).toArray();
        double full = fullAfterPruning(pruner, candidates, sample, maxDegree);
        return new Choice(full >= MIN_FULL ? NARROW : k, full);
    }

    /** Prunes the short lists of the sampled nodes and returns the fraction that keep {@code maxDegree} neighbors. */
    private static double fullAfterPruning(GpuPrune pruner, Candidates.TopK candidates, int[] sample, int maxDegree) {
        int slots = PruneKernels.SLOTS;
        int k = candidates.k();
        // the sampled nodes' lists, as rows 0..s-1 of the pruner's layout
        int[] ids = new int[sample.length * slots];
        float[] scores = new float[sample.length * slots];
        int[] sizes = new int[sample.length];
        Arrays.fill(ids, -1);
        for (int row = 0; row < sample.length; row++) {
            int size = 0;
            for (int c = 0; c < NARROW; c++) {
                int id = candidates.ids()[sample[row] * k + c];
                if (id >= 0) {
                    ids[row * slots + size] = id;
                    scores[row * slots + size] = candidates.scores()[sample[row] * k + c];
                    size++;
                }
            }
            sizes[row] = size;
        }
        int[] rows = new int[sample.length];
        Arrays.setAll(rows, row -> row);
        pruner.prune(ids, scores, sizes, rows);
        return Arrays.stream(sizes).filter(size -> size >= maxDegree).count() / (double) sample.length;
    }
}
