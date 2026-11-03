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

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Diversity pruning of candidate lists on the GPU, in batches of nodes. The packed vectors stay on the device for
 * the pruner's lifetime; each batch transfers only its candidate lists and the keep flags.
 */
final class GpuPrune implements AutoCloseable {

    /** Nodes per batch. */
    static final int BATCH = 4096;

    private final IntArray candIds;
    private final FloatArray candScores;
    private final FloatArray candNorms;
    private final IntArray counts;
    private final IntArray params;
    private final IntArray keep;
    private final float[] squaredNorms;
    private final WorkerGrid gramGrid;
    private final WorkerGrid pruneGrid;
    private final GridScheduler scheduler;
    private final TornadoExecutionPlan plan;

    /**
     * @param vectors   the packed vectors (with squared norms for EUCLIDEAN)
     * @param maxDegree neighbors to keep per node
     * @param alpha     the builder's alpha
     */
    GpuPrune(PackedVectors vectors, int maxDegree, float alpha) {
        int slots = PruneKernels.SLOTS;
        squaredNorms = vectors.squaredNorms;
        candIds = new IntArray(BATCH * slots);
        candScores = new FloatArray(BATCH * slots);
        candNorms = new FloatArray(BATCH * slots);
        counts = new IntArray(BATCH);
        params = IntArray.fromElements(0, vectors.paddedDim, maxDegree, vectors.rowsPerShard, squaredNorms != null ? 1 : 0);
        keep = new IntArray(BATCH * slots);
        FloatArray gram = new FloatArray(BATCH * slots * slots);
        KernelContext ctx = new KernelContext();
        IntArray[] sh = vectors.shards;
        TaskGraph tg = new TaskGraph("prune")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, sh[0], sh[1], sh[2], sh[3])
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, candIds, candScores, candNorms, counts, params)
                .task("gram", PruneKernels::gramMmaPacked, ctx, sh[0], sh[1], sh[2], sh[3], candIds, params, gram)
                .task("prune", PruneKernels::prune, ctx, gram, candScores, candNorms, counts, params, keep, alpha)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, keep);
        gramGrid = new WorkerGrid1D(PruneKernels.MMA_WARPS * 32);
        gramGrid.setLocalWork(PruneKernels.MMA_WARPS * 32, 1, 1);
        pruneGrid = new WorkerGrid1D(slots);
        pruneGrid.setLocalWork(slots, 1, 1);
        scheduler = new GridScheduler();
        scheduler.addWorkerGrid("prune.gram", gramGrid);
        scheduler.addWorkerGrid("prune.prune", pruneGrid);
        plan = new TornadoExecutionPlan(tg.snapshot());
    }

    /**
     * Compiles the kernels now, without running them or allocating device memory, so that the first {@link #prune}
     * does not wait for TornadoVM's JIT compiler. Safe to call while other GPU work runs.
     */
    GpuPrune precompile() {
        plan.withGridScheduler(scheduler).withPreCompilation();
        return this;
    }

    /**
     * Prunes the given lists in place. {@code ids/scores} hold {@link PruneKernels#SLOTS} entries per node, sorted by
     * decreasing score and {@code -1} padded; {@code sizes} holds the number of valid entries. On return each list
     * keeps only the selected entries, compacted in order, and {@code sizes} is updated.
     *
     * @param nodes the nodes (rows of ids/scores/sizes) to prune
     */
    void prune(int[] ids, float[] scores, int[] sizes, int[] nodes) {
        int slots = PruneKernels.SLOTS;
        for (int from = 0; from < nodes.length; from += BATCH) {
            int size = Math.min(BATCH, nodes.length - from);
            for (int b = 0; b < size; b++) {
                int node = nodes[from + b];
                for (int s = 0; s < slots; s++) {
                    int id = ids[node * slots + s];
                    candIds.set(b * slots + s, id);
                    candScores.set(b * slots + s, scores[node * slots + s]);
                    if (squaredNorms != null) {
                        candNorms.set(b * slots + s, id >= 0 ? squaredNorms[id] : 0);
                    }
                }
                counts.set(b, sizes[node]);
            }
            params.set(PruneKernels.P_NODES, size);
            gramGrid.setGlobalWork((long) size * PruneKernels.MMA_WARPS * 32, 1, 1);
            pruneGrid.setGlobalWork((long) size * slots, 1, 1);
            plan.withGridScheduler(scheduler).execute();
            for (int b = 0; b < size; b++) {
                int node = nodes[from + b];
                int w = 0;
                for (int s = 0; s < sizes[node]; s++) {
                    if (keep.get(b * slots + s) == 1) {
                        ids[node * slots + w] = ids[node * slots + s];
                        scores[node * slots + w] = scores[node * slots + s];
                        w++;
                    }
                }
                for (int s = w; s < slots; s++) {
                    ids[node * slots + s] = -1;
                    scores[node * slots + s] = -1;
                }
                sizes[node] = w;
            }
        }
    }

    @Override
    public void close() {
        try {
            plan.close();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
