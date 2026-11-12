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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;

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
    /** Nodes per slice of the parallel copies into and out of the kernels' arrays. */
    private static final int COPY_NODES = 512;
    /** Buffers for the kept flags of the batches still being applied. */
    private static final int KEPT_RING = 4;

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

    /** One batch's lists in host arrays, laid out as the kernels read them. */
    private static final class Staging {
        final int[] ids = new int[BATCH * PruneKernels.SLOTS];
        final float[] scores = new float[BATCH * PruneKernels.SLOTS];
        final float[] norms = new float[BATCH * PruneKernels.SLOTS];
        final int[] counts = new int[BATCH];
        int size;
    }

    /**
     * Prunes the given lists in place. {@code ids/scores} hold {@link PruneKernels#SLOTS} entries per node, sorted by
     * decreasing score and {@code -1} padded; {@code sizes} holds the number of valid entries. On return each list
     * keeps only the selected entries, compacted in order, and {@code sizes} is updated.
     * <p>
     * The batches are pipelined: while the GPU prunes one batch, other threads lay out the next one and apply the
     * previous one's result, so the host work overlaps the GPU's. A node appears in one batch only, so the batches
     * touch disjoint rows of {@code ids/scores/sizes}.
     *
     * @param nodes the nodes (rows of ids/scores/sizes) to prune; each at most once
     */
    void prune(int[] ids, float[] scores, int[] sizes, int[] nodes) {
        int batches = (nodes.length + BATCH - 1) / BATCH;
        if (batches == 0) {
            return;
        }
        Staging[] staging = {new Staging(), new Staging()};
        int[][] keptRing = new int[KEPT_RING][BATCH * PruneKernels.SLOTS];
        CompletableFuture<Staging> next = CompletableFuture.supplyAsync(() -> stage(staging[0], ids, scores, sizes, nodes, 0));
        List<CompletableFuture<Void>> applied = new ArrayList<>(batches);
        long tWait = 0, tCopy = 0, tExec = 0, tRead = 0, t0 = System.nanoTime();
        for (int b = 0; b < batches; b++) {
            long ta = System.nanoTime();
            Staging current = next.join();
            tWait += System.nanoTime() - ta;
            if (b + 1 < batches) {
                int from = (b + 1) * BATCH;
                Staging other = staging[(b + 1) % 2];
                next = CompletableFuture.supplyAsync(() -> stage(other, ids, scores, sizes, nodes, from));
            }
            int size = current.size;
            int n = size * PruneKernels.SLOTS;
            long tb = System.nanoTime();
            // the batch into the kernels' arrays, in parallel slices: one thread copies 5 MB a batch at a few GB/s,
            // 1.3 s of a 10M-node prune on its own
            IntStream.range(0, (size + COPY_NODES - 1) / COPY_NODES).parallel().forEach(c -> {
                long at = (long) c * COPY_NODES * PruneKernels.SLOTS;
                int len = (int) (Math.min((long) size * PruneKernels.SLOTS, at + (long) COPY_NODES * PruneKernels.SLOTS) - at);
                MemorySegment.copy(current.ids, (int) at, candIds.getSegment(), ValueLayout.JAVA_INT, at * Integer.BYTES, len);
                MemorySegment.copy(current.scores, (int) at, candScores.getSegment(), ValueLayout.JAVA_FLOAT, at * Float.BYTES, len);
                if (squaredNorms != null) {
                    MemorySegment.copy(current.norms, (int) at, candNorms.getSegment(), ValueLayout.JAVA_FLOAT, at * Float.BYTES, len);
                }
            });
            MemorySegment.copy(current.counts, 0, counts.getSegment(), ValueLayout.JAVA_INT, 0, size);
            params.set(PruneKernels.P_NODES, size);
            gramGrid.setGlobalWork((long) size * PruneKernels.MMA_WARPS * 32, 1, 1);
            pruneGrid.setGlobalWork((long) size * PruneKernels.SLOTS, 1, 1);
            long tc = System.nanoTime();
            plan.withGridScheduler(scheduler).execute();
            long td = System.nanoTime();
            // the kept flags out in parallel slices, into a buffer of a small ring (its last reader joined first)
            int slot = b % KEPT_RING;
            if (b >= KEPT_RING) {
                applied.get(b - KEPT_RING).join();
            }
            int[] kept = keptRing[slot];
            IntStream.range(0, (n + COPY_NODES * PruneKernels.SLOTS - 1) / (COPY_NODES * PruneKernels.SLOTS)).parallel().forEach(c -> {
                int at = c * COPY_NODES * PruneKernels.SLOTS;
                int len = Math.min(n, at + COPY_NODES * PruneKernels.SLOTS) - at;
                MemorySegment.copy(keep.getSegment(), ValueLayout.JAVA_INT, (long) at * Integer.BYTES, kept, at, len);
            });
            tCopy += tc - tb;
            tExec += td - tc;
            tRead += System.nanoTime() - td;
            int from = b * BATCH;
            applied.add(CompletableFuture.runAsync(() -> apply(kept, ids, scores, sizes, nodes, from, size)));
        }
        long te = System.nanoTime();
        CompletableFuture.allOf(applied.toArray(CompletableFuture[]::new)).join();
        if (Boolean.getBoolean("jvector.gpu.trace") && nodes.length > 1_000_000) {
            System.err.printf("prune %d nodes, %d batches: wait staging %.2f s, copy in %.2f s, execute %.2f s, read keep %.2f s, wait apply %.2f s, total %.2f s%n",
                    nodes.length, batches, tWait / 1e9, tCopy / 1e9, tExec / 1e9, tRead / 1e9, (System.nanoTime() - te) / 1e9, (System.nanoTime() - t0) / 1e9);
        }
    }

    /** Lays out the lists of {@code nodes[from ..]} (one batch) in {@code s}. */
    private Staging stage(Staging s, int[] ids, float[] scores, int[] sizes, int[] nodes, int from) {
        int slots = PruneKernels.SLOTS;
        int size = Math.min(BATCH, nodes.length - from);
        IntStream.range(0, size).parallel().forEach(r -> {
            int node = nodes[from + r];
            System.arraycopy(ids, node * slots, s.ids, r * slots, slots);
            System.arraycopy(scores, node * slots, s.scores, r * slots, slots);
            if (squaredNorms != null) {
                for (int c = 0; c < slots; c++) {
                    int id = s.ids[r * slots + c];
                    s.norms[r * slots + c] = id >= 0 ? squaredNorms[id] : 0;
                }
            }
            s.counts[r] = sizes[node];
        });
        s.size = size;
        return s;
    }

    /** Keeps the flagged entries of each list of {@code nodes[from .. from + size)}, compacted in order. */
    private static void apply(int[] kept, int[] ids, float[] scores, int[] sizes, int[] nodes, int from, int size) {
        int slots = PruneKernels.SLOTS;
        IntStream.range(0, size).parallel().forEach(r -> {
            int node = nodes[from + r];
            int w = 0;
            for (int c = 0; c < sizes[node]; c++) {
                if (kept[r * slots + c] == 1) {
                    ids[node * slots + w] = ids[node * slots + c];
                    scores[node * slots + w] = scores[node * slots + c];
                    w++;
                }
            }
            for (int c = w; c < slots; c++) {
                ids[node * slots + c] = -1;
                scores[node * slots + c] = -1;
            }
            sizes[node] = w;
        });
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
