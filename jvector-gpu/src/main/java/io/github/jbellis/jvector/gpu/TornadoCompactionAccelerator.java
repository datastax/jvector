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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.CellProbe;
import io.github.jbellis.jvector.graph.disk.CellWalk;
import io.github.jbellis.jvector.graph.disk.CompactionAccelerator;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
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
 * Runs {@code OnDiskGraphIndexCompactor}'s cell walk on an NVIDIA GPU with TornadoVM. Select it with
 * {@code -Djvector.compaction.accelerator=tornadovm}.
 * <p>
 * The walk stays out of core: the queries are processed in blocks, the target's rows are streamed from the merge's
 * scratch region in chunks of whole cells, and every buffer has a fixed size taken from the device memory budget, so a
 * larger merge runs as more launches, never as a larger allocation. Scores are FP32 and in JVector's units, so the
 * top k per query are the CPU walk's except near-ties.
 */
public final class TornadoCompactionAccelerator implements CompactionAccelerator {

    private static final Logger LOG = Logger.getLogger(TornadoCompactionAccelerator.class.getName());

    public static final String NAME = "tornadovm";

    private static final boolean TRACE = Boolean.getBoolean("jvector.gpu.trace");

    /** Bytes for the row chunk ({@code -Djvector.gpu.walk.chunkMB}, default 512). */
    static final long CHUNK_BYTES = Long.getLong("jvector.gpu.walk.chunkMB", 512) << 20;
    /**
     * Bytes for the query block ({@code -Djvector.gpu.walk.queryMB}, default 2048): a range of the CPU compactor's size
     * (up to 3 GB of queries) then runs as one block, so each target's rows are streamed once.
     */
    static final long QUERY_BYTES = Long.getLong("jvector.gpu.walk.queryMB", 2048) << 20;
    /** (cell, query) pairs per launch. */
    static final int MAX_PAIRS = 1 << 24;
    /** Shares of TornadoVM's device memory budget ({@code -Dtornado.device.memory}) for the walk and the probes. */
    static final double WALK_SHARE = 0.6;
    static final double PROBE_SHARE = 0.25;
    /** Candidate scores per launch. */
    static final int MAX_CANDIDATES = 1 << 27;

    /** Most centers the GPU ranks exhaustively ({@code -Djvector.gpu.probe.maxCenters}, default 100,000). */
    static final int MAX_PROBE_CENTERS = Integer.getInteger("jvector.gpu.probe.maxCenters", 100_000);

    /** Whether the probe step runs on the GPU too ({@code -Djvector.gpu.probe}, default true). */
    static final boolean PROBE = Boolean.parseBoolean(System.getProperty("jvector.gpu.probe", "true"));

    private Walker walker;
    private Prober prober;

    /** Bytes of the score matrix of one block of probe queries ({@code -Djvector.gpu.probe.scoresMB}, default 1024). */
    static final long SCORE_BYTES = Long.getLong("jvector.gpu.probe.scoresMB", 256) << 20;

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public boolean supports(VectorSimilarityFunction similarity, int dimension) {
        return dimension <= CellWalkKernels.MAX_DIM && runtimeAvailable();
    }

    private static boolean runtimeAvailable() {
        try {
            Class.forName("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime");
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    @Override
    public boolean walk(CellWalk walk) {
        if (walk.topK() > CellWalkKernels.MAX_K) {
            return false;
        }
        int dim = walk.dimension();
        // sized to the compactor's range: TornadoVM keeps a resident host copy of every buffer, so a block larger than
        // the range would only take host memory from the merge's page cache
        if (walker == null || walker.dim != dim || walker.blockQueries < Math.min(walk.queryCount(), walker.maxBlockQueries)) {
            if (walker != null) {
                walker.close();
            }
            walker = new Walker(dim, (long) (TornadoGraphBuildAccelerator.deviceMemoryBudget() * WALK_SHARE), walk.queryCount());
        }
        return walker.walk(walk);
    }

    /** Whether the wide code is trained on the GPU too ({@code -Djvector.gpu.wideCode}, default true). */
    static final boolean WIDE_CODE = Boolean.parseBoolean(System.getProperty("jvector.gpu.wideCode", "true"));

    /** The wide code with {@link GpuProductQuantization#compute}, which trains like {@code ProductQuantization.compute}. */
    @Override
    public ProductQuantization trainWideCode(RandomAccessVectorValues sample, int subspaces, int clusters, boolean globallyCenter) {
        if (!WIDE_CODE) {
            return null;
        }
        long t0 = System.nanoTime();
        ProductQuantization pq = GpuProductQuantization.compute(sample, subspaces, clusters, globallyCenter);
        if (TRACE) {
            long ms = (System.nanoTime() - t0) / 1_000_000;
            LOG.info(() -> String.format(Locale.ROOT, "GPU wide code: %d x %d, %d subspaces, %d ms", sample.size(), sample.dimension(), subspaces, ms));
        }
        return pq;
    }

    @Override
    public boolean probe(CellProbe probe) {
        if (!PROBE) {
            return false;
        }
        // exact ranking costs queries x centers x dim, and the centers grow with the merge: past MAX_PROBE_CENTERS the
        // compactor's beam is cheaper (at 10M x 1024, 313k centers made the GPU probes 1.7x slower than the CPU's)
        if (probe.centerCount() > MAX_PROBE_CENTERS) {
            return false;
        }
        if (probe.maxProbes() > ProbeKernels.MAX_PROBES || probe.dimension() > CellWalkKernels.MAX_DIM) {
            return false;
        }
        int dim = probe.dimension();
        int centers = probe.centerCount();
        if (prober == null || prober.dim != dim || prober.centers != centers || prober.maxProbes != probe.maxProbes()) {
            if (prober != null) {
                prober.close();
            }
            long budget = (long) (TornadoGraphBuildAccelerator.deviceMemoryBudget() * PROBE_SHARE);
            if ((long) centers * dim * Float.BYTES > budget / 2) {
                LOG.info(() -> "the level-1 centers take more than half the GPU probe budget; probing on the CPU");
                return false;
            }
            prober = new Prober(dim, centers, probe.maxProbes(), budget);
        }
        return prober.probe(probe);
    }

    /**
     * The probe step: every query against every center with a tiled matrix product, then each query's best centers.
     * The centers stay on the device; the queries go in blocks whose score matrix fits {@link #SCORE_BYTES}.
     */
    static final class Prober implements AutoCloseable {
        final int dim;
        final int centers;
        final int maxProbes;
        final int stride;
        final int blockQueries;
        final FloatArray centerRows;
        final FloatArray centerNorms;
        final IntArray centerParams;
        final FloatArray queries;
        final IntArray params;
        final FloatArray scores;
        final IntArray probes;
        final WorkerGrid normsGrid;
        final WorkerGrid scoreGrid;
        final WorkerGrid selectGrid;
        final GridScheduler scheduler;
        final TornadoExecutionPlan plan;

        Prober(int dim, int centers, int maxProbes, long budget) {
            this.dim = dim;
            this.centers = centers;
            this.maxProbes = maxProbes;
            stride = (centers + ProbeKernels.TILE - 1) / ProbeKernels.TILE * ProbeKernels.TILE;
            // the score matrix and the query block share what the centers leave of the budget
            long left = budget - (long) centers * dim * Float.BYTES;
            long perQuery = (long) stride * Float.BYTES + (long) dim * Float.BYTES + (long) maxProbes * Integer.BYTES;
            long scoreBytes = Math.min(SCORE_BYTES, left / perQuery * stride * Float.BYTES);
            int byScores = (int) Math.min(Integer.MAX_VALUE / stride, scoreBytes / ((long) stride * Float.BYTES));
            blockQueries = Math.max(ProbeKernels.TILE, byScores / ProbeKernels.TILE * ProbeKernels.TILE);
            centerRows = new FloatArray(centers * dim);
            centerNorms = new FloatArray(centers);
            centerParams = new IntArray(4);
            queries = new FloatArray(blockQueries * dim);
            params = new IntArray(8);
            scores = new FloatArray(blockQueries * stride);
            probes = new IntArray(blockQueries * maxProbes);
            KernelContext ctx = new KernelContext();
            TaskGraph upload = new TaskGraph("centers")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, centerRows, centerParams)
                    .task("norms", CellWalkKernels::norms, ctx, centerRows, centerParams, centerNorms)
                    .persistOnDevice(centerRows, centerNorms);
            TaskGraph rank = new TaskGraph("probe")
                    .consumeFromDevice("centers", centerRows, centerNorms)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, queries, params)
                    .task("scores", ProbeKernels::centerScores, ctx, queries, centerRows, centerNorms, params, scores)
                    .task("select", ProbeKernels::selectTop, ctx, scores, params, probes)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, probes);
            normsGrid = new WorkerGrid1D(CellWalkKernels.GROUP);
            normsGrid.setLocalWork(CellWalkKernels.GROUP, 1, 1);
            normsGrid.setGlobalWork((long) centers * CellWalkKernels.GROUP, 1, 1);
            scoreGrid = new WorkerGrid1D(ProbeKernels.GROUP);
            scoreGrid.setLocalWork(ProbeKernels.GROUP, 1, 1);
            selectGrid = new WorkerGrid1D(ProbeKernels.GROUP);
            selectGrid.setLocalWork(ProbeKernels.GROUP, 1, 1);
            scheduler = new GridScheduler();
            scheduler.addWorkerGrid("centers.norms", normsGrid);
            scheduler.addWorkerGrid("probe.scores", scoreGrid);
            scheduler.addWorkerGrid("probe.select", selectGrid);
            plan = new TornadoExecutionPlan(upload.snapshot(), rank.snapshot());
        }

        boolean probe(CellProbe probe) {
            long t0 = System.nanoTime();
            // the centers, once per range: copied out of the hub map, uploaded, their norms computed on the device
            int rowsPerTask = Math.max(1, centers / 64 + 1);
            IntStream.range(0, (centers + rowsPerTask - 1) / rowsPerTask).parallel().forEach(t -> {
                int from = t * rowsPerTask;
                int count = Math.min(rowsPerTask, centers - from);
                float[] rows = new float[count * dim];
                probe.copyCenters(from, count, rows, 0);
                MemorySegment.copy(rows, 0, centerRows.getSegment(), ValueLayout.JAVA_FLOAT, (long) from * dim * Float.BYTES, rows.length);
            });
            centerParams.set(CellWalkKernels.P_DIM, dim);
            plan.withGraph(0).withGridScheduler(scheduler).execute();
            int n = probe.queryCount();
            params.set(ProbeKernels.P_DIM, dim);
            params.set(ProbeKernels.P_CENTERS, centers);
            params.set(ProbeKernels.P_SIMILARITY, Walker.similarityCode(probe.similarity()));
            params.set(ProbeKernels.P_PROBES, maxProbes);
            params.set(ProbeKernels.P_STRIDE, stride);
            int want = Math.min(maxProbes, centers);
            for (int b0 = 0; b0 < n; b0 += blockQueries) {
                final int start = b0;
                int nq = Math.min(blockQueries, n - b0);
                IntStream.range(0, nq).parallel().forEach(i -> {
                    float[] q = new float[dim];
                    probe.query(start + i, q);
                    MemorySegment.copy(q, 0, queries.getSegment(), ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
                });
                params.set(ProbeKernels.P_QUERIES, nq);
                int tilesQ = (nq + ProbeKernels.TILE - 1) / ProbeKernels.TILE;
                scoreGrid.setGlobalWork((long) tilesQ * (stride / ProbeKernels.TILE) * ProbeKernels.GROUP, 1, 1);
                selectGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                plan.withGraph(1).withGridScheduler(scheduler).execute();
                IntStream.range(0, nq).parallel().forEach(i -> {
                    int[] row = new int[want];
                    MemorySegment.copy(probes.getSegment(), ValueLayout.JAVA_INT, (long) i * maxProbes * Integer.BYTES, row, 0, want);
                    probe.setProbes(start + i, row, 0, want);
                });
            }
            if (TRACE) {
                long ms = (System.nanoTime() - t0) / 1_000_000;
                LOG.info(() -> String.format(Locale.ROOT, "GPU probes: %d queries x %d centers, %d probes each, %d ms", n, centers, want, ms));
            }
            return true;
        }

        @Override
        public void close() {
            plan.freeDeviceMemory();
        }
    }

    /** The buffers and the execution plan for one dimension, reused across walks. */
    static final class Walker implements AutoCloseable {
        final int dim;
        final int blockQueries;
        final int maxBlockQueries;
        final int chunkRows;
        final int maxPairs;
        final int maxCandidates;
        Staging staging;
        final FloatArray queries;
        final FloatArray queryNorms;
        final FloatArray rows;
        final IntArray pairFirst;
        final IntArray pairEnd;
        final IntArray pairOrdinal;
        final IntArray pairQuery;
        final IntArray pairOffset;
        final IntArray params;
        final FloatArray candScore;
        final IntArray candOrdinal;
        final IntArray queryOffset;
        final IntArray queryCount;
        final FloatArray topScore;
        final IntArray topOrdinal;
        final IntArray topCount;
        final WorkerGrid scoreGrid;
        final WorkerGrid topkGrid;
        final WorkerGrid normsGrid;
        final IntArray queryParams;
        final GridScheduler scheduler;
        final TornadoExecutionPlan plan;

        Walker(int dim, long budget, int rangeQueries) {
            this.dim = dim;
            long rowBytes = (long) dim * Float.BYTES;
            // the budget split: at most 40% queries (with their top k, and no more than the range has), 10% rows,
            // 15% candidates, 5% pairs; TornadoVM mirrors each buffer in host memory, so these are host bytes too
            long perQuery = rowBytes + (long) CellWalkKernels.MAX_K * 8 + 16;
            maxBlockQueries = (int) Math.max(1024, Math.min(Integer.MAX_VALUE / dim, Math.min(QUERY_BYTES / rowBytes, (long) (budget * 0.40) / perQuery)));
            blockQueries = Math.max(1024, Math.min(maxBlockQueries, rangeQueries));
            chunkRows = (int) Math.max(1024, Math.min(Integer.MAX_VALUE / dim, Math.min(CHUNK_BYTES, (long) (budget * 0.10)) / rowBytes));
            maxCandidates = (int) Math.max(1 << 20, Math.min(MAX_CANDIDATES, (long) (budget * 0.15) / 8));
            maxPairs = (int) Math.max(1 << 16, Math.min(MAX_PAIRS, (long) (budget * 0.05) / 16));
            if (TRACE) {
                LOG.info(() -> String.format(Locale.ROOT, "GPU walk buffers for a %.1f GB budget: %d queries a block, %d rows a chunk, %d pairs and %d candidates a launch",
                        budget / 1e9, blockQueries, chunkRows, maxPairs, maxCandidates));
            }
            int k = CellWalkKernels.MAX_K;
            queries = new FloatArray(blockQueries * dim);
            queryNorms = new FloatArray(blockQueries);
            rows = new FloatArray(chunkRows * dim);
            pairFirst = new IntArray(maxPairs);
            pairEnd = new IntArray(maxPairs);
            pairOrdinal = new IntArray(maxPairs);
            pairQuery = new IntArray(maxPairs);
            pairOffset = new IntArray(maxPairs);
            params = new IntArray(4);
            candScore = new FloatArray(maxCandidates);
            candOrdinal = new IntArray(maxCandidates);
            queryOffset = new IntArray(blockQueries);
            queryCount = new IntArray(blockQueries);
            topScore = new FloatArray(blockQueries * k);
            topOrdinal = new IntArray(blockQueries * k);
            topCount = new IntArray(blockQueries);
            KernelContext ctx = new KernelContext();
            queryParams = new IntArray(4);
            // graph 0, once per query block: the queries and their norms, kept on the device for graph 1
            TaskGraph block = new TaskGraph("block")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, queries, queryParams)
                    .task("norms", CellWalkKernels::norms, ctx, queries, queryParams, queryNorms)
                    .persistOnDevice(queries, queryNorms);
            // graph 1, once per launch: a chunk of rows and the pairs that hit it
            TaskGraph tg = new TaskGraph("walk")
                    .consumeFromDevice("block", queries, queryNorms)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows, pairFirst, pairEnd, pairQuery, pairOrdinal,
                            pairOffset, params, queryOffset, queryCount, topScore, topOrdinal, topCount)
                    .task("score", CellWalkKernels::score, ctx, queries, queryNorms, rows, pairFirst, pairEnd, pairQuery, pairOffset, pairOrdinal,
                            params, candScore, candOrdinal)
                    .task("topk", CellWalkKernels::topk, ctx, candScore, candOrdinal, queryOffset, queryCount, params, topScore,
                            topOrdinal, topCount)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, topScore, topOrdinal, topCount);
            scoreGrid = new WorkerGrid1D(CellWalkKernels.GROUP);
            scoreGrid.setLocalWork(CellWalkKernels.GROUP, 1, 1);
            topkGrid = new WorkerGrid1D(CellWalkKernels.GROUP);
            topkGrid.setLocalWork(CellWalkKernels.GROUP, 1, 1);
            normsGrid = new WorkerGrid1D(CellWalkKernels.GROUP);
            normsGrid.setLocalWork(CellWalkKernels.GROUP, 1, 1);
            scheduler = new GridScheduler();
            scheduler.addWorkerGrid("block.norms", normsGrid);
            scheduler.addWorkerGrid("walk.score", scoreGrid);
            scheduler.addWorkerGrid("walk.topk", topkGrid);
            plan = new TornadoExecutionPlan(block.snapshot(), tg.snapshot());
        }

        boolean walk(CellWalk walk) {
            long t0 = System.nanoTime();
            int n = walk.queryCount();
            int k = walk.topK();
            long[] pairs = walk.pairs();
            int pairCount = walk.pairCount();
            params.set(CellWalkKernels.P_DIM, dim);
            params.set(CellWalkKernels.P_SIMILARITY, similarityCode(walk.similarity()));
            params.set(CellWalkKernels.P_K, k);
            if (staging == null) {
                staging = new Staging(maxPairs, maxCandidates, chunkRows, blockQueries, dim);
            }
            final Staging st = staging;
            int launches = 0;
            long candidates = 0;
            long tPrep = 0, tRows = 0, tExec = 0, tOut = 0, tQ = 0, tWait = 0, kernelNs = 0, writeNs = 0;
            for (int blockStart = 0; blockStart < n; blockStart += blockQueries) {
                final int b0 = blockStart;
                final int bn = Math.min(blockQueries, n - blockStart);
                long tb = System.nanoTime();
                // the block's queries, uploaded once; their squared norms computed on the device
                IntStream.range(0, bn).parallel().forEach(i -> {
                    float[] q = new float[dim];
                    walk.query(b0 + i, q);
                    MemorySegment.copy(q, 0, queries.getSegment(), ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
                });
                queryParams.set(CellWalkKernels.P_DIM, dim);
                normsGrid.setGlobalWork((long) bn * CellWalkKernels.GROUP, 1, 1);
                plan.withGraph(0).withGridScheduler(scheduler).execute();
                for (int i = 0; i < bn; i++) {
                    topCount.set(i, 0);
                }
                tQ += System.nanoTime() - tb;
                // launches, pipelined: while the GPU runs one, a background thread plans the next and reads its rows
                // (from disk, when the scratch region is not cached) into the staging buffers
                final int[] cursor = { 0 };
                java.util.function.Supplier<Staging> prepare = () -> st.prepare(walk, pairs, pairCount, cursor, b0, bn);
                java.util.concurrent.CompletableFuture<Staging> next = java.util.concurrent.CompletableFuture.supplyAsync(prepare);
                while (true) {
                    long tw = System.nanoTime();
                    Staging ready = next.join();
                    tWait += System.nanoTime() - tw;
                    if (ready.tooLarge) {
                        LOG.info(() -> "cell larger than the GPU walk's row chunk; walking this target on the CPU");
                        return false;
                    }
                    if (ready.size == 0) {
                        break;
                    }
                    tPrep += ready.prepNs;
                    tRows += ready.rowsNs;
                    int size = ready.size;
                    long cands = ready.cands;
                    // into TornadoVM's buffers (memory copies), then the staging is free for the next launch
                    MemorySegment.copy(ready.first, 0, pairFirst.getSegment(), ValueLayout.JAVA_INT, 0, size);
                    MemorySegment.copy(ready.end, 0, pairEnd.getSegment(), ValueLayout.JAVA_INT, 0, size);
                    MemorySegment.copy(ready.ordinal, 0, pairOrdinal.getSegment(), ValueLayout.JAVA_INT, 0, size);
                    MemorySegment.copy(ready.query, 0, pairQuery.getSegment(), ValueLayout.JAVA_INT, 0, size);
                    MemorySegment.copy(ready.offset, 0, pairOffset.getSegment(), ValueLayout.JAVA_INT, 0, size);
                    MemorySegment.copy(ready.queryOffset, 0, queryOffset.getSegment(), ValueLayout.JAVA_INT, 0, bn);
                    MemorySegment.copy(ready.queryCount, 0, queryCount.getSegment(), ValueLayout.JAVA_INT, 0, bn);
                    MemorySegment.copy(MemorySegment.ofBuffer(ready.rows), 0, rows.getSegment(), 0, (long) ready.rowsUsed * dim * Float.BYTES);
                    next = java.util.concurrent.CompletableFuture.supplyAsync(prepare);
                    long te = System.nanoTime();
                    scoreGrid.setGlobalWork((long) size * CellWalkKernels.GROUP, 1, 1);
                    topkGrid.setGlobalWork((long) bn * CellWalkKernels.GROUP, 1, 1);
                    var result = TRACE ? plan.withGraph(1).withGridScheduler(scheduler).withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.SILENT).execute()
                            : plan.withGraph(1).withGridScheduler(scheduler).execute();
                    if (TRACE) {
                        kernelNs += result.getProfilerResult().getDeviceKernelTime();
                        writeNs += result.getProfilerResult().getDeviceWriteTime();
                    }
                    tExec += System.nanoTime() - te;
                    launches++;
                    candidates += cands;
                }
                // the block's top k into the walk
                long to = System.nanoTime();
                IntStream.range(0, bn).parallel().forEach(i -> {
                    int count = topCount.get(i);
                    long[] keys = new long[count];
                    for (int e = 0; e < count; e++) {
                        float s = topScore.get(i * k + e);
                        int o = topOrdinal.get(i * k + e);
                        keys[e] = ((long) Float.floatToRawIntBits(s) << 32) | (o & 0xFFFFFFFFL);
                    }
                    walk.setTopK(b0 + i, keys, 0, count);
                });
                tOut += System.nanoTime() - to;
            }
            if (TRACE) {
                long ms = (System.nanoTime() - t0) / 1_000_000;
                int l = launches;
                long c = candidates;
                long[] t = { tQ / 1_000_000, tPrep / 1_000_000, tRows / 1_000_000, tExec / 1_000_000, tOut / 1_000_000, kernelNs / 1_000_000, writeNs / 1_000_000, tWait / 1_000_000 };
                LOG.info(() -> String.format(Locale.ROOT, "GPU cell walk: %d queries, %d pairs, %d candidates, %d launches, %d ms"
                        + " (queries %d; in the background: pairs %d, rows %d; waited %d; execute %d [kernels %d, to device %d]; results %d)",
                        n, pairCount, c, l, ms, t[0], t[1], t[2], t[7], t[3], t[5], t[6], t[4]));
            }
            return true;
        }

        /**
         * One launch planned on the host: its pairs, its queries' candidate offsets, and its distinct cells' rows read
         * into a direct buffer. Filled by a background thread while the GPU runs the launch before.
         */
        static final class Staging {
            final int[] first;
            final int[] end;
            final int[] ordinal;
            final int[] query;
            final int[] offset;
            final int[] pairIndex;
            final int[] pairRow;
            final int[] cells;
            final int[] cellRows;
            final int[] queryOffset;
            final int[] queryCount;
            final int[] perQuery;
            final ByteBuffer rows;
            final int maxPairs;
            final int maxCandidates;
            final int chunkRows;
            final int dim;
            int size;
            int rowsUsed;
            long cands;
            boolean tooLarge;
            long prepNs;
            long rowsNs;

            Staging(int maxPairs, int maxCandidates, int chunkRows, int blockQueries, int dim) {
                this.maxPairs = maxPairs;
                this.maxCandidates = maxCandidates;
                this.chunkRows = chunkRows;
                this.dim = dim;
                first = new int[maxPairs];
                end = new int[maxPairs];
                ordinal = new int[maxPairs];
                query = new int[maxPairs];
                offset = new int[maxPairs];
                pairIndex = new int[maxPairs];
                pairRow = new int[maxPairs];
                cells = new int[chunkRows];
                cellRows = new int[chunkRows];
                queryOffset = new int[blockQueries];
                queryCount = new int[blockQueries];
                perQuery = new int[blockQueries];
                rows = ByteBuffer.allocateDirect(chunkRows * dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            }

            /**
             * Plans the next launch from {@code cursor[0]}: consecutive pairs of the block (in cell order) whose distinct
             * cells fit one chunk of rows, packed; reads those rows. {@code size == 0} when the block has no more.
             */
            Staging prepare(CellWalk walk, long[] pairs, int pairCount, int[] cursor, int b0, int bn) {
                long t0 = System.nanoTime();
                int at = cursor[0];
                size = 0;
                rowsUsed = 0;
                cands = 0;
                tooLarge = false;
                int lastCell = -1;
                int lastRow = 0;
                int distinct = 0;
                java.util.Arrays.fill(perQuery, 0, bn, 0);
                while (at < pairCount && size < maxPairs) {
                    long pair = pairs[at];
                    if (!inBlock(pair, b0, bn)) {
                        at++;
                        continue;
                    }
                    int cell = cellOf(pair);
                    int members = walk.cellEnd(cell) - walk.cellStart(cell);
                    if (members > chunkRows) {
                        tooLarge = true;
                        cursor[0] = at;
                        return this;
                    }
                    boolean newCell = cell != lastCell;
                    if ((newCell && rowsUsed + members > chunkRows) || cands + members > maxCandidates) {
                        break;
                    }
                    if (newCell) {
                        lastCell = cell;
                        lastRow = rowsUsed;
                        cells[distinct] = cell;
                        cellRows[distinct++] = rowsUsed;
                        rowsUsed += members;
                    }
                    pairRow[size] = lastRow;
                    pairIndex[size++] = at;
                    perQuery[(int) pair - b0] += members;
                    cands += members;
                    at++;
                }
                cursor[0] = at;
                // candidate offsets: each query's candidates contiguous, in cell order
                int run = 0;
                for (int i = 0; i < bn; i++) {
                    queryOffset[i] = run;
                    queryCount[i] = perQuery[i];
                    perQuery[i] = run;
                    run += queryCount[i];
                }
                for (int p = 0; p < size; p++) {
                    long pair = pairs[pairIndex[p]];
                    int cell = cellOf(pair);
                    int qi = (int) pair - b0;
                    int f = walk.cellStart(cell);
                    int e = walk.cellEnd(cell);
                    first[p] = pairRow[p];
                    end[p] = pairRow[p] + e - f;
                    ordinal[p] = f;
                    query[p] = qi;
                    offset[p] = perQuery[qi];
                    perQuery[qi] += e - f;
                }
                long t1 = System.nanoTime();
                final int count = distinct;
                IntStream.range(0, count).parallel().forEach(c -> {
                    int cell = cells[c];
                    ByteBuffer dst = rows.duplicate();
                    dst.position(cellRows[c] * dim * Float.BYTES);
                    walk.copyRows(walk.cellStart(cell), walk.cellEnd(cell) - walk.cellStart(cell), dst);
                });
                prepNs = t1 - t0;
                rowsNs = System.nanoTime() - t1;
                return this;
            }
        }

        private static boolean inBlock(long pair, int b0, int bn) {
            int q = (int) pair;
            return q >= b0 && q < b0 + bn;
        }

        private static int cellOf(long pair) {
            return (int) (pair >>> 32);
        }

        static int similarityCode(VectorSimilarityFunction similarity) {
            switch (similarity) {
                case DOT_PRODUCT:
                    return CellWalkKernels.SIM_DOT;
                case EUCLIDEAN:
                    return CellWalkKernels.SIM_EUCLIDEAN;
                default:
                    return CellWalkKernels.SIM_COSINE;
            }
        }

        @Override
        public void close() {
            plan.freeDeviceMemory();
        }
    }
}
