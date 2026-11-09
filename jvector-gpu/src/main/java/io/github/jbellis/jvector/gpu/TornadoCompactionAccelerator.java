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
    /** Candidate scores per launch. */
    static final int MAX_CANDIDATES = 1 << 27;

    /** Whether the probe step runs on the GPU too ({@code -Djvector.gpu.probe}, default true). */
    static final boolean PROBE = Boolean.parseBoolean(System.getProperty("jvector.gpu.probe", "true"));

    private Walker walker;
    private Prober prober;

    /** Bytes of the score matrix of one block of probe queries ({@code -Djvector.gpu.probe.scoresMB}, default 1024). */
    static final long SCORE_BYTES = Long.getLong("jvector.gpu.probe.scoresMB", 1024) << 20;

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
        if (walker == null || walker.dim != dim) {
            if (walker != null) {
                walker.close();
            }
            walker = new Walker(dim);
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
        if (probe.maxProbes() > ProbeKernels.MAX_PROBES || probe.dimension() > CellWalkKernels.MAX_DIM) {
            return false;
        }
        int dim = probe.dimension();
        int centers = probe.centerCount();
        if (prober == null || prober.dim != dim || prober.centers != centers || prober.maxProbes != probe.maxProbes()) {
            if (prober != null) {
                prober.close();
            }
            prober = new Prober(dim, centers, probe.maxProbes());
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

        Prober(int dim, int centers, int maxProbes) {
            this.dim = dim;
            this.centers = centers;
            this.maxProbes = maxProbes;
            stride = (centers + ProbeKernels.TILE - 1) / ProbeKernels.TILE * ProbeKernels.TILE;
            int byScores = (int) Math.min(Integer.MAX_VALUE / stride, SCORE_BYTES / ((long) stride * Float.BYTES));
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
        final int chunkRows;
        final FloatArray queries;
        final FloatArray queryNorms;
        final FloatArray rows;
        final IntArray pairFirst;
        final IntArray pairEnd;
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

        Walker(int dim) {
            this.dim = dim;
            long rowBytes = (long) dim * Float.BYTES;
            blockQueries = (int) Math.max(1024, Math.min(Integer.MAX_VALUE / dim, QUERY_BYTES / rowBytes));
            chunkRows = (int) Math.max(1024, Math.min(Integer.MAX_VALUE / dim, CHUNK_BYTES / rowBytes));
            int k = CellWalkKernels.MAX_K;
            queries = new FloatArray(blockQueries * dim);
            queryNorms = new FloatArray(blockQueries);
            rows = new FloatArray(chunkRows * dim);
            pairFirst = new IntArray(MAX_PAIRS);
            pairEnd = new IntArray(MAX_PAIRS);
            pairQuery = new IntArray(MAX_PAIRS);
            pairOffset = new IntArray(MAX_PAIRS);
            params = new IntArray(4);
            candScore = new FloatArray(MAX_CANDIDATES);
            candOrdinal = new IntArray(MAX_CANDIDATES);
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
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows, pairFirst, pairEnd, pairQuery,
                            pairOffset, params, queryOffset, queryCount, topScore, topOrdinal, topCount)
                    .task("score", CellWalkKernels::score, ctx, queries, queryNorms, rows, pairFirst, pairEnd, pairQuery, pairOffset,
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
            ByteBuffer rowBuffer = rows.getSegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            int[] launchPairs = new int[MAX_PAIRS];
            int[] perQuery = new int[blockQueries];
            int launches = 0;
            long candidates = 0;
            long tPrep = 0, tRows = 0, tExec = 0, tOut = 0, tQ = 0, kernelNs = 0, writeNs = 0;
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
                // launches: consecutive pairs of this block (in cell order) whose cells fit one chunk of rows
                int at = 0;
                while (true) {
                    long tl = System.nanoTime();
                    while (at < pairCount && !inBlock(pairs[at], b0, bn)) {
                        at++;
                    }
                    if (at >= pairCount) {
                        break;
                    }
                    int firstCell = cellOf(pairs[at]);
                    int row0 = walk.cellStart(firstCell);
                    if (walk.cellEnd(firstCell) - row0 > chunkRows) {
                        LOG.info(() -> "cell larger than the GPU walk's row chunk; walking this target on the CPU");
                        return false;
                    }
                    int rowEnd = row0;
                    int size = 0;
                    long cands = 0;
                    java.util.Arrays.fill(perQuery, 0, bn, 0);
                    while (at < pairCount && size < MAX_PAIRS) {
                        long pair = pairs[at];
                        if (!inBlock(pair, b0, bn)) {
                            at++;
                            continue;
                        }
                        int cell = cellOf(pair);
                        int end = walk.cellEnd(cell);
                        int members = end - walk.cellStart(cell);
                        if (end - row0 > chunkRows || cands + members > MAX_CANDIDATES) {
                            break;
                        }
                        launchPairs[size++] = at;
                        perQuery[(int) pair - b0] += members;
                        cands += members;
                        rowEnd = Math.max(rowEnd, end);
                        at++;
                    }
                    // candidate offsets: each query's candidates contiguous, in cell order
                    int offset = 0;
                    for (int i = 0; i < bn; i++) {
                        queryOffset.set(i, offset);
                        queryCount.set(i, perQuery[i]);
                        perQuery[i] = offset;
                        offset += queryCount.get(i);
                    }
                    for (int p = 0; p < size; p++) {
                        long pair = pairs[launchPairs[p]];
                        int cell = cellOf(pair);
                        int qi = (int) pair - b0;
                        int first = walk.cellStart(cell);
                        int end = walk.cellEnd(cell);
                        pairFirst.set(p, first);
                        pairEnd.set(p, end);
                        pairQuery.set(p, qi);
                        pairOffset.set(p, perQuery[qi]);
                        perQuery[qi] += end - first;
                    }
                    long tr = System.nanoTime();
                    tPrep += tr - tl;
                    rowBuffer.clear();
                    walk.copyRows(row0, rowEnd - row0, rowBuffer);
                    long te = System.nanoTime();
                    tRows += te - tr;
                    params.set(CellWalkKernels.P_ROW0, row0);
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
                long[] t = { tQ / 1_000_000, tPrep / 1_000_000, tRows / 1_000_000, tExec / 1_000_000, tOut / 1_000_000, kernelNs / 1_000_000, writeNs / 1_000_000 };
                LOG.info(() -> String.format(Locale.ROOT, "GPU cell walk: %d queries, %d pairs, %d candidates, %d launches, %d ms"
                        + " (queries %d, pairs %d, rows %d, execute %d [kernels %d, to device %d], results %d)", n, pairCount, c, l, ms, t[0], t[1], t[2], t[3], t[5], t[6], t[4]));
            }
            return true;
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
