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
    private MmaProber mmaProber;

    /** Exact probes score on tensor cores, FP16 in and FP32 accumulate ({@code -Djvector.gpu.probe.mma}, default true). */
    static final boolean PROBE_MMA = Boolean.parseBoolean(System.getProperty("jvector.gpu.probe.mma", "true"));
    private TwoLevelProber twoLevel;

    /** Probes through coarse lists past MAX_PROBE_CENTERS ({@code -Djvector.gpu.probe.twoLevel}, default true). */
    static final boolean TWO_LEVEL = Boolean.parseBoolean(System.getProperty("jvector.gpu.probe.twoLevel", "true"));
    /** Centers per coarse center ({@code -Djvector.gpu.probe.coarseRatio}, default 32). */
    static final int COARSE_RATIO = Integer.getInteger("jvector.gpu.probe.coarseRatio", 32);
    /** Coarse lists a query scans ({@code -Djvector.gpu.probe.coarseProbes}, default 256). */
    static final int COARSE_PROBES = Math.min(ProbeKernels.MAX_COARSE_PROBES, Integer.getInteger("jvector.gpu.probe.coarseProbes", 256));
    /** Most queries a two-level probe launch takes ({@code -Djvector.gpu.probe.block}, default 16,384). */
    static final int TWO_LEVEL_BLOCK = Integer.getInteger("jvector.gpu.probe.block", 1 << 14);
    /** Coarse lists a query scans for its single nearest center ({@code -Djvector.gpu.probe.assignLists}, default 16). */
    static final int ASSIGN_LISTS = Integer.getInteger("jvector.gpu.probe.assignLists", 16);
    /** Most candidate centers a query scores exactly ({@code -Djvector.gpu.probe.listCap}, default 8192). */
    static final int LIST_CAP = Integer.getInteger("jvector.gpu.probe.listCap", 8192);

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
        if (probe.maxProbes() > ProbeKernels.MAX_PROBES || probe.dimension() > CellWalkKernels.MAX_DIM) {
            return false;
        }
        int dim = probe.dimension();
        int centers = probe.centerCount();
        // exact ranking costs queries x centers x dim, and the centers grow with the merge: past MAX_PROBE_CENTERS
        // (at 10M x 1024, 313k centers made exact GPU probes 1.7x slower than the compactor's beam) the probes go
        // through coarse lists instead, or stay on the CPU
        if (centers > MAX_PROBE_CENTERS) {
            if (!TWO_LEVEL) {
                return false;
            }
            if (twoLevel == null || twoLevel.dim != dim || twoLevel.centers != centers || twoLevel.maxProbes != probe.maxProbes()) {
                if (twoLevel != null) {
                    twoLevel.close();
                }
                long budget = (long) (TornadoGraphBuildAccelerator.deviceMemoryBudget() * PROBE_SHARE);
                if ((long) centers * dim * Float.BYTES > budget / 2) {
                    LOG.info(() -> "the level-1 centers take more than half the GPU probe budget; probing on the CPU");
                    return false;
                }
                twoLevel = new TwoLevelProber(dim, centers, probe.maxProbes(), budget);
            }
            return twoLevel.probe(probe);
        }
        if (PROBE_MMA) {
            if (mmaProber == null || !mmaProber.matches(probe)) {
                if (mmaProber != null) {
                    mmaProber.close();
                }
                mmaProber = new MmaProber(probe, (long) (TornadoGraphBuildAccelerator.deviceMemoryBudget() * PROBE_SHARE));
            }
            return mmaProber.probe(probe);
        }
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

    /** Packs {@code dim} floats into {@code words} ints of two FP16 halves (low = even dimension), zero padded. */
    static void packRow(float[] row, int dim, MemorySegment dst, long firstWord, int words) {
        for (int w = 0; w < words; w++) {
            int d = 2 * w;
            int lo = d < dim ? Float.floatToFloat16(row[d]) & 0xFFFF : 0;
            int hi = d + 1 < dim ? Float.floatToFloat16(row[d + 1]) & 0xFFFF : 0;
            dst.setAtIndex(ValueLayout.JAVA_INT, firstWord + w, lo | (hi << 16));
        }
    }

    /**
     * The exact probe step on tensor cores: every query against every center ({@link ProbeKernels#centerScoresMma},
     * FP16 in, FP32 accumulate), the dots turned into ranks, then each query's best centers. The probes only choose
     * cells, which the compactor then scans with exact scores, so FP16 inputs are enough here. The packed centers are
     * uploaded once and stay on the device; a prober is rebuilt when the centers change.
     */
    static final class MmaProber implements AutoCloseable {
        final int dim;
        final int words;
        final int centers;
        final int maxProbes;
        final int stride;
        final int blockQueries;
        final float[] fingerprint;
        final IntArray centersPacked;
        final FloatArray centerNorms;
        final IntArray queriesPacked;
        final IntArray params;
        final FloatArray scores;
        final IntArray probes;
        final WorkerGrid scoreGrid;
        final WorkerGrid rankGrid;
        final WorkerGrid selectGrid;
        final GridScheduler scheduler;
        final TornadoExecutionPlan plan;

        MmaProber(CellProbe probe, long budget) {
            dim = probe.dimension();
            words = (dim + 15) / 16 * 8;
            centers = probe.centerCount();
            maxProbes = probe.maxProbes();
            stride = (centers + ProbeKernels.MMA_TILE - 1) / ProbeKernels.MMA_TILE * ProbeKernels.MMA_TILE;
            long left = budget - (long) centers * words * Integer.BYTES;
            long perQuery = (long) stride * Float.BYTES + (long) words * Integer.BYTES + (long) maxProbes * Integer.BYTES;
            long byScores = Math.min(SCORE_BYTES / ((long) stride * Float.BYTES), left / perQuery);
            blockQueries = (int) Math.max(ProbeKernels.MMA_TILE, Math.min(Integer.MAX_VALUE / stride, byScores) / ProbeKernels.MMA_TILE * ProbeKernels.MMA_TILE);
            centersPacked = new IntArray(centers * words);
            centerNorms = new FloatArray(centers);
            queriesPacked = new IntArray(blockQueries * words);
            params = new IntArray(16);
            scores = new FloatArray(blockQueries * stride);
            probes = new IntArray(blockQueries * maxProbes);
            fingerprint = fingerprint(probe);
            // the centers: packed to FP16 and their norms, on the host, uploaded with the first launch
            int rowsPerTask = Math.max(1, centers / 64 + 1);
            IntStream.range(0, (centers + rowsPerTask - 1) / rowsPerTask).parallel().forEach(t -> {
                int from = t * rowsPerTask;
                int count = Math.min(rowsPerTask, centers - from);
                float[] rows = new float[count * dim];
                float[] row = new float[dim];
                probe.copyCenters(from, count, rows, 0);
                for (int r = 0; r < count; r++) {
                    System.arraycopy(rows, r * dim, row, 0, dim);
                    packRow(row, dim, centersPacked.getSegment(), (long) (from + r) * words, words);
                    double norm = 0;
                    for (float v : row) {
                        norm += (double) v * v;
                    }
                    centerNorms.set(from + r, (float) norm);
                }
            });
            KernelContext ctx = new KernelContext();
            TaskGraph rank = new TaskGraph("mmaProbe")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, centersPacked, centerNorms)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, queriesPacked, params)
                    .task("scores", ProbeKernels::centerScoresMma, ctx, queriesPacked, centersPacked, params, scores)
                    .task("rank", ProbeKernels::rankScores, ctx, scores, centerNorms, params)
                    .task("select", ProbeKernels::selectTop, ctx, scores, params, probes)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, probes);
            if (maxProbes == 1) {
                // the nearest center only (the compactor's cell assignment): an arg-max, not a sorted top k
                rank = new TaskGraph("mmaProbe")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, centersPacked, centerNorms)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, queriesPacked, params)
                        .task("scores", ProbeKernels::centerScoresMma, ctx, queriesPacked, centersPacked, params, scores)
                        .task("rank", ProbeKernels::rankScores, ctx, scores, centerNorms, params)
                        .task("select", ProbeKernels::argmaxRows, ctx, scores, params, probes)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, probes);
            }
            scoreGrid = new WorkerGrid1D(ProbeKernels.MMA_GROUP);
            scoreGrid.setLocalWork(ProbeKernels.MMA_GROUP, 1, 1);
            rankGrid = new WorkerGrid1D(ProbeKernels.GROUP);
            rankGrid.setLocalWork(ProbeKernels.GROUP, 1, 1);
            selectGrid = new WorkerGrid1D(ProbeKernels.GROUP);
            selectGrid.setLocalWork(ProbeKernels.GROUP, 1, 1);
            scheduler = new GridScheduler();
            scheduler.addWorkerGrid("mmaProbe.scores", scoreGrid);
            scheduler.addWorkerGrid("mmaProbe.rank", rankGrid);
            scheduler.addWorkerGrid("mmaProbe.select", selectGrid);
            plan = new TornadoExecutionPlan(rank.snapshot());
        }

        /** A few center values that tell one merge's centers from another's. */
        private static float[] fingerprint(CellProbe probe) {
            int dim = probe.dimension();
            int n = probe.centerCount();
            float[] f = new float[3 * dim];
            probe.copyCenters(0, 1, f, 0);
            probe.copyCenters(n / 2, 1, f, dim);
            probe.copyCenters(n - 1, 1, f, 2 * dim);
            return f;
        }

        boolean matches(CellProbe probe) {
            return probe.dimension() == dim && probe.centerCount() == centers && probe.maxProbes() == maxProbes
                    && java.util.Arrays.equals(fingerprint, fingerprint(probe));
        }

        boolean probe(CellProbe probe) {
            long t0 = System.nanoTime();
            int n = probe.queryCount();
            params.set(ProbeKernels.P_DIM, dim);
            params.set(ProbeKernels.P_WORDS, words);
            params.set(ProbeKernels.P_CENTERS, centers);
            params.set(ProbeKernels.P_SIMILARITY, Walker.similarityCode(probe.similarity()));
            params.set(ProbeKernels.P_PROBES, maxProbes);
            params.set(ProbeKernels.P_STRIDE, stride);
            int want = Math.min(maxProbes, centers);
            long kernelNs = 0;
            for (int b0 = 0; b0 < n; b0 += blockQueries) {
                final int start = b0;
                int nq = Math.min(blockQueries, n - b0);
                IntStream.range(0, nq).parallel().forEach(i -> {
                    float[] q = new float[dim];
                    probe.query(start + i, q);
                    packRow(q, dim, queriesPacked.getSegment(), (long) i * words, words);
                });
                params.set(ProbeKernels.P_QUERIES, nq);
                int tilesQ = (nq + ProbeKernels.MMA_TILE - 1) / ProbeKernels.MMA_TILE;
                scoreGrid.setGlobalWork((long) tilesQ * (stride / ProbeKernels.MMA_TILE) * ProbeKernels.MMA_GROUP, 1, 1);
                rankGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                selectGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                if (TRACE) {
                    var result = plan.withGridScheduler(scheduler).withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.SILENT).execute();
                    kernelNs += result.getProfilerResult().getDeviceKernelTime();
                } else {
                    plan.withGridScheduler(scheduler).execute();
                }
                IntStream.range(0, nq).parallel().forEach(i -> {
                    int[] row = new int[want];
                    MemorySegment.copy(probes.getSegment(), ValueLayout.JAVA_INT, (long) i * maxProbes * Integer.BYTES, row, 0, want);
                    probe.setProbes(start + i, row, 0, want);
                });
            }
            if (TRACE) {
                long ms = (System.nanoTime() - t0) / 1_000_000;
                long kms = kernelNs / 1_000_000;
                LOG.info(() -> String.format(Locale.ROOT, "GPU probes (tensor cores): %d queries x %d centers, %d probes each, %d ms (kernels %d)",
                        n, centers, want, ms, kms));
            }
            return true;
        }

        @Override
        public void close() {
            plan.freeDeviceMemory();
        }
    }

    /**
     * The probe step for merges with many centers, as an inverted-list search over them: every {@link #COARSE_RATIO}-th
     * center is a coarse center and every center joins the list of its nearest coarse center (once per merge, on the
     * GPU); a query scores the coarse centers, scans its {@link #COARSE_PROBES} best lists exactly and keeps the best
     * centers. About {@code coarse + COARSE_PROBES x COARSE_RATIO} center scores a query instead of all of them.
     */
    static final class TwoLevelProber implements AutoCloseable {
        final int dim;
        final int centers;
        final int maxProbes;
        final int coarse;
        final int coarseStride;
        final int blockQueries;
        final FloatArray centerRows;
        final FloatArray centerNorms;
        final FloatArray coarseRows;
        final FloatArray coarseNorms;
        final IntArray uploadParams;
        final IntArray listStart;
        final IntArray listMembers;
        final FloatArray block;
        final IntArray assignParams;
        final FloatArray assignScores;
        final IntArray nearest;
        final IntArray params;
        final IntArray coarseParams;
        final FloatArray coarseScores;
        final IntArray coarseProbes;
        final FloatArray candScore;
        final IntArray candPos;
        final IntArray candCount;
        final IntArray probes;
        final IntArray probeCount;
        final WorkerGrid normsGrid;
        final WorkerGrid coarseNormsGrid;
        final WorkerGrid assignScoreGrid;
        final WorkerGrid assignGrid;
        final WorkerGrid coarseGrid;
        final WorkerGrid coarseSelectGrid;
        final WorkerGrid gatherGrid;
        final WorkerGrid selectGrid;
        final GridScheduler scheduler;
        final TornadoExecutionPlan plan;
        boolean ready;

        TwoLevelProber(int dim, int centers, int maxProbes, long budget) {
            this.dim = dim;
            this.centers = centers;
            this.maxProbes = maxProbes;
            coarse = (centers + COARSE_RATIO - 1) / COARSE_RATIO;
            coarseStride = (coarse + ProbeKernels.TILE - 1) / ProbeKernels.TILE * ProbeKernels.TILE;
            long left = budget - (long) centers * dim * Float.BYTES - (long) coarse * dim * Float.BYTES - (long) centers * 8;
            long perQuery = (long) coarseStride * Float.BYTES * 2 + (long) dim * Float.BYTES + (long) LIST_CAP * 8
                    + (long) COARSE_PROBES * 4 + (long) maxProbes * 4 + 8;
            // at most TWO_LEVEL_BLOCK queries a launch: the score matrices and candidate buffers have resident host copies
            // too (TornadoVM mirrors every buffer), and a 65k-query block made them 7 GB of host memory at 10M
            blockQueries = (int) Math.max(ProbeKernels.TILE, Math.min(TWO_LEVEL_BLOCK, left / perQuery) / ProbeKernels.TILE * ProbeKernels.TILE);
            centerRows = new FloatArray(centers * dim);
            centerNorms = new FloatArray(centers);
            coarseRows = new FloatArray(coarse * dim);
            coarseNorms = new FloatArray(coarse);
            uploadParams = new IntArray(4);
            listStart = new IntArray(coarse + 1);
            listMembers = new IntArray(centers);
            block = new FloatArray(blockQueries * dim);
            assignParams = new IntArray(8);
            assignScores = new FloatArray(blockQueries * coarseStride);
            nearest = new IntArray(blockQueries);
            params = new IntArray(8);
            coarseParams = new IntArray(8);
            coarseScores = new FloatArray(blockQueries * coarseStride);
            coarseProbes = new IntArray(blockQueries * COARSE_PROBES);
            candScore = new FloatArray(blockQueries * LIST_CAP);
            candPos = new IntArray(blockQueries * LIST_CAP);
            candCount = new IntArray(blockQueries);
            probes = new IntArray(blockQueries * maxProbes);
            probeCount = new IntArray(blockQueries);
            KernelContext ctx = new KernelContext();
            // 0: the centers and the coarse centers, with their norms, kept on the device
            TaskGraph upload = new TaskGraph("tlCenters")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, centerRows, coarseRows, uploadParams)
                    .task("norms", CellWalkKernels::norms, ctx, centerRows, uploadParams, centerNorms)
                    .task("coarseNorms", CellWalkKernels::norms, ctx, coarseRows, uploadParams, coarseNorms)
                    .persistOnDevice(centerRows, centerNorms, coarseRows, coarseNorms);
            // 1: a block of centers against the coarse centers: each one's nearest
            TaskGraph assign = new TaskGraph("tlAssign")
                    .consumeFromDevice("tlCenters", coarseRows, coarseNorms)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, block, assignParams)
                    .task("scores", ProbeKernels::centerScores, ctx, block, coarseRows, coarseNorms, assignParams, assignScores)
                    .task("argmax", ProbeKernels::argmaxRows, ctx, assignScores, assignParams, nearest)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, nearest);
            // 2: a block of queries: coarse scores, best lists, exact scores of their members, best centers
            TaskGraph rank = new TaskGraph("tlProbe")
                    .consumeFromDevice("tlCenters", centerRows, centerNorms, coarseRows, coarseNorms)
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, listStart, listMembers)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, block, params, coarseParams)
                    .task("coarse", ProbeKernels::centerScores, ctx, block, coarseRows, coarseNorms, coarseParams, coarseScores)
                    .task("lists", ProbeKernels::selectTop, ctx, coarseScores, coarseParams, coarseProbes)
                    .task("gather", ProbeKernels::gatherScores, ctx, block, centerRows, centerNorms, coarseProbes, listStart, listMembers, params,
                            candScore, candPos, candCount)
                    .task("select", ProbeKernels::selectListed, ctx, candScore, candPos, candCount, params, probes, probeCount)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, probes, probeCount);
            normsGrid = grid((long) centers * ProbeKernels.GROUP);
            coarseNormsGrid = grid((long) coarse * ProbeKernels.GROUP);
            assignScoreGrid = grid(ProbeKernels.GROUP);
            assignGrid = grid(ProbeKernels.GROUP);
            coarseGrid = grid(ProbeKernels.GROUP);
            coarseSelectGrid = grid(ProbeKernels.GROUP);
            gatherGrid = grid(ProbeKernels.GROUP);
            selectGrid = grid(ProbeKernels.GROUP);
            scheduler = new GridScheduler();
            scheduler.addWorkerGrid("tlCenters.norms", normsGrid);
            scheduler.addWorkerGrid("tlCenters.coarseNorms", coarseNormsGrid);
            scheduler.addWorkerGrid("tlAssign.scores", assignScoreGrid);
            scheduler.addWorkerGrid("tlAssign.argmax", assignGrid);
            scheduler.addWorkerGrid("tlProbe.coarse", coarseGrid);
            scheduler.addWorkerGrid("tlProbe.lists", coarseSelectGrid);
            scheduler.addWorkerGrid("tlProbe.gather", gatherGrid);
            scheduler.addWorkerGrid("tlProbe.select", selectGrid);
            plan = new TornadoExecutionPlan(upload.snapshot(), assign.snapshot(), rank.snapshot());
        }

        private static WorkerGrid grid(long global) {
            WorkerGrid g = new WorkerGrid1D(ProbeKernels.GROUP);
            g.setLocalWork(ProbeKernels.GROUP, 1, 1);
            g.setGlobalWork(global, 1, 1);
            return g;
        }

        /** Once per merge: the centers on the device, and the coarse lists. */
        private void prepare(CellProbe probe) {
            long t0 = System.nanoTime();
            int rowsPerTask = Math.max(1, centers / 64 + 1);
            IntStream.range(0, (centers + rowsPerTask - 1) / rowsPerTask).parallel().forEach(t -> {
                int from = t * rowsPerTask;
                int count = Math.min(rowsPerTask, centers - from);
                float[] rows = new float[count * dim];
                probe.copyCenters(from, count, rows, 0);
                MemorySegment.copy(rows, 0, centerRows.getSegment(), ValueLayout.JAVA_FLOAT, (long) from * dim * Float.BYTES, rows.length);
            });
            for (int c = 0; c < coarse; c++) {
                MemorySegment.copy(centerRows.getSegment(), (long) c * COARSE_RATIO * dim * Float.BYTES, coarseRows.getSegment(),
                        (long) c * dim * Float.BYTES, (long) dim * Float.BYTES);
            }
            uploadParams.set(CellWalkKernels.P_DIM, dim);
            plan.withGraph(0).withGridScheduler(scheduler).execute();
            // every center's nearest coarse center, in blocks
            int similarity = Walker.similarityCode(probe.similarity());
            int[] owner = new int[centers];
            assignParams.set(ProbeKernels.P_DIM, dim);
            assignParams.set(ProbeKernels.P_CENTERS, coarse);
            assignParams.set(ProbeKernels.P_SIMILARITY, similarity);
            assignParams.set(ProbeKernels.P_STRIDE, coarseStride);
            for (int b0 = 0; b0 < centers; b0 += blockQueries) {
                int nb = Math.min(blockQueries, centers - b0);
                MemorySegment.copy(centerRows.getSegment(), (long) b0 * dim * Float.BYTES, block.getSegment(), 0, (long) nb * dim * Float.BYTES);
                assignParams.set(ProbeKernels.P_QUERIES, nb);
                int tilesQ = (nb + ProbeKernels.TILE - 1) / ProbeKernels.TILE;
                assignScoreGrid.setGlobalWork((long) tilesQ * (coarseStride / ProbeKernels.TILE) * ProbeKernels.GROUP, 1, 1);
                assignGrid.setGlobalWork((long) nb * ProbeKernels.GROUP, 1, 1);
                plan.withGraph(1).withGridScheduler(scheduler).execute();
                MemorySegment.copy(nearest.getSegment(), ValueLayout.JAVA_INT, 0, owner, b0, nb);
            }
            // the lists, by counting
            int[] start = new int[coarse + 1];
            for (int o : owner) {
                start[o + 1]++;
            }
            for (int c = 0; c < coarse; c++) {
                start[c + 1] += start[c];
            }
            int[] fill = java.util.Arrays.copyOf(start, coarse);
            int[] members = new int[centers];
            for (int c = 0; c < centers; c++) {
                members[fill[owner[c]]++] = c;
            }
            MemorySegment.copy(start, 0, listStart.getSegment(), ValueLayout.JAVA_INT, 0, coarse + 1);
            MemorySegment.copy(members, 0, listMembers.getSegment(), ValueLayout.JAVA_INT, 0, centers);
            ready = true;
            if (TRACE) {
                int largest = 0;
                for (int c = 0; c < coarse; c++) {
                    largest = Math.max(largest, start[c + 1] - start[c]);
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                int l = largest;
                LOG.info(() -> String.format(Locale.ROOT, "GPU coarse lists: %d centers in %d lists (largest %d), %d ms", centers, coarse, l, ms));
            }
        }

        boolean probe(CellProbe probe) {
            long t0 = System.nanoTime();
            if (!ready) {
                prepare(probe);
            }
            int n = probe.queryCount();
            // the nearest center alone (cell assignment) needs few lists; a ranked list of probes needs many
            int lists = Math.min(maxProbes == 1 ? ASSIGN_LISTS : COARSE_PROBES, coarse);
            params.set(ProbeKernels.P_DIM, dim);
            params.set(ProbeKernels.P_CENTERS, coarse);
            params.set(ProbeKernels.P_SIMILARITY, Walker.similarityCode(probe.similarity()));
            params.set(ProbeKernels.P_PROBES, maxProbes);
            params.set(ProbeKernels.P_STRIDE, coarseStride);
            params.set(ProbeKernels.P_COARSE_PROBES, lists);
            params.set(ProbeKernels.P_LIST_CAP, LIST_CAP);
            // the coarse step: queries against the coarse centers, each query's best lists
            coarseParams.set(ProbeKernels.P_DIM, dim);
            coarseParams.set(ProbeKernels.P_CENTERS, coarse);
            coarseParams.set(ProbeKernels.P_SIMILARITY, Walker.similarityCode(probe.similarity()));
            coarseParams.set(ProbeKernels.P_PROBES, lists);
            coarseParams.set(ProbeKernels.P_STRIDE, coarseStride);
            for (int b0 = 0; b0 < n; b0 += blockQueries) {
                final int start = b0;
                int nq = Math.min(blockQueries, n - b0);
                IntStream.range(0, nq).parallel().forEach(i -> {
                    float[] q = new float[dim];
                    probe.query(start + i, q);
                    MemorySegment.copy(q, 0, block.getSegment(), ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
                });
                params.set(ProbeKernels.P_QUERIES, nq);
                coarseParams.set(ProbeKernels.P_QUERIES, nq);
                int tilesQ = (nq + ProbeKernels.TILE - 1) / ProbeKernels.TILE;
                coarseGrid.setGlobalWork((long) tilesQ * (coarseStride / ProbeKernels.TILE) * ProbeKernels.GROUP, 1, 1);
                coarseSelectGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                gatherGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                selectGrid.setGlobalWork((long) nq * ProbeKernels.GROUP, 1, 1);
                plan.withGraph(2).withGridScheduler(scheduler).execute();
                IntStream.range(0, nq).parallel().forEach(i -> {
                    int count = probeCount.get(i);
                    int[] row = new int[count];
                    MemorySegment.copy(probes.getSegment(), ValueLayout.JAVA_INT, (long) i * maxProbes * Integer.BYTES, row, 0, count);
                    probe.setProbes(start + i, row, 0, count);
                });
            }
            if (TRACE) {
                long ms = (System.nanoTime() - t0) / 1_000_000;
                LOG.info(() -> String.format(Locale.ROOT, "GPU two-level probes: %d queries, %d of %d lists each, %d ms", n,
                        lists, coarse, ms));
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
