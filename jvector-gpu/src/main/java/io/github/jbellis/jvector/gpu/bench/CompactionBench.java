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
package io.github.jbellis.jvector.gpu.bench;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.gpu.GpuCompaction;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Compaction of on-disk segments: JVector's {@link OnDiskGraphIndexCompactor} against {@link GpuCompaction}, which
 * rebuilds the merged graph on the GPU from the segments' vectors. The dataset's base vectors are split into
 * contiguous segments, each built by JVector on the CPU (M=32, efConstruction 100, overflow 1.2, alpha 1.2,
 * hierarchy) and written once to the work directory; then both compactions run on them, and both outputs are
 * searched the same way (recall@10 and mean latency per beam, exact scores).
 * <p>
 * Usage: {@code CompactionBench <dataset> [segments=4] [workDir=target/compaction]}, the dataset by name as JVector's
 * {@code Bench} takes it (downloaded on first use).
 */
public final class CompactionBench {
    private static final int[] BEAMS = { 20, 40, 80, 160 };

    private CompactionBench() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: CompactionBench <dataset> [segments=4] [workDir=target/compaction]");
            System.exit(1);
        }
        DataSet ds = DataSets.loadDataSet(args[0]).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + args[0])).getDataSet();
        int parts = args.length > 1 ? Integer.parseInt(args[1]) : 4;
        Path work = Path.of(args.length > 2 ? args[2] : "target/compaction").resolve(ds.getName());
        Files.createDirectories(work);
        int dim = ds.getDimension();
        VectorSimilarityFunction sim = ds.getSimilarityFunction();
        List<VectorFloat<?>> base = ds.getBaseVectors();
        int n = base.size();
        System.out.printf("%s: %,d x %d vectors, %s, %d segments in %s%n%n", ds.getName(), n, dim, sim, parts, work);

        // 1. segments, built by JVector on the CPU (kept for the next run)
        List<Path> segments = new ArrayList<>();
        for (int p = 0; p < parts; p++) {
            int from = (int) ((long) n * p / parts);
            int to = (int) ((long) n * (p + 1) / parts);
            Path path = work.resolve("segment-" + p + "-of-" + parts + ".idx");
            segments.add(path);
            if (!Files.exists(path)) {
                var ravv = new ListRandomAccessVectorValues(base.subList(from, to), dim);
                long t = System.nanoTime();
                var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, sim), dim, 32, 100, 1.2f, 1.2f, true, true);
                OnDiskGraphIndex.write(builder.build(ravv), ravv, path);
                System.out.printf("segment %d: %,d vectors built on the CPU in %.1f s%n", p, to - from, (System.nanoTime() - t) / 1e9);
            }
        }

        // 2. JVector's compactor
        // Each compactor assigns the new ordinals; its ordinalMappers() map the results back to the dataset's.
        Path cpuOut = work.resolve("compacted-cpu.idx");
        int[] cpuToDataset;
        long t0 = System.nanoTime();
        try (var opened = Segments.open(segments)) {
            var compactor = new OnDiskGraphIndexCompactor(opened.graphs, opened.live, sim, null);
            compactor.compact(cpuOut);
            cpuToDataset = opened.toDataset(compactor.ordinalMappers());
        }
        double cpuSeconds = (System.nanoTime() - t0) / 1e9;

        // 3. the GPU rebuild
        Path gpuOut = work.resolve("compacted-gpu.idx");
        int[] gpuToDataset;
        long t1 = System.nanoTime();
        try (var opened = Segments.open(segments)) {
            var compaction = new GpuCompaction(opened.graphs, opened.live, sim);
            compaction.compact(gpuOut);
            gpuToDataset = opened.toDataset(compaction.ordinalMappers());
        }
        double gpuSeconds = (System.nanoTime() - t1) / 1e9;
        System.out.printf("compaction of %d segments (%,d vectors): JVector %.1f s, GPU %.1f s (%.1fx)%n%n", parts, n, cpuSeconds, gpuSeconds,
                cpuSeconds / gpuSeconds);

        // 4. the same search on both outputs, each with its own vectors
        List<VectorFloat<?>> queries = ds.getQueryVectors();
        List<? extends List<Integer>> gt = ds.getGroundTruth();
        System.out.printf("%-8s %5s %10s %10s%n", "output", "beam", "recall@10", "mean us");
        try (ReaderSupplier cpuRs = ReaderSupplierFactory.open(cpuOut); ReaderSupplier gpuRs = ReaderSupplierFactory.open(gpuOut);
             var cpuGraph = OnDiskGraphIndex.load(cpuRs); var gpuGraph = OnDiskGraphIndex.load(gpuRs);
             var cpuView = cpuGraph.getView(); var gpuView = gpuGraph.getView()) {
            GraphSearcher[] searchers = { new GraphSearcher(cpuGraph), new GraphSearcher(gpuGraph) };
            RandomAccessVectorValues[] vectors = { cpuView, gpuView };
            int[][] toDataset = { cpuToDataset, gpuToDataset };
            String[] names = { "JVector", "GPU" };
            for (int beam : BEAMS) {
                for (int g = 0; g < 2; g++) {
                    search(searchers[g], vectors[g], toDataset[g], queries, gt, sim, beam); // warm-up pass
                    double[] r = search(searchers[g], vectors[g], toDataset[g], queries, gt, sim, beam);
                    System.out.printf("%-8s %5d %10.4f %10.1f%n", names[g], beam, r[0], r[1]);
                }
            }
        }
    }

    /** {recall@10, mean microseconds a query} of a pass over the queries. */
    private static double[] search(GraphSearcher searcher, RandomAccessVectorValues vectors, int[] toDataset, List<VectorFloat<?>> queries,
                                   List<? extends List<Integer>> gt, VectorSimilarityFunction sim, int beam) {
        long hits = 0;
        long nanos = 0;
        for (int q = 0; q < queries.size(); q++) {
            long t = System.nanoTime();
            var result = searcher.search(DefaultSearchScoreProvider.exact(queries.get(q), sim, vectors), 10, beam, 0.0f, 0.0f, Bits.ALL);
            nanos += System.nanoTime() - t;
            var truth = new HashSet<>(gt.get(q).subList(0, 10));
            for (var ns : result.getNodes()) {
                if (truth.contains(toDataset[ns.node])) {
                    hits++;
                }
            }
        }
        return new double[] { hits / (10.0 * queries.size()), nanos / 1e3 / queries.size() };
    }

    /** Opened segments with the compactors' inputs: every node live; dataset ordinals offset by the segments before. */
    private static final class Segments implements AutoCloseable {
        final List<ReaderSupplier> suppliers = new ArrayList<>();
        final List<OnDiskGraphIndex> graphs = new ArrayList<>();
        final List<FixedBitSet> live = new ArrayList<>();
        final List<Integer> offsets = new ArrayList<>();

        /** For each new ordinal of a compacted output, the dataset ordinal of its vector. */
        int[] toDataset(List<OrdinalMapper> mappers) {
            int total = graphs.stream().mapToInt(g -> g.size(0)).sum();
            int[] out = new int[total];
            for (int s = 0; s < graphs.size(); s++) {
                for (int node = 0; node < graphs.get(s).size(0); node++) {
                    out[mappers.get(s).oldToNew(node)] = offsets.get(s) + node;
                }
            }
            return out;
        }

        static Segments open(List<Path> paths) throws Exception {
            var s = new Segments();
            int offset = 0;
            for (Path path : paths) {
                var rs = ReaderSupplierFactory.open(path);
                s.suppliers.add(rs);
                var g = OnDiskGraphIndex.load(rs);
                s.graphs.add(g);
                int size = g.size(0);
                var bits = new FixedBitSet(size);
                bits.set(0, size);
                s.live.add(bits);
                s.offsets.add(offset);
                offset += size;
            }
            return s;
        }

        @Override
        public void close() throws Exception {
            for (var g : graphs) {
                g.close();
            }
            for (var rs : suppliers) {
                rs.close();
            }
        }
    }
}
