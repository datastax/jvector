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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.CompactionAccelerators;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * The out-of-core runs of the GPU compaction, in two steps so the measured one can run under a memory limit:
 * <ul>
 * <li>{@code compact <segmentDir> <segments> <similarity> cpu|gpu <out>}: opens the segments and nothing else, runs
 * {@link OnDiskGraphIndexCompactor} (with the GPU accelerator for {@code gpu}), and writes the output and, next to it,
 * {@code <out>.map}, the dataset ordinal of every new ordinal;</li>
 * <li>{@code eval <dataset> <out> [<out2>]}: loads the dataset, compares the two outputs' base-layer lists, and
 * measures recall@10 of the same search on each;</li>
 * <li>{@code segments <base.f32> <dim> <segments> <similarity> <outDir> cpu|gpu}: splits a raw float32 file into
 * contiguous segments and builds each (M=32, efConstruction 100, alpha 1.2, hierarchy), on the GPU for {@code gpu};</li>
 * <li>{@code evalraw <dir> <dim> <similarity> <out> [<out2>]}: as {@code eval}, with the first 1,000 queries of
 * {@code dir/query.f32} and the ground truth in {@code dir/gt_ip_100.ivecs}.</li>
 * </ul>
 */
public final class OocBench {

    private static final int[] BEAMS = { 20, 40, 80, 160 };

    public static void main(String[] args) throws Exception {
        if (args.length >= 6 && args[0].equals("compact")) {
            compact(Path.of(args[1]), Integer.parseInt(args[2]), VectorSimilarityFunction.valueOf(args[3]), args[4], Path.of(args[5]));
        } else if (args.length >= 7 && args[0].equals("segments")) {
            segments(Path.of(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]), VectorSimilarityFunction.valueOf(args[4]),
                    Path.of(args[5]), args[6]);
        } else if (args.length >= 5 && args[0].equals("evalraw")) {
            List<Path> outs = new ArrayList<>();
            for (int i = 4; i < args.length; i++) {
                outs.add(Path.of(args[i]));
            }
            evalRaw(Path.of(args[1]), Integer.parseInt(args[2]), VectorSimilarityFunction.valueOf(args[3]), outs);
        } else if (args.length >= 3 && args[0].equals("eval")) {
            List<Path> outs = new ArrayList<>();
            for (int i = 2; i < args.length; i++) {
                outs.add(Path.of(args[i]));
            }
            eval(args[1], outs);
        } else {
            System.err.println("usage: OocBench compact <segmentDir> <segments> <similarity> cpu|gpu <out> | eval <dataset> <out> [<out2>]"
                    + " | segments <base.f32> <dim> <segments> <similarity> <outDir> cpu|gpu | evalraw <dir> <dim> <similarity> <out> [<out2>]");
            System.exit(1);
        }
    }

    static void compact(Path dir, int parts, VectorSimilarityFunction sim, String mode, Path out) throws Exception {
        List<Path> segments = new ArrayList<>();
        for (int p = 0; p < parts; p++) {
            segments.add(dir.resolve("segment-" + p + "-of-" + parts + ".idx"));
        }
        if (mode.equals("gpu")) {
            System.setProperty(CompactionAccelerators.PROPERTY, "tornadovm");
        }
        long t = System.nanoTime();
        int[] toDataset;
        try (var opened = CompactionBench.Segments.open(segments)) {
            var compactor = new OnDiskGraphIndexCompactor(opened.graphs, opened.live, sim, null);
            compactor.compact(out);
            toDataset = opened.toDataset(compactor.ordinalMappers());
        }
        double seconds = (System.nanoTime() - t) / 1e9;
        try (var o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(mapPath(out))))) {
            o.writeInt(toDataset.length);
            for (int v : toDataset) {
                o.writeInt(v);
            }
        }
        Runtime rt = Runtime.getRuntime();
        System.out.printf("compaction (%s, %d segments, -Xmx %.1f GB): %.1f s%n", mode, parts, rt.maxMemory() / 1e9, seconds);
    }

    static void eval(String dataset, List<Path> outs) throws Exception {
        DataSet ds = DataSets.loadDataSet(dataset).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + dataset)).getDataSet();
        int[][] maps = new int[outs.size()][];
        for (int r = 0; r < outs.size(); r++) {
            maps[r] = readMap(outs.get(r));
        }
        if (outs.size() == 2) {
            WalkBench.compareLists(outs.get(0), outs.get(1), maps[0], maps[1]);
        }
        System.out.printf("%n%-28s %5s %10s %10s%n", "output", "beam", "recall@10", "mean us");
        for (int beam : BEAMS) {
            for (int r = 0; r < outs.size(); r++) {
                try (ReaderSupplier rs = ReaderSupplierFactory.open(outs.get(r)); var graph = OnDiskGraphIndex.load(rs); var view = graph.getView()) {
                    var searcher = new GraphSearcher(graph);
                    CompactionBench.search(searcher, view, maps[r], ds.getQueryVectors(), ds.getGroundTruth(), ds.getSimilarityFunction(), beam);
                    double[] res = CompactionBench.search(searcher, view, maps[r], ds.getQueryVectors(), ds.getGroundTruth(), ds.getSimilarityFunction(), beam);
                    System.out.printf("%-28s %5d %10.4f %10.1f%n", outs.get(r).getFileName(), beam, res[0], res[1]);
                }
            }
        }
    }

    static void segments(Path base, int dim, int parts, VectorSimilarityFunction sim, Path outDir, String mode) throws Exception {
        Files.createDirectories(outDir);
        try (var all = new io.github.jbellis.jvector.gpu.MappedVectors(base, dim, -1)) {
            int n = all.size();
            for (int p = 0; p < parts; p++) {
                int from = (int) ((long) n * p / parts);
                int to = (int) ((long) n * (p + 1) / parts);
                Path path = outDir.resolve("segment-" + p + "-of-" + parts + ".idx");
                if (Files.exists(path)) {
                    continue;
                }
                var slice = new Slice(all, from, to - from);
                long t = System.nanoTime();
                var builder = new io.github.jbellis.jvector.graph.GraphIndexBuilder(slice, sim, 32, 100, 1.2f, 1.2f, true);
                var accelerator = new io.github.jbellis.jvector.gpu.TornadoGraphBuildAccelerator();
                if (mode.equals("gpu") && accelerator.supports(builder, slice, sim)) {
                    accelerator.build(builder, slice, sim);
                    builder.cleanup();
                } else {
                    for (int i = 0; i < slice.size(); i++) {
                        builder.addGraphNode(i, slice.getVector(i));
                    }
                    builder.cleanup();
                }
                OnDiskGraphIndex.write(builder.getGraph(), slice, path);
                System.out.printf("segment %d: %,d vectors built (%s) and written in %.1f s%n", p, to - from, mode, (System.nanoTime() - t) / 1e9);
            }
        }
    }

    static void evalRaw(Path dir, int dim, VectorSimilarityFunction sim, List<Path> outs) throws Exception {
        List<int[]> truth = WikipediaBench.ivecs(dir.resolve("gt_ip_100.ivecs"));
        List<io.github.jbellis.jvector.vector.types.VectorFloat<?>> queries = new ArrayList<>();
        List<List<Integer>> gt = new ArrayList<>();
        try (var q = new io.github.jbellis.jvector.gpu.MappedVectors(dir.resolve("query.f32"), dim, 1000)) {
            for (int i = 0; i < q.size(); i++) {
                queries.add(q.getVector(i));
                List<Integer> row = new ArrayList<>();
                for (int e = 0; e < 10; e++) {
                    row.add(truth.get(i)[e]);
                }
                gt.add(row);
            }
        }
        int[][] maps = new int[outs.size()][];
        for (int r = 0; r < outs.size(); r++) {
            maps[r] = readMap(outs.get(r));
        }
        if (outs.size() == 2) {
            WalkBench.compareLists(outs.get(0), outs.get(1), maps[0], maps[1]);
        }
        System.out.printf("%n%-28s %5s %10s %10s%n", "output", "beam", "recall@10", "mean us");
        for (int beam : BEAMS) {
            for (int r = 0; r < outs.size(); r++) {
                try (ReaderSupplier rs = ReaderSupplierFactory.open(outs.get(r)); var graph = OnDiskGraphIndex.load(rs); var view = graph.getView()) {
                    var searcher = new GraphSearcher(graph);
                    CompactionBench.search(searcher, view, maps[r], queries, gt, sim, beam);
                    double[] res = CompactionBench.search(searcher, view, maps[r], queries, gt, sim, beam);
                    System.out.printf("%-28s %5d %10.4f %10.1f%n", outs.get(r).getFileName(), beam, res[0], res[1]);
                }
            }
        }
    }

    /** Rows {@code [from, from + size)} of a mapped file, as their own vector values. */
    static final class Slice implements io.github.jbellis.jvector.graph.RandomAccessVectorValues {
        private final io.github.jbellis.jvector.gpu.MappedVectors all;
        private final int from;
        private final int size;

        Slice(io.github.jbellis.jvector.gpu.MappedVectors all, int from, int size) {
            this.all = all;
            this.from = from;
            this.size = size;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int dimension() {
            return all.dimension();
        }

        @Override
        public io.github.jbellis.jvector.vector.types.VectorFloat<?> getVector(int node) {
            return all.getVector(from + node);
        }

        @Override
        public boolean isValueShared() {
            return all.isValueShared();
        }

        @Override
        public io.github.jbellis.jvector.graph.RandomAccessVectorValues copy() {
            return this;
        }
    }

    private static Path mapPath(Path out) {
        return out.resolveSibling(out.getFileName() + ".map");
    }

    private static int[] readMap(Path out) throws Exception {
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(mapPath(out))))) {
            int[] map = new int[in.readInt()];
            for (int i = 0; i < map.length; i++) {
                map[i] = in.readInt();
            }
            return map;
        }
    }
}
