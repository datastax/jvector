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
import java.util.Arrays;
import java.util.List;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.CompactionAccelerators;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * JVector's {@link OnDiskGraphIndexCompactor} with and without a {@code CompactionAccelerator} (the GPU cell walk), on
 * the same segments as {@link CompactionBench} (built there and kept): time, how many base-layer neighbour lists are
 * identical, and recall@10 of the same search on both outputs.
 * Usage: {@code WalkBench <dataset> [segments=4] [workDir=target/compaction] [cpu|gpu|both|cpucpu=both]}; {@code cpucpu}
 * runs the CPU walk twice, the baseline for how much two merges of the same segments differ.
 */
public final class WalkBench {

    private static final int[] BEAMS = { 20, 40, 80, 160 };

    public static void main(String[] args) throws Exception {
        DataSet ds = DataSets.loadDataSet(args[0]).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + args[0])).getDataSet();
        int parts = args.length > 1 ? Integer.parseInt(args[1]) : 4;
        Path work = Path.of(args.length > 2 ? args[2] : "target/compaction").resolve(ds.getName());
        String which = args.length > 3 ? args[3] : "both";
        VectorSimilarityFunction sim = ds.getSimilarityFunction();
        List<Path> segments = new ArrayList<>();
        for (int p = 0; p < parts; p++) {
            Path path = work.resolve("segment-" + p + "-of-" + parts + ".idx");
            if (!Files.exists(path)) {
                throw new IllegalStateException(path + " missing: run CompactionBench first (run.sh compact)");
            }
            segments.add(path);
        }
        System.out.printf("%s: %d segments in %s%n%n", ds.getName(), parts, work);
        String[] names = which.equals("both") ? new String[] { "cpu", "gpu" }
                : which.equals("cpucpu") ? new String[] { "cpu", "cpu2" } : new String[] { which };
        Path[] outs = new Path[names.length];
        int[][] toDataset = new int[names.length][];
        for (int r = 0; r < names.length; r++) {
            if (names[r].equals("gpu")) {
                System.setProperty(CompactionAccelerators.PROPERTY, "tornadovm");
            } else {
                System.clearProperty(CompactionAccelerators.PROPERTY);
            }
            outs[r] = work.resolve("walk-" + names[r] + ".idx");
            long t = System.nanoTime();
            try (var opened = CompactionBench.Segments.open(segments)) {
                var compactor = new OnDiskGraphIndexCompactor(opened.graphs, opened.live, sim, null);
                compactor.compact(outs[r]);
                toDataset[r] = opened.toDataset(compactor.ordinalMappers());
            }
            System.out.printf("compaction (%s walk): %.1f s%n", names[r], (System.nanoTime() - t) / 1e9);
        }
        System.clearProperty(CompactionAccelerators.PROPERTY);
        if (names.length == 2) {
            compareLists(outs[0], outs[1], toDataset[0], toDataset[1]);
        }
        List<VectorFloat<?>> queries = ds.getQueryVectors();
        System.out.printf("%n%-8s %5s %10s %10s%n", "walk", "beam", "recall@10", "mean us");
        for (int beam : BEAMS) {
            for (int r = 0; r < names.length; r++) {
                try (ReaderSupplier rs = ReaderSupplierFactory.open(outs[r]); var graph = OnDiskGraphIndex.load(rs); var view = graph.getView()) {
                    var searcher = new GraphSearcher(graph);
                    CompactionBench.search(searcher, view, toDataset[r], queries, ds.getGroundTruth(), sim, beam);
                    double[] res = CompactionBench.search(searcher, view, toDataset[r], queries, ds.getGroundTruth(), sim, beam);
                    System.out.printf("%-8s %5d %10.4f %10.1f%n", names[r], beam, res[0], res[1]);
                }
            }
        }
    }

    /** Fraction of nodes whose base-layer neighbour sets are identical, and the mean overlap, in dataset ordinals. */
    private static void compareLists(Path a, Path b, int[] aToDataset, int[] bToDataset) throws Exception {
        int[] datasetToB = new int[bToDataset.length];
        for (int i = 0; i < bToDataset.length; i++) {
            datasetToB[bToDataset[i]] = i;
        }
        try (ReaderSupplier ra = ReaderSupplierFactory.open(a); ReaderSupplier rb = ReaderSupplierFactory.open(b);
             var ga = OnDiskGraphIndex.load(ra); var gb = OnDiskGraphIndex.load(rb); var va = ga.getView(); var vb = gb.getView()) {
            int n = aToDataset.length;
            long identical = 0;
            double overlap = 0;
            for (int node = 0; node < n; node++) {
                int[] na = neighbours(va, node, aToDataset);
                int[] nb = neighbours(vb, datasetToB[aToDataset[node]], bToDataset);
                if (Arrays.equals(na, nb)) {
                    identical++;
                }
                int common = 0;
                for (int x : na) {
                    if (Arrays.binarySearch(nb, x) >= 0) {
                        common++;
                    }
                }
                overlap += na.length == 0 ? 1 : (double) common / Math.max(na.length, nb.length);
            }
            System.out.printf("%nbase-layer neighbour lists identical: %.2f%% of %,d nodes; mean overlap %.4f%n",
                    100.0 * identical / n, n, overlap / n);
        }
    }

    private static int[] neighbours(OnDiskGraphIndex.View view, int node, int[] toDataset) {
        var it = view.getNeighborsIterator(0, node);
        int[] out = new int[it.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = toDataset[it.nextInt()];
        }
        Arrays.sort(out);
        return out;
    }
}
