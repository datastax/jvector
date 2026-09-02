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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.TokenStreamRetrofit;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.work.ProgressTracker;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * The 2 x 2 merge experiment of vector_merge_splat_design.md s8.7 on real index files:
 * residentSearch x candidateScoring, same sources, recall against brute-force ground truth.
 *
 * <pre>
 *   retrofit  src dst                      copy an SAI Terms component, drop its container trailer,
 *                                          append the token stream (with keys): a second-generation source
 *   gt        queries seed gtFile src...    sample queries from the sources, brute-force top-K over all of them
 *   arm       name resident adc threads gtFile outDir src...
 *                                          merge with the flags, then recall@K of the merged graph
 * </pre>
 * Sources may be raw SAI Terms components (the graph header is found by scanning for its magic;
 * the token stream, if any, by the jvector footer inside the container) or retrofitted copies.
 */
public final class MergeExperiment {
    static final int K = 10;
    static final int DEPTH = 100;
    static final int JV_MAGIC = 0xFFFF0D61;
    static final int JV_FOOTER_MAGIC = 0x4a564244;
    static final VectorSimilarityFunction VSF = VectorSimilarityFunction.DOT_PRODUCT;

    public static void main(String[] args) throws Exception {
        switch (args[0]) {
            case "retrofit": retrofit(Path.of(args[1]), Path.of(args[2])); break;
            case "gt": groundTruth(Integer.parseInt(args[1]), Long.parseLong(args[2]), Path.of(args[3]), paths(args, 4)); break;
            case "arm": arm(args[1], Boolean.parseBoolean(args[2]), Boolean.parseBoolean(args[3]), Integer.parseInt(args[4]),
                            Path.of(args[5]), Path.of(args[6]), paths(args, 7)); break;
            default: throw new IllegalArgumentException("mode " + args[0]);
        }
    }

    private static List<Path> paths(String[] args, int from) {
        List<Path> out = new ArrayList<>();
        for (int i = from; i < args.length; i++) out.add(Path.of(args[i]));
        return out;
    }

    // ---- locating the graph inside a file

    static long headerOffset(Path p) throws IOException {
        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ)) {
            ByteBuffer b = ByteBuffer.allocate(4096);
            fc.read(b, 0);
            for (int i = 0; i + 4 <= b.position(); i++) {
                if (b.getInt(i) == JV_MAGIC) return i;
            }
        }
        throw new IllegalStateException("no jvector header in the first 4 KB of " + p);
    }

    /** End of the last jvector footer in the file (the footer magic's end), found in the tail. */
    static long footerEnd(Path p) throws IOException {
        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ)) {
            long size = fc.size();
            int tail = (int) Math.min(size, 1 << 20);
            ByteBuffer b = ByteBuffer.allocate(tail);
            fc.read(b, size - tail);
            for (int i = tail - 4; i >= 0; i--) {
                if (b.getInt(i) == JV_FOOTER_MAGIC) return size - tail + i + 4;
            }
        }
        throw new IllegalStateException("no jvector footer in the tail of " + p);
    }

    static final class Source implements AutoCloseable {
        final ReaderSupplier rs;
        final OnDiskGraphIndex graph;
        final int size;

        Source(Path p) throws IOException {
            long offset = headerOffset(p);
            long end = footerEnd(p);
            rs = ReaderSupplierFactory.open(p);
            boolean footerLast = end == Files.size(p);
            if (footerLast) {
                graph = OnDiskGraphIndex.load(rs, offset);
            } else {
                graph = OnDiskGraphIndex.load(rs, offset, false);
                graph.discoverTokenStream(end);
            }
            size = graph.size(0);
            System.out.printf("source %s: %d nodes, degree %d, dim %d, tokenStream=%s%n", p.getFileName(), size,
                    graph.getDegree(0), graph.getDimension(), graph.tokenStreamSection().isPresent());
        }

        @Override
        public void close() throws IOException {
            graph.close();
            rs.close();
        }
    }

    // ---- retrofit

    static void retrofit(Path src, Path dst) throws IOException {
        long t0 = System.nanoTime();
        Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
        long offset = headerOffset(dst);
        long end = footerEnd(dst);
        try (FileChannel fc = FileChannel.open(dst, StandardOpenOption.WRITE)) {
            fc.truncate(end);
        }
        var r = TokenStreamRetrofit.append(dst, offset, ProgressTracker.PhaseScope.NOOP);
        System.out.printf("retrofit %s -> %s: %d nodes (%d live), %d edges, section %d bytes, %d ms (copy+append %d ms)%n",
                src.getFileName(), dst.getFileName(), r.nodes, r.liveNodes, r.edges, r.sectionBytes, r.millis,
                (System.nanoTime() - t0) / 1_000_000L);
    }

    // ---- ground truth

    static void groundTruth(int queries, long seed, Path gtFile, List<Path> paths) throws Exception {
        List<Source> sources = new ArrayList<>();
        for (Path p : paths) sources.add(new Source(p));
        int[] offset = new int[sources.size() + 1];
        for (int s = 0; s < sources.size(); s++) offset[s + 1] = offset[s] + sources.get(s).size;
        int total = offset[sources.size()];
        var rnd = new Random(seed);
        int[] qids = new int[queries];
        for (int q = 0; q < queries; q++) qids[q] = rnd.nextInt(total);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        int dim = sources.get(0).graph.getDimension();
        VectorFloat<?>[] qv = new VectorFloat<?>[queries];
        for (int q = 0; q < queries; q++) {
            qv[q] = vts.createFloatVector(dim);
            readVector(sources, offset, qids[q], qv[q]);
        }
        long t0 = System.nanoTime();
        // one pass over every node, chunked across the pool; per-chunk top-K per query, merged at the end
        int threads = Math.max(8, Runtime.getRuntime().availableProcessors() - 8);
        float[][] best = new float[queries][K];
        int[][] bestId = new int[queries][K];
        for (float[] b : best) Arrays.fill(b, Float.NEGATIVE_INFINITY);
        for (int[] b : bestId) Arrays.fill(b, -1);
        var pool = new ForkJoinPool(threads);
        try {
            for (int s = 0; s < sources.size(); s++) {
                final Source src = sources.get(s);
                final int base = offset[s];
                int chunks = threads * 4;
                int chunk = (src.size + chunks - 1) / chunks;
                java.util.concurrent.Callable<List<float[][]>> work = () -> IntStream.range(0, chunks).parallel().mapToObj(c -> {
                    int lo = c * chunk, hi = Math.min(src.size, lo + chunk);
                    float[][] sc = new float[queries][K];
                    int[][] id = new int[queries][K];
                    for (float[] b : sc) Arrays.fill(b, Float.NEGATIVE_INFINITY);
                    VectorFloat<?> v = vts.createFloatVector(dim);
                    try (var view = src.graph.getView()) {
                        for (int n = lo; n < hi; n++) {
                            view.getVectorInto(n, v, 0);
                            for (int q = 0; q < queries; q++) {
                                insert(sc[q], id[q], VSF.compare(qv[q], v), base + n);
                            }
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    // pack ids into the scores array's companion via a holder
                    float[][] packed = new float[queries * 2][];
                    for (int q = 0; q < queries; q++) {
                        packed[q] = sc[q];
                        float[] ids = new float[K];
                        for (int k = 0; k < K; k++) ids[k] = Float.intBitsToFloat(id[q][k]);
                        packed[queries + q] = ids;
                    }
                    return packed;
                }).collect(java.util.stream.Collectors.toList());
                List<float[][]> partialScores = pool.submit(work).get();
                for (float[][] packed : partialScores) {
                    for (int q = 0; q < queries; q++) {
                        for (int k = 0; k < K; k++) {
                            int id = Float.floatToRawIntBits(packed[queries + q][k]);
                            if (id >= 0) insert(best[q], bestId[q], packed[q][k], id);
                        }
                    }
                }
            }
        } finally {
            pool.shutdown();
        }
        List<String> lines = new ArrayList<>();
        for (int q = 0; q < queries; q++) {
            StringBuilder sb = new StringBuilder().append(qids[q]);
            for (int k = 0; k < K; k++) sb.append(' ').append(bestId[q][k]);
            lines.add(sb.toString());
        }
        Files.write(gtFile, lines);
        System.out.printf("ground truth: %d queries over %d nodes (%d sources), top-%d, in %d ms -> %s%n",
                queries, total, sources.size(), K, (System.nanoTime() - t0) / 1_000_000L, gtFile);
        for (Source s : sources) s.close();
    }

    /** Keeps the top-K by score, descending, in place. */
    static void insert(float[] scores, int[] ids, float score, int id) {
        if (score <= scores[K - 1]) return;
        int pos = K - 1;
        while (pos > 0 && scores[pos - 1] < score) {
            scores[pos] = scores[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        scores[pos] = score;
        ids[pos] = id;
    }

    /** This process's own block-device traffic so far (Linux /proc/self/io), independent of whatever else shares the array. */
    static String procIo() {
        try {
            long read = 0, write = 0;
            for (String line : Files.readAllLines(Path.of("/proc/self/io"))) {
                if (line.startsWith("read_bytes:")) read = Long.parseLong(line.substring(11).trim());
                if (line.startsWith("write_bytes:")) write = Long.parseLong(line.substring(12).trim());
            }
            return String.format("read=%d MiB write=%d MiB", read >> 20, write >> 20);
        } catch (Exception e) {
            return "unavailable: " + e;
        }
    }

    static void readVector(List<Source> sources, int[] offset, int global, VectorFloat<?> into) throws IOException {
        int s = 0;
        while (global >= offset[s + 1]) s++;
        try (var view = sources.get(s).graph.getView()) {
            view.getVectorInto(global - offset[s], into, 0);
        }
    }

    // ---- one arm

    static void arm(String name, boolean resident, boolean adc, int threads, Path gtFile, Path outDir, List<Path> paths) throws Exception {
        List<Source> sources = new ArrayList<>();
        for (Path p : paths) sources.add(new Source(p));
        int[] offset = new int[sources.size() + 1];
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        for (int s = 0; s < sources.size(); s++) {
            offset[s + 1] = offset[s] + sources.get(s).size;
            graphs.add(sources.get(s).graph);
            remappers.add(new OrdinalMapper.OffsetMapper(offset[s], sources.get(s).size));
            var live = new FixedBitSet(sources.get(s).size);
            live.set(0, sources.get(s).size);
            liveNodes.add(live);
        }
        int total = offset[sources.size()];
        Path out = outDir.resolve("merged-" + name + ".jv");
        Files.deleteIfExists(out);
        var pool = new ForkJoinPool(threads);
        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, VSF,
                io.github.jbellis.jvector.graph.ParallelExecutor.forkJoin(pool), -1);
        compactor.setSimilarityOrdinals(true);
        compactor.setCandidateScoringFromCodes(adc);
        compactor.setResidentSearch(resident);
        System.out.printf("ARM %s: resident=%s adc=%s threads=%d sources=%d nodes=%d%n", name, resident, adc, threads, sources.size(), total);
        System.out.println("PROCIO before compaction: " + procIo());
        long t0 = System.nanoTime();
        compactor.compact(out);
        long compactMs = (System.nanoTime() - t0) / 1_000_000L;
        pool.shutdown();
        System.out.println("PROCIO after compaction: " + procIo());

        // new ordinal -> global id, from the mapping the merge actually used
        var effective = compactor.effectiveRemappers();
        int[] newToGlobal = new int[total];
        Arrays.fill(newToGlobal, -1);
        for (int s = 0; s < sources.size(); s++) {
            for (int old = 0; old < sources.get(s).size; old++) {
                int nw = effective.get(s).oldToNew(old);
                if (nw >= 0 && nw < total) newToGlobal[nw] = offset[s] + old;
            }
        }
        // the sources were released by the compactor's refine step only if enabled; reopen for query vectors
        List<Source> again = new ArrayList<>();
        for (Path p : paths) again.add(new Source(p));
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        int dim = again.get(0).graph.getDimension();
        List<String> gt = Files.readAllLines(gtFile);
        double hits = 0;
        long searchNanos = 0;
        try (var rs = ReaderSupplierFactory.open(out); var merged = OnDiskGraphIndex.load(rs);
             var view = merged.getView(); var searcher = new GraphSearcher.Builder(view).build()) {
            searcher.usePruning(false);
            VectorFloat<?> q = vts.createFloatVector(dim);
            VectorFloat<?> tmp = vts.createFloatVector(dim);
            for (String line : gt) {
                String[] f = line.split(" ");
                int qid = Integer.parseInt(f[0]);
                readVector(again, offset, qid, q);
                ScoreFunction.ExactScoreFunction sf = node -> {
                    view.getVectorInto(node, tmp, 0);
                    return VSF.compare(q, tmp);
                };
                long s0 = System.nanoTime();
                SearchResult r = searcher.search(new DefaultSearchScoreProvider(sf), K, DEPTH, 0f, 0f, Bits.ALL);
                searchNanos += System.nanoTime() - s0;
                java.util.Set<Integer> truth = new java.util.HashSet<>();
                for (int k = 1; k <= K; k++) truth.add(Integer.parseInt(f[k]));
                for (var n : r.getNodes()) {
                    if (truth.contains(newToGlobal[n.node])) hits++;
                }
            }
        }
        double recall = hits / (gt.size() * (double) K);
        System.out.printf("RESULT arm=%s resident=%s adc=%s nodes=%d compactMs=%d recall@%d=%.4f depth=%d queries=%d searchMeanMs=%.3f output=%s%n",
                name, resident, adc, total, compactMs, K, recall, DEPTH, gt.size(), searchNanos / 1e6 / gt.size(), out);
        for (Source s : again) s.close();
    }
}
