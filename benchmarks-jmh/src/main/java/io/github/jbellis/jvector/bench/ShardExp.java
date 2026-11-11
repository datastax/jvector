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
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Sharded compaction on a real dataset: the jvector side of Cassandra's {@code ShardedVectorCompactionBench}.
 * <p>
 * A Cassandra compaction with UCS sharding writes its output as K sstables, each covering a token range,
 * and every output sstable's vector index is produced by a merge over <em>all</em> source graphs that keeps
 * only the nodes of its range. This program reproduces that shape against raw {@code .fvecs} files:
 * <pre>
 *   build-sources               build S source graphs over consecutive slices of the base (the "sstables")
 *   shard K merge|sharded|rebuild
 *                               produce K output shards, rows assigned to shards by a hash of the row (the token)
 *                               merge:   one OnDiskGraphIndexCompactor per shard over ALL sources, liveNodes = the shard
 *                               sharded: one OnDiskGraphIndexCompactor, one sharded() session, compactShard per shard
 *                               rebuild: one fresh graph per shard from that shard's vectors (graph merge disabled)
 *   sharded K                   same as "shard K sharded"
 *   recall K merge|rebuild|sharded NQ
 *                               search every shard for NQ queries, merge the top-10, recall@10 against the ground truth
 * </pre>
 * Every command prints {@code RESULT} lines: per-shard and total wall and CPU time, and recall. Outputs land
 * under {@code exp.dir}: the sources in {@code sources/}, the shards of a run in {@code out/<mode>-K<K>/}
 * with a {@code .rows} file per shard that maps its ordinals back to base rows for the recall pass.
 * <p>
 * System properties: {@code exp.base}, {@code exp.query}, {@code exp.gt} (the dataset's base and query
 * {@code .fvecs} and ground-truth {@code .ivecs}; required), {@code exp.dir} (output directory, default
 * {@code shardexp}), {@code exp.sources} (2), {@code exp.rows} (all), {@code exp.degree} (32), {@code exp.beam} (100),
 * {@code exp.tag} (suffix on the output directory and the RESULT mode, to keep variants apart).
 * <p>
 * Run from the shaded benchmark jar, for example on tessera-2M with two 1M-vector sources:
 * <pre>
 *   J="java -Xmx24g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
 *      -Dexp.base=base.fvecs -Dexp.query=query.fvecs -Dexp.gt=gt.ivecs \
 *      -cp benchmarks-jmh/target/compactor-benchmark.jar io.github.jbellis.jvector.bench.ShardExp"
 *   $J build-sources
 *   for K in 1 4 16 64; do $J shard $K merge; $J shard $K rebuild; $J sharded $K; done
 *   for K in 1 4 16 64; do for m in merge rebuild sharded; do $J recall $K $m 10000; done; done
 * </pre>
 * See the "Compacting into several outputs (sharding)" section of {@code docs/compaction.md} for the
 * numbers this produced.
 */
public class ShardExp {
    static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final VectorSimilarityFunction VSF = VectorSimilarityFunction.COSINE;
    static final Path DIR = Path.of(System.getProperty("exp.dir", "shardexp"));
    static final int SOURCES = Integer.getInteger("exp.sources", 2);
    static final int DEGREE = Integer.getInteger("exp.degree", 32);
    static final int BEAM = Integer.getInteger("exp.beam", 100);

    /** A path property a command needs: {@code build-sources} and {@code shard} read the base, {@code recall} the queries and the ground truth. */
    static String required(String property) {
        String v = System.getProperty(property);
        if (v == null) throw new IllegalArgumentException("-D" + property + "=<path> is required");
        return v;
    }

    /** .fvecs (4-byte dim header per row), mapped read-only in chunks of whole rows (a mapping holds at most 2 GB). */
    static final class Fvecs {
        final MappedByteBuffer[] chunks; final int dim; final int rows; final long rec; final int rowsPerChunk;
        Fvecs(String path) throws IOException {
            try (var fc = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
                var head = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                if (fc.read(head, 0) != 4) throw new IOException("empty fvecs file " + path);
                dim = head.getInt(0);
                rec = 4L + 4L * dim;
                rows = (int) (fc.size() / rec);
                rowsPerChunk = (int) (Integer.MAX_VALUE / rec);
                chunks = new MappedByteBuffer[(rows + rowsPerChunk - 1) / rowsPerChunk];
                for (int c = 0; c < chunks.length; c++) {
                    long from = (long) c * rowsPerChunk * rec;
                    long to = Math.min(fc.size(), from + (long) rowsPerChunk * rec);
                    chunks[c] = fc.map(FileChannel.MapMode.READ_ONLY, from, to - from);
                    chunks[c].order(ByteOrder.LITTLE_ENDIAN);
                }
            }
        }
        float[] row(int r) {
            float[] a = new float[dim];
            var b = chunks[r / rowsPerChunk];
            int at = (int) ((r % rowsPerChunk) * rec) + 4;
            for (int i = 0; i < dim; i++) a[i] = b.getFloat(at + 4 * i);   // absolute reads: thread-safe
            return a;
        }
    }

    /** The rows {@code rows[i]} of a mapped file as a vector source with ordinals 0..n-1. */
    static final class RowsRavv implements RandomAccessVectorValues {
        final Fvecs f; final int[] rows;
        RowsRavv(Fvecs f, int[] rows) { this.f = f; this.rows = rows; }
        public int size() { return rows.length; }
        public int dimension() { return f.dim; }
        public VectorFloat<?> getVector(int o) { return VTS.createFloatVector(f.row(rows[o])); }
        public boolean isValueShared() { return false; }
        public RandomAccessVectorValues copy() { return this; }
    }

    static int rowsTotal(Fvecs f) { return Integer.getInteger("exp.rows", f.rows); }
    static int perSource(Fvecs f) { return rowsTotal(f) / SOURCES; }

    /** UCS token-range shard of a row: a 64-bit hash of the key, split into K equal ranges. */
    static int shardOf(int row, int k) {
        long z = row * 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z ^= z >>> 31;
        return (int) Math.multiplyHigh(z >>> 1, (long) k << 1);   // in [0, k)
    }

    static long cpuNs() {
        return ((com.sun.management.OperatingSystemMXBean) java.lang.management.ManagementFactory.getOperatingSystemMXBean()).getProcessCpuTime();
    }

    /** PQ-scored build + FusedPQ write, as CompactorBenchmark builds partitions (and as Cassandra's CompactionGraph rebuilds). */
    static void buildGraph(RandomAccessVectorValues ravv, Path out) throws IOException {
        int dim = ravv.dimension();
        var pq = ProductQuantization.compute(ravv, dim / 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(ravv);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(VSF, pqv);
        try (var builder = new GraphIndexBuilder(bsp, dim, DEGREE, BEAM, 1.2f, 1.2f, true)) {
            var graph = builder.build(ravv);
            Files.deleteIfExists(out);
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, out)
                    .with(new InlineVectors(dim)).with(new FusedPQ(graph.maxDegree(), pq)).build()) {
                var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                suppliers.put(FeatureId.INLINE_VECTORS, o -> new InlineVectors.State(ravv.getVector(o)));
                var view = graph.getView();
                suppliers.put(FeatureId.FUSED_PQ, o -> new FusedPQ.State(view, pqv, o));
                writer.write(suppliers);
            }
        }
    }

    static Path sourcePath(int s) { return DIR.resolve("sources").resolve("source-" + s); }
    static final String TAG = System.getProperty("exp.tag", "");
    static Path shardDir(int k, String mode) { return DIR.resolve("out").resolve(mode + TAG + "-K" + k); }


    static void buildSources() throws IOException {
        var f = new Fvecs(required("exp.base"));
        int per = perSource(f);
        Files.createDirectories(sourcePath(0).getParent());
        for (int s = 0; s < SOURCES; s++) {
            int[] rows = IntStream.range(s * per, (s + 1) * per).toArray();
            long t0 = System.nanoTime(), c0 = cpuNs();
            buildGraph(new RowsRavv(f, rows), sourcePath(s));
            System.out.printf("RESULT build-source s=%d n=%d wall_ms=%d cpu_ms=%d bytes=%d%n", s, per,
                    (System.nanoTime() - t0) / 1_000_000, (cpuNs() - c0) / 1_000_000, Files.size(sourcePath(s)));
        }
    }

    static void writeRows(Path p, int[] rowOf) throws IOException {
        try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(p)))) {
            out.writeInt(rowOf.length);
            for (int r : rowOf) out.writeInt(r);
        }
    }

    static int[] readRows(Path p) throws IOException {
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(p)))) {
            int[] a = new int[in.readInt()];
            for (int i = 0; i < a.length; i++) a[i] = in.readInt();
            return a;
        }
    }

    /**
     * K output shards, a row's shard by {@link #shardOf}. {@code merge}: one compactor per shard over
     * all sources with the shard's nodes live; {@code sharded}: one compactor over every node, one
     * session, {@code compactShard} per shard; {@code rebuild}: a fresh graph from the shard's vectors.
     */
    static void shard(int K, String mode) throws IOException {
        var f = new Fvecs(required("exp.base"));
        int per = perSource(f);
        Path dir = shardDir(K, mode);
        Files.createDirectories(dir);
        boolean merge = mode.equals("merge"), sharded = mode.equals("sharded");
        if (!merge && !sharded && !mode.equals("rebuild")) throw new IllegalArgumentException("mode must be merge, sharded or rebuild");
        List<ReaderSupplier> rss = new ArrayList<>();
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        if (merge || sharded) {
            for (int s = 0; s < SOURCES; s++) {
                rss.add(ReaderSupplierFactory.open(sourcePath(s)));
                graphs.add(OnDiskGraphIndex.load(rss.get(s)));
            }
        }
        OnDiskGraphIndexCompactor.ShardedCompaction session = null;
        if (sharded) {
            List<FixedBitSet> all = new ArrayList<>();
            for (int s = 0; s < SOURCES; s++) {
                var bs = new FixedBitSet(graphs.get(s).getIdUpperBound());
                bs.set(0, per);
                all.add(bs);
            }
            session = new OnDiskGraphIndexCompactor(graphs, all, VSF, null).sharded();
        }
        long wallTotal = 0, cpuTotal = 0;
        for (int kk = 0; kk < K; kk++) {
            final int k = kk;
            Path out = dir.resolve("shard-" + k);
            Files.deleteIfExists(out);
            long t0 = System.nanoTime(), c0 = cpuNs();
            int[] rowOf;
            if (merge || sharded) {
                List<FixedBitSet> live = new ArrayList<>();
                int n = 0;
                for (int s = 0; s < SOURCES; s++) {
                    var bs = new FixedBitSet(graphs.get(s).getIdUpperBound());
                    for (int i = 0; i < per; i++) if (shardOf(s * per + i, K) == k) { bs.set(i); n++; }
                    live.add(bs);
                }
                List<OrdinalMapper> mappers;
                if (sharded) {
                    mappers = session.compactShard(live, out);
                } else {
                    var compactor = new OnDiskGraphIndexCompactor(graphs, live, VSF, null);
                    compactor.compact(out);
                    mappers = compactor.ordinalMappers();
                }
                rowOf = new int[n];
                for (int s = 0; s < SOURCES; s++)
                    for (int i = 0; i < per; i++)
                        if (live.get(s).get(i)) rowOf[mappers.get(s).oldToNew(i)] = s * per + i;
            } else {
                rowOf = IntStream.range(0, SOURCES * per).filter(r -> shardOf(r, K) == k).toArray();
                buildGraph(new RowsRavv(f, rowOf), out);
            }
            long wall = (System.nanoTime() - t0) / 1_000_000, cpu = (cpuNs() - c0) / 1_000_000;
            wallTotal += wall; cpuTotal += cpu;
            writeRows(dir.resolve("shard-" + k + ".rows"), rowOf);
            System.out.printf("RESULT shard mode=%s K=%d k=%d nodes=%d wall_ms=%d cpu_ms=%d%n", mode + TAG, K, k, rowOf.length, wall, cpu);
        }
        for (var g : graphs) g.close();
        for (var r : rss) r.close();
        System.out.printf("RESULT total mode=%s K=%d wall_ms=%d cpu_ms=%d%n", mode + TAG, K, wallTotal, cpuTotal);
    }

    static void recall(int K, String mode, int nq) throws Exception {
        var q = new Fvecs(required("exp.query"));
        int[][] gt;
        try (var in = new DataInputStream(new BufferedInputStream(new FileInputStream(required("exp.gt"))))) {
            gt = new int[nq][];
            for (int i = 0; i < nq; i++) {
                int k = Integer.reverseBytes(in.readInt());
                gt[i] = new int[k];
                for (int j = 0; j < k; j++) gt[i][j] = Integer.reverseBytes(in.readInt());
            }
        }
        Path dir = shardDir(K, mode);
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        int[][] rowOf = new int[K][];
        for (int k = 0; k < K; k++) {
            rss.add(ReaderSupplierFactory.open(dir.resolve("shard-" + k)));
            graphs.add(OnDiskGraphIndex.load(rss.get(k)));
            rowOf[k] = readRows(dir.resolve("shard-" + k + ".rows"));
        }
        ThreadLocal<GraphSearcher[]> searchers = ThreadLocal.withInitial(() -> {
            var a = new GraphSearcher[K];
            for (int k = 0; k < K; k++) { a[k] = new GraphSearcher(graphs.get(k)); a[k].usePruning(false); }
            return a;
        });
        int topK = 10;
        long hits = IntStream.range(0, nq).parallel().mapToLong(i -> {
            var qv = VTS.createFloatVector(q.row(i));
            var ss = searchers.get();
            var all = new ArrayList<SearchResult.NodeScore>();
            int[] owner = new int[K * topK]; int at = 0;
            List<long[]> cand = new ArrayList<>();
            for (int k = 0; k < K; k++) {
                var view = (ImmutableGraphIndex.ScoringView) ss[k].getView();
                var ssp = new DefaultSearchScoreProvider(view.approximateScoreFunctionFor(qv, VSF), view.rerankerFor(qv, VSF));
                var r = ss[k].search(ssp, topK, topK, 0f, 0f, Bits.ALL);
                for (var ns : r.getNodes()) cand.add(new long[]{Float.floatToIntBits(ns.score), rowOf[k][ns.node]});
            }
            cand.sort((a, b) -> Float.compare(Float.intBitsToFloat((int) b[0]), Float.intBitsToFloat((int) a[0])));
            var truth = new HashSet<Integer>();
            for (int j = 0; j < topK; j++) truth.add(gt[i][j]);
            long h = 0;
            for (int j = 0; j < Math.min(topK, cand.size()); j++) if (truth.contains((int) cand.get(j)[1])) h++;
            return h;
        }).sum();
        System.out.printf("RESULT recall mode=%s K=%d nq=%d recall@10=%.4f%n", mode + TAG, K, nq, hits / (double) (nq * topK));
        for (var g : graphs) g.close();
        for (var r : rss) r.close();
    }

    public static void main(String[] args) throws Exception {
        switch (args[0]) {
            case "build-sources": buildSources(); break;
            case "shard": shard(Integer.parseInt(args[1]), args[2]); break;
            case "sharded": shard(Integer.parseInt(args[1]), "sharded"); break;
            case "recall": recall(Integer.parseInt(args[1]), args[2], Integer.parseInt(args[3])); break;
            default: throw new IllegalArgumentException("unknown command " + args[0] + "; see the class comment");
        }
    }
}
