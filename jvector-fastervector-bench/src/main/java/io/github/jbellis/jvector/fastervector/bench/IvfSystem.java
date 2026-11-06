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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import io.github.jbellis.jvector.fastervector.FvecThreadPool;
import io.github.jbellis.jvector.fastervector.IvfAshIndex;
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
import io.github.jbellis.jvector.fastervector.RandomAccessVectorValuesVecs;
import io.github.jbellis.jvector.fastervector.Vecs;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.ConstructionPoint;
import io.github.jbellis.jvector.fastervector.utils.Array2DVecs;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * One IVF index, built or reopened for a construction point, plus the dataset's queries in
 * the form the searchers take. Owns the index and its build thread pool.
 */
public final class IvfSystem implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(IvfSystem.class);

    /** Resident memory the open added: the index's in-memory footprint. */
    static final String INDEX_RSS_KEY = "construction.index_rss_mb";
    /** Peak resident memory during the build, above the memory before it. */
    static final String BUILD_PEAK_RSS_KEY = "construction.build_peak_rss_mb";

    private final DataSet ds;
    private final ConstructionPoint point;
    private final IvfAshIndex index;
    private final FvecThreadPool buildPool;
    /** Seconds spent in build(); null if the index was reopened. */
    private final Double buildSeconds;
    private final long indexBytes;
    private final int indexFileCount;
    private final Long buildHeapBytes;
    private final Long buildOffHeapBytes;
    private final Memory memory;
    private final float[][] queries;
    private final Vecs queryVecs;

    /**
     * Process resident memory (RSS) around the index, in bytes; each is null where it couldn't be
     * measured (not Linux, or the peak couldn't be reset).
     *
     * @param baselineBytes  RSS just before the index was opened
     * @param indexBytes     RSS the open added: the index's own in-memory footprint
     * @param buildPeakBytes peak RSS during the build, above the RSS before it; null if reopened
     */
    record Memory(Long baselineBytes, Long indexBytes, Long buildPeakBytes) {}

    private IvfSystem(DataSet ds, ConstructionPoint point, IvfAshIndex index, FvecThreadPool buildPool,
                      Double buildSeconds, List<Path> indexFiles, Memory memory) throws IOException {
        this.ds = ds;
        this.point = point;
        this.index = index;
        this.buildPool = buildPool;
        this.buildSeconds = buildSeconds;
        this.memory = memory;
        long bytes = 0;
        for (Path file : indexFiles) {
            bytes += Files.size(file);
        }
        this.indexBytes = bytes;
        this.indexFileCount = indexFiles.size();
        // JVM memory right after the build or open, as the graph runner records it
        try (var diagnostics = new BenchmarkDiagnostics(DiagnosticLevel.NONE)) {
            diagnostics.capturePostPhaseSnapshot("Build");
            var snapshot = diagnostics.getLatestSystemSnapshot();
            this.buildHeapBytes = snapshot == null ? null : snapshot.memoryStats.heapUsed;
            this.buildOffHeapBytes = snapshot == null ? null : snapshot.memoryStats.getTotalOffHeapMemory();
        }
        this.queries = toArrays(ds.getQueryVectors());
        this.queryVecs = new Array2DVecs(queries);
    }

    /**
     * Builds the index for {@code point}, or reopens it if {@code useSavedIndexIfExists} is set
     * and a complete earlier build is on disk. A rebuild first deletes that construction point's
     * own index directories. A built index is closed and reopened, so its resident memory is
     * measured from a fresh open either way, as a server would load it.
     */
    public static IvfSystem buildOrOpen(DataSet ds, IvfConfig config, ConstructionPoint point) throws IOException {
        List<Path> ssdPaths = config.ssdPaths();
        String subfolder = point.subfolder(ds.getName());
        Path manifest = manifestPath(ssdPaths, subfolder);
        List<Path> devPaths = devPaths(ssdPaths, subfolder);
        int threads = config.construction.buildThreads;
        FvecThreadPool pool = threads == 0 ? null : FvecThreadPool.create(threads);
        try {
            boolean saved = Files.exists(manifest) && devPaths.stream().allMatch(Files::exists);
            if (config.construction.useSavedIndexIfExists && saved) {
                log.info("Reopening {} for {}", manifest, point);
                return open(ds, point, manifest, devPaths, pool, null, null, indexFiles(ssdPaths, subfolder));
            }

            for (Path ssd : ssdPaths) {
                deleteRecursively(ssd.resolve(subfolder));
            }
            log.info("Building {} for {} ({} vectors, dim {})",
                     subfolder, point, ds.getBaseRavv().size(), ds.getDimension());
            ResidentMemory.trim();
            var beforeBuild = ResidentMemory.snapshot();
            boolean peakReset = ResidentMemory.resetPeak();
            long start = System.nanoTime();
            IvfAshIndex built = IvfAshIndex.builder()
                    .nlist(point.nlist())
                    .storageFactor(point.storageFactor())
                    .vectors(new RandomAccessVectorValuesVecs(ds.getBaseRavv()))
                    .ssdPaths(ssdPaths)
                    .subfolder(subfolder)
                    .buildThreadPool(pool)
                    .build();
            double seconds = (System.nanoTime() - start) / 1e9;
            var afterBuild = ResidentMemory.snapshot();
            Long buildPeak = peakReset && beforeBuild.isPresent() && afterBuild.isPresent()
                    ? afterBuild.get().peakRssBytes() - beforeBuild.get().rssBytes()
                    : null;
            built.close();
            return open(ds, point, manifest, devPaths, pool, seconds, buildPeak, indexFiles(ssdPaths, subfolder));
        } catch (IOException | RuntimeException e) {
            if (pool != null) {
                pool.close();
            }
            throw e;
        }
    }

    /** Opens the index, measuring the resident memory the open adds. */
    private static IvfSystem open(DataSet ds, ConstructionPoint point, Path manifest, List<Path> devPaths,
                                  FvecThreadPool pool, Double buildSeconds, Long buildPeakBytes,
                                  List<Path> indexFiles) throws IOException {
        ResidentMemory.trim();
        var before = ResidentMemory.snapshot();
        IvfAshIndex index = IvfAshIndex.prepareOpen()
                .manifestPath(manifest.toString())
                .devPaths(devPaths.stream().map(Path::toString).toList())
                .dim(ds.getDimension())
                .buildThreadPool(pool)
                .open();
        var after = ResidentMemory.snapshot();
        Memory memory = new Memory(before.map(ResidentMemory.Snapshot::rssBytes).orElse(null),
                                   before.isPresent() && after.isPresent()
                                           ? after.get().rssBytes() - before.get().rssBytes() : null,
                                   buildPeakBytes);
        return new IvfSystem(ds, point, index, pool, buildSeconds, indexFiles, memory);
    }

    /** Manifest location for a build into {@code ssdPaths}: beside device file 0. */
    static Path manifestPath(List<Path> ssdPaths, String subfolder) {
        return ssdPaths.get(0).resolve(subfolder).resolve("dev0.ash.blk.manifest");
    }

    /** Device file {@code i} is {@code <ssdPaths[i]>/<subfolder>/dev<i>.ash.blk}, as IvfAshIndex builds it. */
    static List<Path> devPaths(List<Path> ssdPaths, String subfolder) {
        List<Path> paths = new ArrayList<>(ssdPaths.size());
        for (int i = 0; i < ssdPaths.size(); i++) {
            paths.add(ssdPaths.get(i).resolve(subfolder).resolve("dev" + i + ".ash.blk"));
        }
        return paths;
    }

    /** Total size of the regular files in each SSD path's index directory. */
    static long indexBytes(List<Path> ssdPaths, String subfolder) throws IOException {
        long total = 0;
        for (Path file : indexFiles(ssdPaths, subfolder)) {
            total += Files.size(file);
        }
        return total;
    }

    /** The regular files in each SSD path's index directory. */
    static List<Path> indexFiles(List<Path> ssdPaths, String subfolder) throws IOException {
        List<Path> files = new ArrayList<>();
        for (Path ssd : ssdPaths) {
            Path dir = ssd.resolve(subfolder);
            if (!Files.isDirectory(dir)) {
                continue;
            }
            try (Stream<Path> walk = Files.walk(dir)) {
                walk.filter(Files::isRegularFile).forEach(files::add);
            }
        }
        return files;
    }

    private static void deleteRecursively(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(dir)) {
            for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) {
                Files.delete(p);
            }
        }
    }

    private static float[][] toArrays(List<VectorFloat<?>> vectors) {
        float[][] out = new float[vectors.size()][];
        for (int i = 0; i < out.length; i++) {
            VectorFloat<?> v = vectors.get(i);
            out[i] = new float[v.length()];
            for (int j = 0; j < v.length(); j++) {
                out[i][j] = v.get(j);
            }
        }
        return out;
    }

    /** A new searcher; the caller closes it. {@code pool} null means one thread. */
    public IvfAshIndexSearcher searcher(int topK, int nprobe, FvecThreadPool pool) {
        return index.searcherBuilder().topk(topK).nprobe(nprobe).searchThreadPool(pool).build();
    }

    /**
     * Construction and disk metrics, keyed like the graph benchmarks' so run-config.yml's
     * {@code construction} and {@code disk} selections apply. Build time is absent for a reopened index.
     */
    public List<Metric> constructionMetrics() {
        List<Metric> out = new ArrayList<>();
        double mb = indexBytes / 1024.0 / 1024.0;
        if (buildSeconds != null) {
            out.add(Metric.of("construction.index_build_time_s", "Index build time (s)", ".2f", buildSeconds));
        }
        out.add(Metric.of("construction.index_file_size_mb", "Index file size (MB)", ".2f", mb));
        out.add(Metric.of("search.disk.total_file_size_mb", "Total file size (MB)", ".1f", mb));
        out.add(Metric.of("search.disk.file_count", "File count", ".0f", indexFileCount));
        if (buildHeapBytes != null) {
            out.add(Metric.of("construction.build_heap_mb", "Build heap used (MB)", ".1f", buildHeapBytes / 1024.0 / 1024.0));
            out.add(Metric.of("construction.build_offheap_mb", "Build off-heap (MB)", ".1f",
                              buildOffHeapBytes / 1024.0 / 1024.0));
        }
        if (memory.indexBytes() != null) {
            out.add(Metric.of(INDEX_RSS_KEY, "Index memory (MB)", ".1f", ResidentMemory.mb(memory.indexBytes())));
        }
        if (memory.buildPeakBytes() != null) {
            out.add(Metric.of(BUILD_PEAK_RSS_KEY, "Build peak memory (MB)", ".1f",
                              ResidentMemory.mb(memory.buildPeakBytes())));
        }
        return out;
    }

    /** RSS just before the index was opened, in bytes; null where it couldn't be measured. */
    Long rssBaselineBytes() {
        return memory.baselineBytes();
    }

    public DataSet dataSet() {
        return ds;
    }

    public ConstructionPoint constructionPoint() {
        return point;
    }

    /** True if the index was reopened rather than built in this run. */
    public boolean reopened() {
        return buildSeconds == null;
    }

    /** All queries, in dataset order, as one batch. */
    public Vecs queryVecs() {
        return queryVecs;
    }

    /** Query {@code i} as a float array; don't modify it. */
    public float[] query(int i) {
        return queries[i];
    }

    public int queryCount() {
        return queries.length;
    }

    @Override
    public void close() {
        try {
            index.close();
        } finally {
            if (buildPool != null) {
                buildPool.close();
            }
        }
    }
}
