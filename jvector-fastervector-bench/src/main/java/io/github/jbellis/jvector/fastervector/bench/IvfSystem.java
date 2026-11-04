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
import java.io.UncheckedIOException;
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

    private final DataSet ds;
    private final ConstructionPoint point;
    private final IvfAshIndex index;
    private final FvecThreadPool buildPool;
    /** Seconds spent in build(); null if the index was reopened. */
    private final Double buildSeconds;
    private final long indexBytes;
    private final float[][] queries;
    private final Vecs queryVecs;

    private IvfSystem(DataSet ds, ConstructionPoint point, IvfAshIndex index, FvecThreadPool buildPool,
                      Double buildSeconds, long indexBytes) {
        this.ds = ds;
        this.point = point;
        this.index = index;
        this.buildPool = buildPool;
        this.buildSeconds = buildSeconds;
        this.indexBytes = indexBytes;
        this.queries = toArrays(ds.getQueryVectors());
        this.queryVecs = new Array2DVecs(queries);
    }

    /**
     * Builds the index for {@code point}, or reopens it if {@code useSavedIndexIfExists} is set
     * and a complete earlier build is on disk. A rebuild first deletes that construction point's
     * own index directories.
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
                IvfAshIndex index = IvfAshIndex.prepareOpen()
                        .manifestPath(manifest.toString())
                        .devPaths(devPaths.stream().map(Path::toString).toList())
                        .dim(ds.getDimension())
                        .buildThreadPool(pool)
                        .open();
                return new IvfSystem(ds, point, index, pool, null, indexBytes(ssdPaths, subfolder));
            }

            for (Path ssd : ssdPaths) {
                deleteRecursively(ssd.resolve(subfolder));
            }
            log.info("Building {} for {} ({} vectors, dim {})",
                     subfolder, point, ds.getBaseRavv().size(), ds.getDimension());
            long start = System.nanoTime();
            IvfAshIndex index = IvfAshIndex.builder()
                    .nlist(point.nlist())
                    .storageFactor(point.storageFactor())
                    .vectors(new RandomAccessVectorValuesVecs(ds.getBaseRavv()))
                    .ssdPaths(ssdPaths)
                    .subfolder(subfolder)
                    .buildThreadPool(pool)
                    .build();
            double seconds = (System.nanoTime() - start) / 1e9;
            return new IvfSystem(ds, point, index, pool, seconds, indexBytes(ssdPaths, subfolder));
        } catch (IOException | RuntimeException e) {
            if (pool != null) {
                pool.close();
            }
            throw e;
        }
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
        for (Path ssd : ssdPaths) {
            Path dir = ssd.resolve(subfolder);
            if (!Files.isDirectory(dir)) {
                continue;
            }
            try (Stream<Path> files = Files.walk(dir)) {
                total += files.filter(Files::isRegularFile).mapToLong(p -> {
                    try {
                        return Files.size(p);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }).sum();
            }
        }
        return total;
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

    /** Construction metrics, keyed like the graph benchmarks'. Build time is absent for a reopened index. */
    public List<Metric> constructionMetrics() {
        List<Metric> out = new ArrayList<>();
        if (buildSeconds != null) {
            out.add(Metric.of("construction.index_build_time_s", "Index build time (s)", ".2f", buildSeconds));
        }
        out.add(Metric.of("construction.index_file_size_mb", "Index file size (MB)", ".2f",
                          indexBytes / 1024.0 / 1024.0));
        return out;
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
