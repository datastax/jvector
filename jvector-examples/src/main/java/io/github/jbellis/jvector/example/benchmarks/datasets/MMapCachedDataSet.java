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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.example.util.MappedFvecsRandomAccessVectorValues;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/// A {@link DataSetWrapper} whose base vectors are served from a memory-mapped fvecs file.
///
/// If the origin's base vectors are already a {@link MappedFvecsRandomAccessVectorValues} (as
/// produced by {@link DataSetLoaderSimpleMFD}), they are adopted as-is. Otherwise the base
/// vectors are spilled to `<cacheDir>/<name>-<count>x<dimension>.fvecs` and mapped from there, so
/// the heap no longer holds them. An existing spill file of exactly the expected size is reused
/// rather than rewritten, both because a mapped file cannot be modified on Windows and because
/// the spill is then paid once per dataset rather than once per run; delete the file to force a
/// fresh spill.
///
/// The default cache directory is `$DATASET_CACHE_DIR/mmap` when that variable is set, else
/// `dataset_cache/mmap` relative to the working directory.
///
/// Registered in {@link DataSets#wrapperProviders} under the name {@value #WRAPPER_NAME}.
public final class MMapCachedDataSet implements DataSetWrapper {
    private static final Logger logger = LoggerFactory.getLogger(MMapCachedDataSet.class);
    private static final String ENV_DATASET_CACHE_DIR = "DATASET_CACHE_DIR";

    /// Symbolic wrapper name, as used in `dataset(mmap)` and `wrappers: [mmap]`.
    public static final String WRAPPER_NAME = "mmap";

    /// Provider that memory-maps a dataset's base vectors under {@link #defaultCacheDir()}; idempotent for already-mapped datasets.
    public static final DataSetWrapper.Provider PROVIDER = MMapCachedDataSet::of;

    /// Registry entry for {@value #WRAPPER_NAME}; this wrapper takes no options.
    public static final DataSetWrapper.Factory FACTORY = DataSetWrapper.Factory.optionless(WRAPPER_NAME, PROVIDER);

    private final DataSet origin;
    private final RandomAccessVectorValues baseRavv;

    /// Returns `origin` itself if it is already an {@link MMapCachedDataSet}, otherwise a new mapped view
    /// using {@link #defaultCacheDir()} for any spill file.
    ///
    /// @param origin the dataset whose base vectors should be memory-mapped
    /// @return a memory-mapped view of `origin`
    public static DataSetWrapper of(DataSet origin) {
        if (origin instanceof MMapCachedDataSet) {
            return (MMapCachedDataSet) origin;
        }
        return new MMapCachedDataSet(origin, defaultCacheDir());
    }

    /// @return the directory spill files are written to: `$DATASET_CACHE_DIR/mmap` or `dataset_cache/mmap`
    public static Path defaultCacheDir() {
        String env = System.getenv(ENV_DATASET_CACHE_DIR);
        Path root = (env != null && !env.isEmpty()) ? Paths.get(env) : Paths.get("dataset_cache");
        return root.resolve("mmap");
    }

    /// Creates a mapped view of `origin`, spilling heap-resident base vectors to a file under `cacheDir`.
    ///
    /// @param origin   the dataset to wrap
    /// @param cacheDir where a spill file is written when the origin is not already memory-mapped
    /// @throws UncheckedIOException if the spill file cannot be written or mapped
    public MMapCachedDataSet(DataSet origin, Path cacheDir) {
        this.origin = origin;
        RandomAccessVectorValues source = origin.getBaseRavv();
        if (source instanceof MappedFvecsRandomAccessVectorValues) {
            this.baseRavv = source;
            logger.info("Serving {} base vectors of '{}' from mapped file {}",
                    source.size(), origin.getName(), ((MappedFvecsRandomAccessVectorValues) source).getPath());
            return;
        }
        Path file = cacheDir.resolve(safeFileName(origin.getName()) + "-" + source.size() + "x" + source.dimension() + ".fvecs");
        long expectedBytes = (long) source.size() * (Integer.BYTES + (long) source.dimension() * Float.BYTES);
        long start = System.nanoTime();
        try {
            Files.createDirectories(cacheDir);
            boolean reused = Files.isRegularFile(file) && Files.size(file) == expectedBytes;
            if (!reused) {
                spill(source, file, cacheDir);
            }
            this.baseRavv = new MappedFvecsRandomAccessVectorValues(file);
            logger.info("{} {} base vectors of '{}' {} {} and mapped them in {}s",
                    reused ? "Reused" : "Spilled", source.size(), origin.getName(), reused ? "from" : "to", file,
                    String.format("%.2f", (System.nanoTime() - start) / 1e9));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to spill base vectors of '" + origin.getName() + "' to " + file, e);
        }
    }

    /// Writes `source` to a temporary file beside `file` and moves it into place. The target is never
    /// rewritten in place: a spill file for the same dataset may already be mapped by another wrapper in
    /// this JVM, and Windows refuses to modify a file with a live mapping.
    private static void spill(RandomAccessVectorValues source, Path file, Path cacheDir) throws IOException {
        Path temp = Files.createTempFile(cacheDir, file.getFileName().toString(), ".tmp");
        try {
            SiftLoader.writeFvecs(temp, source);
            try {
                Files.move(temp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    private static String safeFileName(String name) {
        String safe = name.replaceAll("[^A-Za-z0-9._-]", "_");
        return safe.isEmpty() ? "dataset" : safe;
    }

    @Override
    public DataSet getOrigin() {
        return origin;
    }

    @Override
    public RandomAccessVectorValues getBaseRavv() {
        return baseRavv;
    }

    @Override
    public int getDimension() {
        return baseRavv.dimension();
    }
}
