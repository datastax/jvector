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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.VectorCompressor;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A deterministic, file-based cache for {@link OnDiskGraphIndex} artifacts.
 *
 * <p>This class does <em>not</em> build indexes. It provides an easy-to-use API for:
 * <ul>
 *   <li>deriving a stable signature (cache key) from build-defining inputs</li>
 *   <li>loading an existing cached index</li>
 *   <li>obtaining a write path for a new index (with tmp naming)</li>
 *   <li>atomically committing the new index into the cache (tmp → final rename)</li>
 *   <li>cleaning up stale tmp files</li>
 * </ul>
 *
 * <p>The cache is a <b>flat directory</b>. Each index is stored as a single file named from its signature:
 * {@code graph_<signature>} (sanitized to be filesystem-safe). When writing, the cache uses a temp filename
 * (e.g. {@code tmp_graph_<signature>}) and commits via rename so partial writes never match cache lookups.</p>
 */
public final class OnDiskGraphIndexCache {

    public enum Overwrite {
        ALLOW,
        DENY
    }

    private final boolean enabled;
    private final Path cacheDir;

    private OnDiskGraphIndexCache(boolean enabled, Path cacheDir) {
        this.enabled = enabled;
        this.cacheDir = cacheDir;
    }

    public static OnDiskGraphIndexCache disabled() {
        return new OnDiskGraphIndexCache(false, null);
    }

    /**
     * Creates an enabled cache rooted at {@code cacheDir}, ensures the directory exists, and cleans up stale temp files.
     */
    public static OnDiskGraphIndexCache initialize(Path cacheDir) throws IOException {
        Files.createDirectories(cacheDir);
        OnDiskGraphIndexCache cache = new OnDiskGraphIndexCache(true, cacheDir);
        cache.cleanupStaleTemps();

        System.out.println(
                "Index caching enabled: " + cacheDir.toAbsolutePath() + ". " +
                        "\nDelete this directory to remove saved indices and free up disk space."
        );

        return cache;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Path cacheDir() {
        return cacheDir;
    }

    /**
     * Deletes any stale temp files (e.g. {@code tmp_graph_*}) left behind by interrupted builds.
     * Safe to call at startup or periodically. No-op when disabled.
     */
    private void cleanupStaleTemps() throws IOException {
        if (!enabled) return;

        try (Stream<Path> s = Files.list(cacheDir)) {
            s.filter(p -> p.getFileName().toString().startsWith("tmp_"))
                    .forEach(p -> {
                        try { Files.deleteIfExists(p); } catch (IOException ignored) {}
                    });
        }
    }

    // ---------------------------
    // Key / Entry
    // ---------------------------

    /**
     * Semantic identifier for a single index file.
     * The cache computes the signature and final filename from this key.
     * The signature must be updated when new parameters / options are added to indices.
     */
    public static final class CacheKey {
        public final String datasetName;
        public final Set<FeatureId> featureSet;
        public final int M;
        public final int efConstruction;
        public final float neighborOverflow;
        public final boolean addHierarchy;
        public final boolean refineFinalGraph;
        public final String compressorId;

        private CacheKey(String datasetName,
                         Set<FeatureId> featureSet,
                         int M,
                         int efConstruction,
                         float neighborOverflow,
                         boolean addHierarchy,
                         boolean refineFinalGraph,
                         String compressorId) {
            this.datasetName = datasetName;
            this.featureSet = featureSet;
            this.M = M;
            this.efConstruction = efConstruction;
            this.neighborOverflow = neighborOverflow;
            this.addHierarchy = addHierarchy;
            this.refineFinalGraph = refineFinalGraph;
            this.compressorId = compressorId;
        }
    }

    /**
     * Convenience factory for a {@link CacheKey}. Uses {@code buildCompressor.toString()} (whitespace removed)
     * as the compressor id, matching existing naming conventions.
     */
    public CacheKey key(String datasetName,
                        Set<FeatureId> featureSet,
                        int M,
                        int efConstruction,
                        float neighborOverflow,
                        boolean addHierarchy,
                        boolean refineFinalGraph,
                        VectorCompressor<?> buildCompressor) {
        Objects.requireNonNull(datasetName, "datasetName");
        Objects.requireNonNull(featureSet, "featureSet");
        Objects.requireNonNull(buildCompressor, "buildCompressor");
        String compressorId = buildCompressor.toString().replaceAll("\\s+", "");
        return new CacheKey(datasetName, featureSet, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, compressorId);
    }

    /**
     * Resolved cache entry: signature + final file path in the cache directory.
     */
    public Entry resolve(CacheKey key) {
        Objects.requireNonNull(key, "key");
        if (!enabled) {
            return Entry.disabled();
        }
        return Entry.compute(cacheDir, key);
    }

    public boolean exists(CacheKey key) {
        if (!enabled) return false;
        return Files.exists(resolve(key).finalPath);
    }

    /**
     * Attempts to load the cached index. Returns empty on miss.
     *
     * <p>If the file exists but cannot be loaded (corrupt/partial), this returns empty.
     * (Callers typically rebuild in that case.)</p>
     */
    public Optional<ImmutableGraphIndex> tryLoad(CacheKey key) throws IOException {
        if (!enabled) return Optional.empty();
        Entry e = resolve(key);
        if (!Files.exists(e.finalPath)) return Optional.empty();

        try {
            return Optional.of(OnDiskGraphIndex.load(ReaderSupplierFactory.open(e.finalPath)));
        } catch (RuntimeException | IOException ex) {
            // treat as miss; leave cleanup policy to the caller (or call invalidate(key) if desired)
            return Optional.empty();
        }
    }

    public void invalidate(CacheKey key) throws IOException {
        if (!enabled) return;
        Files.deleteIfExists(resolve(key).finalPath);
    }

    // ---------------------------
    // Write handle (tmp + commit)
    // ---------------------------

    public WriteHandle beginWrite(CacheKey key, Overwrite overwrite) throws IOException {
        if (!enabled) {
            throw new IllegalStateException("Cache is disabled");
        }
        Objects.requireNonNull(overwrite, "overwrite");
        Entry e = resolve(key);

        Files.createDirectories(cacheDir);

        if (overwrite == Overwrite.DENY && Files.exists(e.finalPath)) {
            throw new IllegalStateException("Cache entry already exists: " + e.finalPath);
        }

        // Ensure any previous temp is gone
        Files.deleteIfExists(e.tmpPath);

        // If overwriting, remove the final first (best-effort)
        if (overwrite == Overwrite.ALLOW) {
            Files.deleteIfExists(e.finalPath);
        }

        return new WriteHandle(this, e);
    }

    public static final class WriteHandle implements AutoCloseable {
        private final OnDiskGraphIndexCache cache;
        private final Entry entry;
        private boolean committed;

        private WriteHandle(OnDiskGraphIndexCache cache, Entry entry) {
            this.cache = cache;
            this.entry = entry;
            this.committed = false;
        }

        /** Path the builder should write to (tmp filename). */
        public Path writePath() {
            return entry.tmpPath;
        }

        /** Final cache path (for logging/debug). */
        public Path finalPath() {
            return entry.finalPath;
        }

        /** Signature for logging. */
        public String signature() {
            return entry.signature;
        }

        /**
         * Atomically commits the write (tmp → final). After commit, {@link OnDiskGraphIndexCache#tryLoad(CacheKey)}
         * will see this entry.
         */
        public void commit() throws IOException {
            if (committed) return;

            try {
                Files.move(entry.tmpPath, entry.finalPath,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                // Still safe against partial reads: final name appears only after move.
                Files.move(entry.tmpPath, entry.finalPath, StandardCopyOption.REPLACE_EXISTING);
            }

            committed = true;
        }

        /** Best-effort cleanup of the temp file. */
        public void abort() throws IOException {
            Files.deleteIfExists(entry.tmpPath);
        }

        @Override
        public void close() throws IOException {
            if (!committed) {
                abort();
            }
        }
    }

    // ---------------------------
    // Entry + signature computation
    // ---------------------------

    public static final class Entry {
        public final boolean disabled;
        public final String signature;
        public final Path finalPath;
        public final Path tmpPath;

        private Entry(boolean disabled, String signature, Path finalPath, Path tmpPath) {
            this.disabled = disabled;
            this.signature = signature;
            this.finalPath = finalPath;
            this.tmpPath = tmpPath;
        }

        static Entry disabled() {
            return new Entry(true, "", Path.of(""), Path.of(""));
        }

        static Entry compute(Path cacheDir, CacheKey key) {
            String rawName = Paths.get(key.datasetName).getFileName().toString();
            String datasetBase = rawName.replaceFirst("\\.[^.]+$", "");

            String featureSetName = key.featureSet.stream()
                    .map(FeatureId::name)
                    .sorted()
                    .collect(Collectors.joining("-"));

            String signature = String.join("_",
                    datasetBase,
                    featureSetName,
                    "M" + key.M,
                    "ef" + key.efConstruction,
                    "of" + key.neighborOverflow,
                    key.addHierarchy ? "H1" : "H0",
                    key.refineFinalGraph ? "R1" : "R0",
                    key.compressorId
            );

            String finalName = sanitizePathComponent("graph_" + signature);
            Path finalPath = cacheDir.resolve(finalName);
            Path tmpPath = finalPath.resolveSibling("tmp_" + finalName);
            return new Entry(false, signature, finalPath, tmpPath);
        }
    }

    /**
     * Sanitizes a filename component to be filesystem-friendly.
     */
    public static String sanitizePathComponent(String name) {
        // replace any character that isn’t a letter, digit, dot or underscore with an underscore
        return name.replaceAll("[^A-Za-z0-9._]", "_");
    }
}
