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

import io.github.jbellis.jvector.example.util.SiftLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;
import software.amazon.awssdk.auth.credentials.AnonymousCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.transfer.s3.S3TransferManager;
import software.amazon.awssdk.transfer.s3.model.CompletedFileDownload;
import software.amazon.awssdk.transfer.s3.model.DownloadFileRequest;
import software.amazon.awssdk.transfer.s3.model.FileDownload;
import software.amazon.awssdk.transfer.s3.progress.LoggingTransferListener;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Stream;

/// A dataset loader that works with fvec/ivec datasets described by YAML catalog files
/// matching {@code *.yaml} or {@code *.yml}.
/// Supports S3, HTTP, local-only, and combined remote+local modes.
///
/// ### Catalog format
///
/// Each YAML catalog file lists datasets with their base, query, and ground truth
/// files. Optional fields control where files are stored and fetched:
///
/// - {@code base_url} — overrides the default remote base URL for this entry
/// - {@code cache_dir} — overrides where files are cached locally (relative or absolute path)
///
/// A special {@code _defaults} entry provides default values that are folded into all other
/// entries (unless the entry already specifies a value). Any root key starting with {@code _}
/// is excluded from dataset names.
///
/// The environment variable {@code DATASET_CACHE_DIR} sets a global default cache directory
/// when no {@code cache_dir} is specified at any level.
///
/// Field values may contain {@code ${VAR}} references to environment variables, which are
/// expanded at load time. The bash-style {@code ${VAR:-default}} syntax is supported to
/// provide a fallback value when the variable is not set. An {@link IllegalArgumentException}
/// is thrown if a referenced variable is not set and no default is provided.
///
/// A special {@code _include} entry can reference a remote catalog URL. The remote catalog
/// is fetched and its raw contents are cached locally in a hidden snapshot file for offline use.
/// On each run, the effective included entries are rebuilt by applying the local
/// {@code _defaults} to the fetched (or cached) remote entries. Local entries in the same
/// wrapper file are processed afterward and therefore take precedence over included remote entries.
/// This lets a single local file act as a thin configuration wrapper around a remote catalog:
/// ```yaml
/// _defaults:
///   cache_dir: ${DATASET_CACHE_DIR:-fvec}
/// _include:
///   url: s3://bucket/datasets-clean/catalog_entries.yaml
/// ```
///
/// ```yaml
/// _defaults:
///   base_url: s3://my-bucket/${DATASET_HASH}/
///   cache_dir: /data/cache
///
/// ada002-100k:
///   base: ada_002_100k_base_99287.fvecs
///   query: ada_002_100k_query_10000.fvecs
///   gt: ada_002_100k_gt_ip_100.ivecs
///
/// # private dataset with its own remote source and cache location
/// dpr-1M:
///   base_url: s3://my-bucket/SECRET_HASH/dpr/
///   cache_dir: /fast-ssd/dpr
///   base: c4-en_base_1M_norm.fvecs
///   query: c4-en_query_10k_norm.fvecs
///   gt: dpr_1m_gt_norm_ip_k100.ivecs
/// ```
/// Filenames are resolved relative to the entry's cache directory (local) or the base URL (remote).
/// When {@code base_url} is present on an entry, it is used instead of the loader's default remote
/// base URL for that entry's files.
///
/// ### Usage patterns
///
/// **Remote with local caching** — files are downloaded on first use and cached locally.
/// Subsequent runs use cached files. Set {@code checkForUpdates=true} to be warned when the
/// remote catalog changes. Supports both HTTP and S3 URLs.
/// ```java
/// var loader = new DataSetLoaderSimpleMFD(
///     "s3://bucket/datasets-clean/catalog_entries.yaml",
///     "fvec/catalog_entries.yaml",    // local cache path
///     true                            // warn if remote catalog differs from local
/// );
/// ```
///
/// **Local-only with recursive discovery** — the single-arg constructor accepts a directory
/// and recursively scans it for all {@code .yaml}/{@code .yml} files. This lets you organise
/// datasets in subdirectories, including private datasets with per-entry {@code base_url} overrides:
/// ```
/// local_datasets/
///   mydatasets/
///     user_entries.yaml               # your personal local datasets
///   private-infra/
///     private_entries.yaml            # private remote datasets with base_url per entry
/// ```
/// ```java
/// var loader = new DataSetLoaderSimpleMFD("local_datasets");
/// ```
///
/// **Remote+local hybrid** — if the local directory already contains {@code catalog_entries.yaml}
/// and data files, they are used as-is. Missing data files are downloaded from the remote.
/// ```java
/// var loader = new DataSetLoaderSimpleMFD(
///     "s3://bucket/datasets-clean/catalog_entries.yaml",
///     "/data/datasets/catalog_entries.yaml",
///     true
/// );
/// ```
///
/// ### Metadata
///
/// Dataset metadata (similarity function, load behavior) is resolved from
/// {@code dataset-metadata.yml} via {@link DataSetMetadataReader}. A custom metadata reader
/// can be provided via the 4-argument constructor.
///
/// @see DataSetLoader
public class DataSetLoaderSimpleMFD implements DataSetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DataSetLoaderSimpleMFD.class);
    private static final String DEFAULT_CATALOG_FILENAME = "catalog_entries.yaml";
    private static final String CATALOG_GLOB = "*.{yaml,yml}";

    // ========================================================================================
    // LOG REDACTION — auto-redacts secret-like path segments to prevent leakage
    // ========================================================================================

    /// Minimum number of hex characters (ignoring separators) for a path segment to be
    /// considered a potential secret (hash, API key, token, etc.).
    private static final int MIN_HEX_CHARS = 20;

    /// Set JVECTOR_LOG_REDACT=false to disable automatic redaction of secret-like path segments.
    private static final boolean REDACT_ENABLED;
    static {
        String env = System.getenv("JVECTOR_LOG_REDACT");
        REDACT_ENABLED = !"false".equalsIgnoreCase(env);
    }

    /// Redacts path segments that look like secrets (hashes, API keys, tokens) to prevent
    /// accidental leakage in log output and exception messages.
    ///
    /// A path segment is redacted if it contains {@value #MIN_HEX_CHARS} or more hex
    /// characters after stripping common separators ({@code -}, {@code .}, {@code _}) and
    /// the {@code 0x} prefix. This catches SHA-1 (40), SHA-256 (64), API keys, and similar
    /// patterns while preserving normal names like {@code datasets-clean} or {@code e5-base-v2-100k}.
    ///
    /// Set {@code JVECTOR_LOG_REDACT=false} to disable.
    static String redact(Object value) {
        if (value == null) return "null";
        if (!REDACT_ENABLED) return value.toString();
        String s = value.toString();
        if (s.isEmpty()) return s;

        var sb = new StringBuilder(s.length());
        int i = 0;
        while (i < s.length()) {
            // find the next path segment (delimited by / or \)
            int segStart = i;
            while (i < s.length() && s.charAt(i) != '/' && s.charAt(i) != '\\') {
                i++;
            }
            String segment = s.substring(segStart, i);
            sb.append(looksLikeSecret(segment) ? "[[redacted]]" : segment);

            // append the delimiter(s)
            while (i < s.length() && (s.charAt(i) == '/' || s.charAt(i) == '\\')) {
                sb.append(s.charAt(i));
                i++;
            }
        }
        return sb.toString();
    }

    /// Returns true if the segment looks like a hash, token, or API key.
    /// Strips common separators and 0x prefix, then counts hex characters.
    private static boolean looksLikeSecret(String segment) {
        if (segment.isEmpty()) return false;

        String stripped = segment;
        // strip 0x or 0X prefix
        if (stripped.startsWith("0x") || stripped.startsWith("0X")) {
            stripped = stripped.substring(2);
        }

        int hexCount = 0;
        int totalSignificant = 0; // non-separator characters
        for (int i = 0; i < stripped.length(); i++) {
            char c = stripped.charAt(i);
            if (c == '-' || c == '.' || c == '_') continue; // ignore separators
            totalSignificant++;
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
                hexCount++;
            }
        }

        // must have enough hex chars and they must be the majority of significant chars
        return hexCount >= MIN_HEX_CHARS && totalSignificant > 0
                && (double) hexCount / totalSignificant >= 0.75;
    }

    /// Entry source. Local entries always take precedence over included remote entries.
    private enum CatalogSource {
        LOCAL,
        INCLUDED_REMOTE
    }

    /// Resolved entry in the merged catalog. Tracks where the entry came from so that
    /// local file resolution, precedence, and per-entry remote base URL overrides work correctly.
    private static class CatalogEntry {
        final Map<String, String> fields;
        final Path cacheDir;       // where data files are cached locally
        final String baseUrl;      // per-entry base_url override, or null
        final CatalogSource source;

        CatalogEntry(Map<String, String> fields, Path cacheDir, String baseUrl, CatalogSource source) {
            this.fields = fields;
            this.cacheDir = cacheDir;
            this.baseUrl = baseUrl;
            this.source = source;
        }
    }

    private static final String ENV_DATASET_CACHE_DIR = "DATASET_CACHE_DIR";

    private final String remoteBasePath;
    private final Map<String, CatalogEntry> catalog;
    private final Path localCacheDir;
    private final DataSetMetadataReader metadata;
    private final HttpClient httpClient;

    // S3 instances for connection pooling
    private S3AsyncClient s3Client;
    private S3TransferManager s3TransferManager;

    /// Creates a local-only loader that recursively discovers all {@code .yaml}/{@code .yml}
    /// files under the given path.
    ///
    /// The {@code localPath} may be either a directory (scanned recursively for catalog files)
    /// or the full path to a single catalog YAML file.
    ///
    /// If the path does not exist or contains no catalog files, the loader is constructed
    /// successfully but will return empty for all dataset lookups. This allows it to be safely
    /// registered in a loader list without failing when local datasets are not present.
    ///
    /// @param localPath the local directory to scan or full path to a catalog YAML file
    public DataSetLoaderSimpleMFD(String localPath) {
        this(null, localPath, false, DataSetMetadataReader.load());
    }

    /// Creates a loader using the default dataset metadata from {@code dataset-metadata.yml}.
    ///
    /// The {@code localPath} may be either a directory or the full path to a catalog YAML file.
    /// If it ends in {@code .yaml} or {@code .yml}, that file is used as the catalog.
    /// Otherwise, the directory is scanned recursively for all {@code .yaml}/{@code .yml} files.
    ///
    /// Entries without an explicit {@code cache_dir} default to {@code DATASET_CACHE_DIR}
    /// when that environment variable is set; otherwise they default to the catalog file's
    /// directory. In constructor-driven remote-catalog mode (when no local catalog exists and
    /// {@code catalogUrl} is used), fetched remote entries default to {@code dataset_cache/}.
    /// Entry-level and {@code _defaults}-level {@code cache_dir} values take precedence.
    ///
    /// @param catalogUrl      the full URL (HTTP or S3) to the remote catalog, or null/empty
    ///                        for local-only mode
    /// @param localPath       the local directory or full path to a catalog YAML file
    /// @param checkForUpdates if true and a local catalog already exists, the remote catalog is
    ///                        fetched and compared; a warning is logged if they differ
    public DataSetLoaderSimpleMFD(String catalogUrl, String localPath, boolean checkForUpdates) {
        this(catalogUrl, localPath, checkForUpdates, DataSetMetadataReader.load());
    }

    /// Creates a loader with a custom metadata reader for resolving dataset properties.
    ///
    /// @param catalogUrl      the full URL (HTTP or S3) to the remote catalog, or null/empty
    ///                        for local-only mode
    /// @param localPath       the local directory or full path to a catalog YAML file
    /// @param checkForUpdates if true and a local catalog already exists, the remote catalog is
    ///                        fetched and compared; a warning is logged if they differ.
    ///                        Ignored when catalogUrl is null/empty.
    /// @param metadata        the metadata reader for resolving dataset properties
    public DataSetLoaderSimpleMFD(String catalogUrl, String localPath, boolean checkForUpdates, DataSetMetadataReader metadata) {
        this.metadata = metadata;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        // resolve localPath for catalog discovery. For discovered local/include catalogs,
        // entries without an explicit cache_dir fall back to DATASET_CACHE_DIR or the
        // catalog file's directory. Pure constructor-driven remote catalogs fall back to
        // dataset_cache.
        Path resolvedPath = Paths.get(localPath);
        Path localCatalog;
        this.localCacheDir = Paths.get("dataset_cache");

        if (localPath.endsWith(".yaml") || localPath.endsWith(".yml")) {
            localCatalog = resolvedPath;
        } else {
            localCatalog = resolvedPath.resolve(DEFAULT_CATALOG_FILENAME);
        }

        // determine whether we have a remote URL (S3 or HTTP)
        boolean isRemote = catalogUrl != null && !catalogUrl.isEmpty()
                && (catalogUrl.startsWith("http://") || catalogUrl.startsWith("https://") || catalogUrl.startsWith("s3://"));

        // derive remote base path by stripping the filename from the catalog URL
        if (isRemote) {
            int lastSlash = catalogUrl.lastIndexOf('/');
            this.remoteBasePath = catalogUrl.substring(0, lastSlash + 1);
        } else {
            this.remoteBasePath = null;
        }

        // load local catalog entries — either from a single file or by scanning a directory tree
        Map<String, CatalogEntry> localEntries = new HashMap<>();
        if (localPath.endsWith(".yaml") || localPath.endsWith(".yml")) {
            // single file mode
            if (Files.exists(localCatalog)) {
                loadCatalogEntries(localCatalog, localEntries);
            }
        } else if (Files.isDirectory(resolvedPath)) {
            // recursive scan mode
            scanForCatalogs(resolvedPath, localEntries);
        } else if (Files.exists(localCatalog)) {
            // directory doesn't exist yet but might after remote fetch — check the default file
            loadCatalogEntries(localCatalog, localEntries);
        }

        if (!localEntries.isEmpty()) {
            logger.info("Loaded {} datasets from local catalog(s) under {}", localEntries.size(), redact(localCacheDir));
        }

        if (isRemote) {
            if (!localEntries.isEmpty()) {
                this.catalog = localEntries;
                if (checkForUpdates) checkRemoteCatalogForUpdates(catalogUrl, localEntries);
            } else {
                logger.info("No local catalog found, fetching from {}", redact(catalogUrl));
                var remoteCatalogData = fetchRemoteCatalogRaw(catalogUrl);
                this.catalog = toCatalogEntries(remoteCatalogData, localCacheDir);
                saveCatalogLocally(localCatalog, catalogUrl, remoteCatalogData);
            }
        } else {
            if (!localEntries.isEmpty()) {
                this.catalog = localEntries;
            } else {
                logger.info("No catalog found under {}. This loader will not match any datasets.", redact(localCacheDir));
                this.catalog = Map.of();
            }
        }
    }

    @Override
    public Optional<DataSetInfo> loadDataSet(String dataSetName) {
        var entry = catalog.get(dataSetName);
        if (entry == null) return Optional.empty();

        var baseFile = entry.fields.get("base");
        var queryFile = entry.fields.get("query");
        var gtFile = entry.fields.get("gt");
        if (baseFile == null || queryFile == null || gtFile == null) {
            logger.error("Dataset '{}' is missing required fields (base, query, gt) in catalog", dataSetName);
            return Optional.empty();
        }

        logger.info("Found dataset '{}' in catalog", dataSetName);
        var startTime = System.nanoTime();

        // determine the effective remote base URL and local cache directory for this entry
        String effectiveBaseUrl = entry.baseUrl != null ? entry.baseUrl : remoteBasePath;
        Path effectiveCacheDir = entry.cacheDir;

        // Execute downloads simultaneously to maximize network bandwidth
        try {
            var f1 = CompletableFuture.runAsync(() -> ensureQuietly(baseFile, effectiveCacheDir, effectiveBaseUrl));
            var f2 = CompletableFuture.runAsync(() -> ensureQuietly(queryFile, effectiveCacheDir, effectiveBaseUrl));
            var f3 = CompletableFuture.runAsync(() -> ensureQuietly(gtFile, effectiveCacheDir, effectiveBaseUrl));

            CompletableFuture.allOf(f1, f2, f3).join();
        } catch (Exception e) {
            throw new RuntimeException("Failed to obtain dataset files for " + dataSetName, e);
        }

        logger.info("Dataset files ready for '{}' in {}s", dataSetName, String.format("%.2f", (System.nanoTime() - startTime) / 1e9));

        var props = metadata.getProperties(dataSetName)
                .orElseThrow(() -> new IllegalArgumentException(
                        String.format(
                                "Dataset '%s' was found in dataset catalog, but no metadata entry was found in dataset-metadata.yml. ",
                                dataSetName)));
        return Optional.of(new DataSetInfo(props, () -> {
            var baseVectors = SiftLoader.readFvecs(effectiveCacheDir.resolve(baseFile).toString());
            var queryVectors = SiftLoader.readFvecs(effectiveCacheDir.resolve(queryFile).toString());
            var gtVectors = SiftLoader.readIvecs(effectiveCacheDir.resolve(gtFile).toString());
            return DataSetUtils.processDataSet(dataSetName, props, baseVectors, queryVectors, gtVectors);
        }));
    }

    // ========================================================================================
    // CATALOG DISCOVERY & LOADING
    // ========================================================================================

    /// Returns the effective source for a discovered catalog file.
    /// Generated remote-catalog snapshots are treated as included remote entries so that
    /// real local catalogs continue to take precedence across runs.
    private static CatalogSource catalogSource(Map<String, Map<String, String>> raw) {
        Map<String, String> meta = raw.get("_meta");
        if (meta != null && "true".equalsIgnoreCase(meta.get("generated_remote_catalog"))) {
            return CatalogSource.INCLUDED_REMOTE;
        }
        return CatalogSource.LOCAL;
    }

    /// Inserts an entry while preserving the precedence rule that real local entries
    /// always win over included remote entries.
    private static void putCatalogEntry(Map<String, CatalogEntry> target, String name, CatalogEntry entry) {
        CatalogEntry existing = target.get(name);
        if (existing == null || entry.source == CatalogSource.LOCAL || existing.source != CatalogSource.LOCAL) {
            target.put(name, entry);
        }
    }

    /// Returns the hidden cache file used to persist the raw contents of an included remote catalog.
    private static Path includeCacheFile(Path catalogDir, String includeUrl) {
        return catalogDir.resolve(".catalog-cache")
                .resolve("include-" + sha256Hex(includeUrl) + ".yaml.cache");
    }

    private static String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                hex.append(Character.forDigit((b >> 4) & 0xF, 16));
                hex.append(Character.forDigit(b & 0xF, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 should always be available", e);
        }
    }

    /// Recursively scans a directory tree for {@code .yaml}/{@code .yml} files and merges
    /// all entries into the given map. Later entries of the same source type may override
    /// earlier ones, but real local entries always take precedence over included remote entries.
    private void scanForCatalogs(Path rootDir, Map<String, CatalogEntry> target) {
        try (Stream<Path> paths = Files.walk(rootDir)) {
            var matcher = rootDir.getFileSystem().getPathMatcher("glob:" + CATALOG_GLOB);
            paths.filter(p -> p.getFileName() != null && matcher.matches(p.getFileName()))
                    .forEach(catalogFile -> loadCatalogEntries(catalogFile, target));
        } catch (IOException e) {
            logger.warn("Error scanning for catalogs under {}: {}", redact(rootDir), redact(e.getMessage()));
        }
    }

    /// Loads entries from a single catalog file into the target map.
    /// Handles {@code _defaults} folding, {@code _include} remote fetching, and
    /// {@code _}-prefixed key exclusion.
    ///
    /// When {@code _include} is present, its value (after env var expansion) is treated as a
    /// remote catalog URL. The remote entries are fetched and merged with the local defaults,
    /// so a single local file can act as a thin wrapper around a remote catalog.
    private void loadCatalogEntries(Path catalogFile, Map<String, CatalogEntry> target) {
        var raw = loadCatalogFromFile(catalogFile);
        if (raw.isEmpty()) return;

        Path catalogDir = catalogFile.getParent() != null ? catalogFile.getParent() : Paths.get(".");
        CatalogSource source = catalogSource(raw);

        // extract and expand _defaults if present
        Map<String, String> defaults = raw.getOrDefault("_defaults", Map.of());
        if (!defaults.isEmpty()) {
            defaults = resolveEnvVars(defaults);
        }

        // handle _include: fetch remote catalog and merge with local defaults
        Map<String, String> includeEntry = raw.get("_include");
        if (includeEntry != null) {
            String includeUrl = includeEntry.get("url");
            if (includeUrl != null) {
                includeUrl = expandEnvVars(includeUrl);
                loadRemoteInclude(includeUrl, defaults, catalogDir, includeCacheFile(catalogDir, includeUrl), target);
            }
        }

        // count real entries (non-underscore keys)
        long entryCount = raw.keySet().stream().filter(k -> !k.startsWith("_")).count();
        if (entryCount > 0) {
            logger.info("Loading catalog from {} ({} entries)", redact(catalogFile), entryCount);
        }

        for (var e : raw.entrySet()) {
            String name = e.getKey();
            // skip entries whose key starts with _
            if (name.startsWith("_")) continue;

            // fold defaults into this entry (entry values take precedence)
            Map<String, String> fields = new HashMap<>(defaults);
            if (e.getValue() != null) {
                fields.putAll(e.getValue());
            }

            putCatalogEntry(target, name, buildCatalogEntry(fields, catalogDir, source));
        }
    }

    /// Fetches a remote catalog via {@code _include}, caches its raw contents locally for
    /// offline reuse, and merges the resulting entries with the local defaults. If the remote
    /// fetch fails and a cached snapshot exists, the cached catalog is used instead.
    private void loadRemoteInclude(String includeUrl, Map<String, String> defaults,
                                   Path catalogDir, Path cachedIncludeFile,
                                   Map<String, CatalogEntry> target) {
        Map<String, Map<String, String>> remoteCatalog;
        boolean usedCachedSnapshot = false;

        try {
            logger.info("Including remote catalog from {}", redact(includeUrl));
            remoteCatalog = fetchRemoteCatalogRaw(includeUrl, cachedIncludeFile);
        } catch (Exception e) {
            if (!Files.isRegularFile(cachedIncludeFile)) {
                logger.warn("Failed to include remote catalog from {}: {}", redact(includeUrl), redact(e.getMessage()));
                return;
            }

            logger.warn("Failed to include remote catalog from {}: {}. Using cached catalog {}",
                    redact(includeUrl), redact(e.getMessage()), redact(cachedIncludeFile));
            remoteCatalog = loadCatalogFromFile(cachedIncludeFile);
            usedCachedSnapshot = true;
        }

        // derive the remote base path from the include URL
        int lastSlash = includeUrl.lastIndexOf('/');
        String remoteBase = lastSlash >= 0 ? includeUrl.substring(0, lastSlash + 1) : null;

        long entryCount = 0;
        for (var e : remoteCatalog.entrySet()) {
            if (e.getKey().startsWith("_")) continue;
            entryCount++;

            // fold local defaults into remote entry (remote values take precedence over defaults,
            // but local entries always take precedence — those are handled in the caller's loop)
            Map<String, String> fields = new HashMap<>(defaults);
            if (e.getValue() != null) {
                fields.putAll(e.getValue());
            }
            // if the entry doesn't already have a base_url, use the remote catalog's base path
            if (!fields.containsKey("base_url") && remoteBase != null) {
                fields.put("base_url", remoteBase);
            }

            putCatalogEntry(target, e.getKey(), buildCatalogEntry(fields, catalogDir, CatalogSource.INCLUDED_REMOTE));
        }

        logger.info("Included {} datasets from {} catalog", entryCount,
                usedCachedSnapshot ? "cached" : "remote");
    }

    /// Converts a raw catalog map (from a remote fetch) into CatalogEntry objects.
    /// Handles {@code _defaults} folding and {@code _}-prefixed key exclusion.
    private static Map<String, CatalogEntry> toCatalogEntries(Map<String, Map<String, String>> raw, Path localDir) {
        Map<String, String> defaults = raw.getOrDefault("_defaults", Map.of());

        var result = new HashMap<String, CatalogEntry>();
        for (var e : raw.entrySet()) {
            if (e.getKey().startsWith("_")) continue;

            Map<String, String> fields = new HashMap<>(defaults);
            if (e.getValue() != null) {
                fields.putAll(e.getValue());
            }

            putCatalogEntry(result, e.getKey(), buildCatalogEntry(fields, localDir, CatalogSource.INCLUDED_REMOTE));
        }
        return result;
    }

    private static final java.util.Set<String> KNOWN_FIELDS = java.util.Set.of(
            "base", "query", "gt", "base_url", "cache_dir"
    );

    /// Builds a CatalogEntry from merged fields, resolving env vars, base_url, and cache_dir.
    /// Throws if any unknown fields are present.
    private static CatalogEntry buildCatalogEntry(Map<String, String> fields, Path catalogDir, CatalogSource source) {
        // validate that all fields are recognized
        for (String key : fields.keySet()) {
            if (!KNOWN_FIELDS.contains(key)) {
                throw new IllegalArgumentException(
                        "Unknown field '" + key + "' in catalog entry. Known fields: " + KNOWN_FIELDS);
            }
        }

        // expand ${VAR} references in all field values
        var resolved = resolveEnvVars(fields);

        String baseUrl = resolved.get("base_url");
        if (baseUrl != null && !baseUrl.endsWith("/")) {
            baseUrl = baseUrl + "/";
        }

        // resolve cache_dir: entry field > DATASET_CACHE_DIR env var > catalog file's directory
        Path cacheDir;
        String cacheDirField = resolved.get("cache_dir");
        if (cacheDirField != null && !cacheDirField.isEmpty()) {
            cacheDir = Paths.get(cacheDirField);
        } else {
            String envCacheDir = System.getenv(ENV_DATASET_CACHE_DIR);
            if (envCacheDir != null && !envCacheDir.isEmpty()) {
                cacheDir = Paths.get(envCacheDir);
            } else {
                cacheDir = catalogDir;
            }
        }

        return new CatalogEntry(resolved, cacheDir, baseUrl, source);
    }

    /// Matches {@code ${VAR}} and {@code ${VAR:-default}} syntax.
    private static final java.util.regex.Pattern ENV_VAR_PATTERN =
            java.util.regex.Pattern.compile("\\$\\{([^:}]+)(?::-((?:[^}]*)?))?}");

    /// Expands {@code ${VAR}} and {@code ${VAR:-default}} references in all field values
    /// using environment variables. Throws {@link IllegalArgumentException} if a referenced
    /// variable is not set and no default is provided.
    private static Map<String, String> resolveEnvVars(Map<String, String> fields) {
        var resolved = new HashMap<String, String>(fields.size());
        for (var e : fields.entrySet()) {
            resolved.put(e.getKey(), expandEnvVars(e.getValue()));
        }
        return resolved;
    }

    /// Expands all {@code ${VAR}} and {@code ${VAR:-default}} occurrences in a single string value.
    private static String expandEnvVars(String value) {
        if (value == null || !value.contains("${")) {
            return value;
        }
        var matcher = ENV_VAR_PATTERN.matcher(value);
        var sb = new StringBuilder();
        while (matcher.find()) {
            String varName = matcher.group(1);
            String defaultValue = matcher.group(2); // null if no :- was present
            String envValue = System.getenv(varName);
            if (envValue == null) {
                if (defaultValue != null) {
                    envValue = defaultValue;
                } else {
                    throw new IllegalArgumentException(
                            "Environment variable '${" + varName + "}' referenced in catalog entry is not set");
                }
            }
            matcher.appendReplacement(sb, java.util.regex.Matcher.quoteReplacement(envValue));
        }
        matcher.appendTail(sb);
        return sb.toString();
    }

    // ========================================================================================
    // FILE AVAILABILITY
    // ========================================================================================

    private void ensureQuietly(String filename, Path cacheDir, String baseUrl) {
        try {
            ensureFileAvailable(filename, cacheDir, baseUrl);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /// Ensures a dataset file is available locally. Checks in the entry's cache directory first.
    /// If not found and a remote base URL is available (either per-entry or loader-level),
    /// downloads the file.
    private void ensureFileAvailable(String filename, Path cacheDir, String baseUrl) throws IOException {
        Path localPath = cacheDir.resolve(filename);
        if (Files.exists(localPath)) return;
        if (baseUrl == null) throw new IOException("File not found locally and no remote URL configured: " + redact(localPath));

        Path parent = localPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        String url = baseUrl + filename;
        logger.info("Downloading {} -> {}", redact(url), redact(localPath));
        downloadUrlToFile(url, localPath);
    }

    // ========================================================================================
    // REMOTE CATALOG OPERATIONS
    // ========================================================================================

    private Map<String, Map<String, String>> fetchRemoteCatalogRaw(String catalogUrl) {
        return fetchRemoteCatalogRaw(catalogUrl, null);
    }

    private Map<String, Map<String, String>> fetchRemoteCatalogRaw(String catalogUrl, Path snapshotFile) {
        try {
            Path tempDir = snapshotFile != null && snapshotFile.getParent() != null
                    ? snapshotFile.getParent()
                    : null;
            if (tempDir != null) {
                Files.createDirectories(tempDir);
            }

            Path tempFile = tempDir != null
                    ? Files.createTempFile(tempDir, "catalog-", ".tmp")
                    : Files.createTempFile("catalog-", ".tmp");
            try {
                downloadUrlToFile(catalogUrl, tempFile);
                var catalog = loadCatalogFromFile(tempFile);

                if (snapshotFile != null) {
                    Files.move(tempFile, snapshotFile,
                            StandardCopyOption.ATOMIC_MOVE,
                            StandardCopyOption.REPLACE_EXISTING);
                }
                return catalog;
            } finally {
                Files.deleteIfExists(tempFile);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to fetch dataset catalog from " + redact(catalogUrl), e);
        }
    }

    private void saveCatalogLocally(Path localCatalog, String catalogUrl,
                                    Map<String, Map<String, String>> catalogData) {
        try {
            Path parent = localCatalog.getParent() != null ? localCatalog.getParent() : Paths.get(".");
            Files.createDirectories(parent);

            Path tempFile = Files.createTempFile(parent, "catalog-", ".tmp");
            try {
                Map<String, Map<String, String>> annotated = new LinkedHashMap<>();
                Map<String, String> meta = new LinkedHashMap<>();
                meta.put("generated_remote_catalog", "true");
                meta.put("remote_catalog_url", catalogUrl);
                annotated.put("_meta", meta);
                annotated.putAll(catalogData);

                Files.writeString(tempFile, new Yaml().dump(annotated));
                Files.move(tempFile, localCatalog,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } finally {
                Files.deleteIfExists(tempFile);
            }
        } catch (Exception e) {
            logger.warn("Failed to cache catalog locally: {}", redact(e.getMessage()));
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, String>> loadCatalogFromFile(Path path) {
        try (InputStream in = Files.newInputStream(path)) {
            Map<String, Map<String, String>> result = new Yaml().load(in);
            return result != null ? result : Map.of();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load catalog from " + redact(path), e);
        }
    }

    /// Fetches the remote catalog and compares it to the local one, logging a warning if they differ.
    private void checkRemoteCatalogForUpdates(String catalogUrl, Map<String, CatalogEntry> localEntries) {
        try {
            var remoteCatalogData = fetchRemoteCatalogRaw(catalogUrl);
            // compare just the dataset names and file fields, ignoring localDir
            boolean differs = false;
            if (remoteCatalogData.size() != localEntries.size()) {
                differs = true;
            } else {
                for (var e : remoteCatalogData.entrySet()) {
                    var local = localEntries.get(e.getKey());
                    if (local == null || !local.fields.equals(e.getValue())) {
                        differs = true;
                        break;
                    }
                }
            }
            if (differs) {
                logger.warn("Remote catalog at {} differs from local catalog. Consider updating your local copy.", redact(catalogUrl));
            }
        } catch (Exception e) {
            logger.warn("Could not check remote catalog for updates: {}", redact(e.getMessage()));
        }
    }

    // ========================================================================================
    // TRANSPORT PROTOCOL ROUTING (S3 vs HTTP)
    // ========================================================================================

    private void downloadUrlToFile(String url, Path localPath) throws IOException {
        if (url.startsWith("s3://")) {
            downloadFileS3(url, localPath);
        } else if (url.startsWith("http://") || url.startsWith("https://")) {
            downloadFileHttp(url, localPath);
        } else {
            throw new IllegalArgumentException("Unsupported URL scheme for download: " + redact(url));
        }
    }

    // ========================================================================================
    // S3 TRANSFER MANAGER IMPLEMENTATION
    // ========================================================================================

    private synchronized S3TransferManager getS3TransferManager() {
        if (s3TransferManager == null) {
            s3Client = s3AsyncClient();
            s3TransferManager = S3TransferManager.builder().s3Client(s3Client).build();
        }
        return s3TransferManager;
    }

    private void downloadFileS3(String s3Url, Path localPath) throws IOException {
        String withoutScheme = s3Url.substring(5);
        int slashIdx = withoutScheme.indexOf('/');
        String bucket = withoutScheme.substring(0, slashIdx);
        String key = withoutScheme.substring(slashIdx + 1);

        S3TransferManager tm = getS3TransferManager();

        DownloadFileRequest request = DownloadFileRequest.builder()
                .getObjectRequest(b -> b.bucket(bucket).key(key))
                .addTransferListener(LoggingTransferListener.create())
                .destination(localPath)
                .build();

        boolean downloaded = false;
        for (int i = 0; i < 3; i++) { // 3 retries
            try {
                FileDownload downloadFile = tm.downloadFile(request);
                CompletedFileDownload result = downloadFile.completionFuture().join();
                long downloadedSize = Files.size(localPath);
                Long expectedSize = result.response().contentLength();

                // Null check prevents NullPointerException during unboxing.
                // If expectedSize is null, we trust the transfer manager's successful completion.
                if (expectedSize != null && downloadedSize != expectedSize) {
                    logger.error("Incomplete download (got {} of {} bytes). Retrying...", downloadedSize, expectedSize);
                    Files.deleteIfExists(localPath);
                    continue;
                }

                downloaded = true;
                break;
            } catch (Exception e) {
                logger.error("Download attempt {} failed for {}: {}", i + 1, redact(key), redact(e.getMessage()));
                Files.deleteIfExists(localPath);
            }
        }
        if (!downloaded) {
            throw new IOException("Failed to download " + redact(s3Url) + " after 3 attempts");
        }
    }

    private static S3AsyncClient s3AsyncClient() {
        return S3AsyncClient.crtBuilder()
                .region(Region.US_EAST_1)
                .credentialsProvider(AnonymousCredentialsProvider.create())
                .targetThroughputInGbps(10.0)
                .minimumPartSizeInBytes(8L * 1024 * 1024)
                .build();
    }

    // ========================================================================================
    // HTTP CLIENT IMPLEMENTATION
    // ========================================================================================

    private void downloadFileHttp(String url, Path localPath) throws IOException {
        var request = HttpRequest.newBuilder().uri(URI.create(url)).GET().build();

        Path targetDir = localPath.toAbsolutePath().getParent();
        if (targetDir != null) {
            Files.createDirectories(targetDir);
        }
        Path tempFile = Files.createTempFile(targetDir, "download-", ".tmp");

        try {
            var response = httpClient.send(request, HttpResponse.BodyHandlers.ofFile(tempFile));
            if (response.statusCode() != 200) {
                throw new IOException("HTTP " + response.statusCode() + " downloading " + redact(url));
            }
            Files.move(tempFile, localPath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Files.deleteIfExists(tempFile);
            throw new IOException("Interrupted downloading " + redact(url), e);
        } catch (Exception e) {
            Files.deleteIfExists(tempFile);
            throw e;
        }
    }
}
