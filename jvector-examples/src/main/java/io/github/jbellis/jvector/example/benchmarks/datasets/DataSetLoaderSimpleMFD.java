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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Stream;

/// A dataset loader that works with fvec/ivec datasets described by YAML catalog files
/// matching the pattern {@code *entries.yaml} (e.g. {@code catalog_entries.yaml},
/// {@code entries.yaml}, {@code private_entries.yaml}).
/// Supports S3, HTTP, local-only, and combined remote+local modes.
///
/// ### Catalog format
///
/// Each {@code catalog_entries.yaml} file lists datasets with their base, query, and ground truth
/// files. An optional {@code baseurl} field overrides the default remote base URL for that entry:
/// ```yaml
/// ada002-100k:
///   base: ada_002_100k_base_99287.fvecs
///   query: ada_002_100k_query_10000.fvecs
///   gt: ada_002_100k_gt_ip_100.ivecs
///
/// # private dataset with its own remote source
/// dpr-1M:
///   baseurl: s3://my-bucket/SECRET_HASH/dpr/
///   base: c4-en_base_1M_norm.fvecs
///   query: c4-en_query_10k_norm.fvecs
///   gt: dpr_1m_gt_norm_ip_k100.ivecs
/// ```
/// Filenames are resolved relative to the catalog's directory (local) or the base URL (remote).
/// When {@code baseurl} is present on an entry, it is used instead of the loader's default remote
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
/// and recursively scans it for all files matching {@code *entries.yaml}. This lets you organise
/// datasets in subdirectories, including private datasets with per-entry {@code baseurl} overrides:
/// ```
/// local_datasets/
///   mydatasets/
///     user_entries.yaml               # your personal local datasets
///   private-infra/
///     private_entries.yaml            # private remote datasets with baseurl per entry
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
/// {@code dataset_metadata.yml} via {@link DataSetMetadataReader}. A custom metadata reader
/// can be provided via the 4-argument constructor.
///
/// @see DataSetLoaderMFD
public class DataSetLoaderSimpleMFD implements DataSetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DataSetLoaderSimpleMFD.class);
    private static final String DEFAULT_CATALOG_FILENAME = "catalog_entries.yaml";
    private static final String CATALOG_GLOB = "*entries.yaml";

    /// Resolved entry in the merged catalog. Tracks where the entry came from so that
    /// local file resolution and per-entry remote base URL overrides work correctly.
    private static class CatalogEntry {
        final Map<String, String> fields;
        final Path localDir;       // directory containing this entry's catalog file
        final String baseUrl;      // per-entry baseurl override, or null

        CatalogEntry(Map<String, String> fields, Path localDir, String baseUrl) {
            this.fields = fields;
            this.localDir = localDir;
            this.baseUrl = baseUrl;
        }
    }

    private final String remoteBasePath;
    private final Map<String, CatalogEntry> catalog;
    private final Path localCacheDir;
    private final DataSetMetadataReader metadata;
    private final HttpClient httpClient;

    // S3 instances for connection pooling
    private S3AsyncClient s3Client;
    private S3TransferManager s3TransferManager;

    /// Creates a local-only loader that recursively discovers all {@code *entries.yaml}
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

    /// Creates a loader using the default dataset metadata from {@code dataset_metadata.yml}.
    ///
    /// The {@code localPath} may be either a directory or the full path to a catalog YAML file.
    /// If it ends in {@code .yaml} or {@code .yml}, the parent directory is used as the cache
    /// directory and that file is used as the catalog. Otherwise, the directory is scanned
    /// recursively for all {@code *entries.yaml} files.
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
    @SuppressWarnings("unchecked")
    public DataSetLoaderSimpleMFD(String catalogUrl, String localPath, boolean checkForUpdates, DataSetMetadataReader metadata) {
        this.metadata = metadata;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        // resolve localPath: if it points to a yaml file, use the parent as the cache dir
        Path resolvedPath = Paths.get(localPath);
        Path localCatalog;
        if (localPath.endsWith(".yaml") || localPath.endsWith(".yml")) {
            this.localCacheDir = resolvedPath.getParent() != null ? resolvedPath.getParent() : Paths.get(".");
            localCatalog = resolvedPath;
        } else {
            this.localCacheDir = resolvedPath;
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
            logger.info("Loaded {} datasets from local catalog(s) under {}", localEntries.size(), localCacheDir);
        }

        if (isRemote) {
            if (!localEntries.isEmpty()) {
                this.catalog = localEntries;
                if (checkForUpdates) checkRemoteCatalogForUpdates(catalogUrl, localEntries);
            } else {
                logger.info("No local catalog found, fetching from {}", catalogUrl);
                var remoteCatalogData = fetchRemoteCatalogRaw(catalogUrl);
                this.catalog = toCatalogEntries(remoteCatalogData, localCacheDir);
                saveCatalogLocally(localCatalog, catalogUrl);
            }
        } else {
            if (!localEntries.isEmpty()) {
                this.catalog = localEntries;
            } else {
                logger.info("No catalog found under {}. This loader will not match any datasets.", localCacheDir);
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

        // determine the effective remote base URL for this entry
        String effectiveBaseUrl = entry.baseUrl != null ? entry.baseUrl : remoteBasePath;
        Path effectiveLocalDir = entry.localDir;

        // Execute downloads simultaneously to maximize network bandwidth
        try {
            var f1 = CompletableFuture.runAsync(() -> ensureQuietly(baseFile, effectiveLocalDir, effectiveBaseUrl));
            var f2 = CompletableFuture.runAsync(() -> ensureQuietly(queryFile, effectiveLocalDir, effectiveBaseUrl));
            var f3 = CompletableFuture.runAsync(() -> ensureQuietly(gtFile, effectiveLocalDir, effectiveBaseUrl));

            CompletableFuture.allOf(f1, f2, f3).join();
        } catch (Exception e) {
            throw new RuntimeException("Failed to obtain dataset files for " + dataSetName, e);
        }

        logger.info("Dataset files ready for '{}' in {}s", dataSetName, String.format("%.2f", (System.nanoTime() - startTime) / 1e9));

        var props = metadata.getProperties(dataSetName).orElseThrow();
        return Optional.of(new DataSetInfo(props, () -> {
            var baseVectors = SiftLoader.readFvecs(effectiveLocalDir.resolve(baseFile).toString());
            var queryVectors = SiftLoader.readFvecs(effectiveLocalDir.resolve(queryFile).toString());
            var gtVectors = SiftLoader.readIvecs(effectiveLocalDir.resolve(gtFile).toString());
            return DataSetUtils.processDataSet(dataSetName, props, baseVectors, queryVectors, gtVectors);
        }));
    }

    // ========================================================================================
    // CATALOG DISCOVERY & LOADING
    // ========================================================================================

    /// Recursively scans a directory tree for files matching {@code *entries.yaml} and merges
    /// all entries into the given map. Later entries with the same name override earlier ones.
    private void scanForCatalogs(Path rootDir, Map<String, CatalogEntry> target) {
        try (Stream<Path> paths = Files.walk(rootDir)) {
            var matcher = rootDir.getFileSystem().getPathMatcher("glob:" + CATALOG_GLOB);
            paths.filter(p -> p.getFileName() != null && matcher.matches(p.getFileName()))
                    .forEach(catalogFile -> loadCatalogEntries(catalogFile, target));
        } catch (IOException e) {
            logger.warn("Error scanning for catalogs under {}: {}", rootDir, e.getMessage());
        }
    }

    /// Loads entries from a single catalog file into the target map.
    @SuppressWarnings("unchecked")
    private void loadCatalogEntries(Path catalogFile, Map<String, CatalogEntry> target) {
        var raw = loadCatalogFromFile(catalogFile);
        if (raw.isEmpty()) return;

        Path catalogDir = catalogFile.getParent() != null ? catalogFile.getParent() : Paths.get(".");
        logger.info("Loading catalog from {} ({} entries)", catalogFile, raw.size());

        for (var e : raw.entrySet()) {
            String name = e.getKey();
            Map<String, String> fields = e.getValue();
            String baseUrl = fields.get("baseurl");
            // ensure baseurl ends with / if present
            if (baseUrl != null && !baseUrl.endsWith("/")) {
                baseUrl = baseUrl + "/";
            }
            target.put(name, new CatalogEntry(fields, catalogDir, baseUrl));
        }
    }

    /// Converts a raw catalog map (from a remote fetch) into CatalogEntry objects.
    private static Map<String, CatalogEntry> toCatalogEntries(Map<String, Map<String, String>> raw, Path localDir) {
        var result = new HashMap<String, CatalogEntry>();
        for (var e : raw.entrySet()) {
            String baseUrl = e.getValue().get("baseurl");
            if (baseUrl != null && !baseUrl.endsWith("/")) {
                baseUrl = baseUrl + "/";
            }
            result.put(e.getKey(), new CatalogEntry(e.getValue(), localDir, baseUrl));
        }
        return result;
    }

    // ========================================================================================
    // FILE AVAILABILITY
    // ========================================================================================

    private void ensureQuietly(String filename, Path localDir, String baseUrl) {
        try {
            ensureFileAvailable(filename, localDir, baseUrl);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /// Ensures a dataset file is available locally. Checks in the entry's local directory first.
    /// If not found and a remote base URL is available (either per-entry or loader-level),
    /// downloads the file.
    private void ensureFileAvailable(String filename, Path localDir, String baseUrl) throws IOException {
        Path localPath = localDir.resolve(filename);
        if (Files.exists(localPath)) return;
        if (baseUrl == null) throw new IOException("File not found locally and no remote URL configured: " + localPath);

        Path parent = localPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        String url = baseUrl + filename;
        logger.info("Downloading {} -> {}", url, localPath);
        downloadUrlToFile(url, localPath);
    }

    // ========================================================================================
    // REMOTE CATALOG OPERATIONS
    // ========================================================================================

    @SuppressWarnings("unchecked")
    private Map<String, Map<String, String>> fetchRemoteCatalogRaw(String catalogUrl) {
        try {
            Path tempFile = Files.createTempFile("catalog-", ".tmp");
            try {
                downloadUrlToFile(catalogUrl, tempFile);
                return loadCatalogFromFile(tempFile);
            } finally {
                Files.deleteIfExists(tempFile);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to fetch dataset catalog from " + catalogUrl, e);
        }
    }

    private void saveCatalogLocally(Path localCatalog, String catalogUrl) {
        try {
            Files.createDirectories(localCacheDir);
            Path tempFile = Files.createTempFile(localCacheDir, "catalog-", ".tmp");
            downloadUrlToFile(catalogUrl, tempFile);
            Files.move(tempFile, localCatalog, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception e) {
            logger.warn("Failed to cache catalog locally: {}", e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, String>> loadCatalogFromFile(Path path) {
        try (InputStream in = Files.newInputStream(path)) {
            Map<String, Map<String, String>> result = new Yaml().load(in);
            return result != null ? result : Map.of();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load catalog from " + path, e);
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
                logger.warn("Remote catalog at {} differs from local catalog. Consider updating your local copy.", catalogUrl);
            }
        } catch (Exception e) {
            logger.warn("Could not check remote catalog for updates: {}", e.getMessage());
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
            throw new IllegalArgumentException("Unsupported URL scheme for download: " + url);
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

                if (downloadedSize != result.response().contentLength()) {
                    logger.error("Incomplete download (got {} of {} bytes). Retrying...", downloadedSize, result.response().contentLength());
                    Files.deleteIfExists(localPath);
                    continue;
                }

                downloaded = true;
                break;
            } catch (Exception e) {
                logger.error("Download attempt {} failed for {}: {}", i + 1, key, e.getMessage());
                Files.deleteIfExists(localPath);
            }
        }
        if (!downloaded) {
            throw new IOException("Failed to download " + s3Url + " after 3 attempts");
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
                throw new IOException("HTTP " + response.statusCode() + " downloading " + url);
            }
            Files.move(tempFile, localPath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Files.deleteIfExists(tempFile);
            throw new IOException("Interrupted downloading " + url, e);
        } catch (Exception e) {
            Files.deleteIfExists(tempFile);
            throw e;
        }
    }
}
