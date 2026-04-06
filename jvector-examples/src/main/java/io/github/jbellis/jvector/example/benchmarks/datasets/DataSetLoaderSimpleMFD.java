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
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/// A dataset loader that works with fvec/ivec datasets described by a {@code catalog_entries.yaml}
/// catalog. Supports S3, HTTP, local-only, and combined remote+local modes.
///
/// ### Catalog format
///
/// The {@code catalog_entries.yaml} file lists datasets with their base, query, and ground truth files:
/// ```yaml
/// ada002-100k:
///   base: ada_002_100k_base_99287.fvecs
///   query: ada_002_100k_query_10000.fvecs
///   gt: ada_002_100k_gt_ip_100.ivecs
/// ```
/// Filenames are resolved relative to the same directory as the catalog (remote base path or
/// local cache directory).
///
/// ### Usage patterns
///
/// **Remote with local caching** — files are downloaded on first use and cached locally.
/// Subsequent runs use cached files. Set {@code checkForUpdates=true} to be warned when the
/// remote catalog changes. Supports both HTTP and S3 URLs.
/// ```java
/// var loader = new DataSetLoaderSimpleMFD(
///     "https://bucket.s3.amazonaws.com/datasets-clean/catalog_entries.yaml",
///     "fvec/catalog_entries.yaml",    // local cache path
///     true                            // warn if remote catalog differs from local
/// );
/// ```
///
/// **Local-only with pre-populated files** — no remote access at all. The local directory must
/// contain both {@code catalog_entries.yaml} and all referenced fvec/ivec files. Pass null or empty
/// string for the catalog URL, or use the single-arg constructor.
/// ```java
/// var loader = new DataSetLoaderSimpleMFD("local_datasets/catalog_entries.yaml");
/// ```
///
/// **Remote+local hybrid** — if the local directory already contains {@code catalog_entries.yaml}
/// and data files, they are used as-is. Missing data files are downloaded from the remote.
/// This allows pre-populating some datasets locally while fetching others on demand.
/// ```java
/// var loader = new DataSetLoaderSimpleMFD(
///     "https://bucket.s3.amazonaws.com/datasets-clean/catalog_entries.yaml",
///     "/data/datasets/catalog_entries.yaml",  // may already contain some files
///     true                                    // check if remote catalog has changed
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
    private static final String CATALOG_FILENAME = "catalog_entries.yaml";

    private final String remoteBasePath;
    private final Map<String, Map<String, String>> catalog;
    private final Path localCacheDir;
    private final DataSetMetadataReader metadata;
    private final HttpClient httpClient;

    // S3 instances for connection pooling
    private S3AsyncClient s3Client;
    private S3TransferManager s3TransferManager;

    /// Creates a local-only loader. No remote access is performed.
    ///
    /// The {@code localPath} may be either a directory (in which case {@code catalog_entries.yaml}
    /// is assumed inside it) or the full path to a catalog YAML file (in which case the parent
    /// directory is used for data files).
    ///
    /// If the path or catalog file does not exist, the loader is constructed successfully
    /// but will return empty for all dataset lookups. This allows it to be safely registered
    /// in a loader list without failing when local datasets are not present.
    ///
    /// @param localPath the local directory or full path to a catalog YAML file
    public DataSetLoaderSimpleMFD(String localPath) {
        this(null, localPath, false, DataSetMetadataReader.load());
    }

    /// Creates a loader using the default dataset metadata from {@code dataset_metadata.yml}.
    ///
    /// The {@code localPath} may be either a directory or the full path to a catalog YAML file.
    /// If it ends in {@code .yaml} or {@code .yml}, the parent directory is used as the cache
    /// directory and that file is used as the catalog. Otherwise, {@code catalog_entries.yaml}
    /// is assumed inside the given directory.
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
    /// The {@code localPath} may be either a directory or the full path to a catalog YAML file.
    /// If it ends in {@code .yaml} or {@code .yml}, the parent directory is used as the cache
    /// directory and that file is used as the catalog. Otherwise, {@code catalog_entries.yaml}
    /// is assumed inside the given directory.
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
            localCatalog = resolvedPath.resolve(CATALOG_FILENAME);
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

        // try to load the local catalog
        Map<String, Map<String, String>> localCatalogData = null;
        if (Files.exists(localCatalog)) {
            logger.info("Loading local dataset catalog from {}", localCatalog);
            localCatalogData = loadCatalogFromFile(localCatalog);
        }

        if (isRemote) {
            if (localCatalogData != null) {
                this.catalog = localCatalogData;
                if (checkForUpdates) checkRemoteCatalogForUpdates(catalogUrl, localCatalogData);
            } else {
                logger.info("No local catalog found, fetching from {}", catalogUrl);
                this.catalog = fetchRemoteCatalog(catalogUrl);
                saveCatalogLocally(localCatalog, catalogUrl);
            }
        } else {
            // local-only mode (catalogUrl is null, empty, or a non-remote path)
            if (localCatalogData != null) {
                this.catalog = localCatalogData;
            } else {
                logger.info("No catalog found at {}. This loader will not match any datasets.", localCatalog);
                this.catalog = Map.of();
            }
        }
    }

    @Override
    public Optional<DataSetInfo> loadDataSet(String dataSetName) {
        var entry = catalog.get(dataSetName);
        if (entry == null) return Optional.empty();

        var baseFile = entry.get("base");
        var queryFile = entry.get("query");
        var gtFile = entry.get("gt");
        if (baseFile == null || queryFile == null || gtFile == null) {
            logger.error("Dataset '{}' is missing required fields (base, query, gt) in catalog", dataSetName);
            return Optional.empty();
        }

        logger.info("Found dataset '{}' in catalog", dataSetName);
        var startTime = System.nanoTime();

        // Execute downloads simultaneously to maximize network bandwidth
        try {
            var f1 = CompletableFuture.runAsync(() -> ensureQuietly(baseFile));
            var f2 = CompletableFuture.runAsync(() -> ensureQuietly(queryFile));
            var f3 = CompletableFuture.runAsync(() -> ensureQuietly(gtFile));

            CompletableFuture.allOf(f1, f2, f3).join();
        } catch (Exception e) {
            throw new RuntimeException("Failed to obtain dataset files for " + dataSetName, e);
        }

        System.out.printf("Total elapsed time (s): %.2f\n", (System.nanoTime() - startTime) / 1e9);

        var props = metadata.getProperties(dataSetName).orElseThrow();
        return Optional.of(new DataSetInfo(props, () -> {
            var baseVectors = SiftLoader.readFvecs(localCacheDir.resolve(baseFile).toString());
            var queryVectors = SiftLoader.readFvecs(localCacheDir.resolve(queryFile).toString());
            var gtVectors = SiftLoader.readIvecs(localCacheDir.resolve(gtFile).toString());
            return DataSetUtils.processDataSet(dataSetName, props, baseVectors, queryVectors, gtVectors);
        }));
    }

    // ========================================================================================
    // PRIMARY FILE AVAILABILITY & CATALOG LOGIC
    // ========================================================================================

    private void ensureQuietly(String filename) {
        try {
            ensureFileAvailable(filename);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void ensureFileAvailable(String filename) throws IOException {
        Path localPath = localCacheDir.resolve(filename);
        if (Files.exists(localPath)) return;
        if (remoteBasePath == null) throw new IOException("File not found locally and no remote URL configured: " + localPath);

        Path parent = localPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        String url = remoteBasePath + filename;
        logger.info("Downloading {} -> {}", url, localPath);
        downloadUrlToFile(url, localPath);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Map<String, String>> fetchRemoteCatalog(String catalogUrl) {
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
    private void checkRemoteCatalogForUpdates(String catalogUrl, Map<String, Map<String, String>> localCatalogData) {
        try {
            var remoteCatalogData = fetchRemoteCatalog(catalogUrl);
            if (!remoteCatalogData.equals(localCatalogData)) {
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
