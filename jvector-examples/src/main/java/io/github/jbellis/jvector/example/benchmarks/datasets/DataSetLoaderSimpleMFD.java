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

import java.io.IOException;
import java.io.InputStream;
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

/// A dataset loader that works with fvec/ivec datasets described by a {@code datasets.yaml}
/// catalog. Supports both local-only and remote (HTTP) catalogs.
///
/// The constructor accepts the full URL (or local path) to a {@code datasets.yaml} file.
/// The base path for dataset files is derived by stripping the filename from that URL/path.
/// Filenames in the catalog are resolved relative to that base path.
///
/// ### Local-first with optional remote sync
///
/// If a local {@code datasets.yaml} already exists in the cache directory, it is used directly
/// and no download is needed. When {@code checkForUpdates} is enabled (the default), the loader
/// also fetches the remote catalog and warns if it differs from the local copy. This lets you
/// work fully offline with pre-populated local datasets while still being notified when the
/// remote catalog changes.
///
/// ### Example catalog format
/// ```yaml
/// ada002-100k:
///   base: ada_002_100k_base_99287.fvecs
///   query: ada_002_100k_query_10000.fvecs
///   gt: ada_002_100k_gt_ip_100.ivecs
/// ```
///
/// Given {@code catalogUrl = "https://bucket.s3.amazonaws.com/datasets-clean/datasets.yaml"},
/// the base vectors file would be fetched from:
/// ```
/// https://bucket.s3.amazonaws.com/datasets-clean/ada_002_100k_base_99287.fvecs
/// ```
///
/// Dataset metadata (similarity function, load behavior) is resolved from
/// {@code dataset_metadata.yml} via {@link DataSetMetadataReader}.
///
/// @see DataSetLoaderMFD
public class DataSetLoaderSimpleMFD implements DataSetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DataSetLoaderSimpleMFD.class);
    private static final String CATALOG_FILENAME = "datasets.yaml";
    private static final DataSetMetadataReader metadata = DataSetMetadataReader.load();

    private final String remoteBasePath;
    private final Map<String, Map<String, String>> catalog;
    private final Path localCacheDir;
    private final HttpClient httpClient;

    /// Creates a loader from the full URL to a remote {@code datasets.yaml} file.
    ///
    /// @param catalogUrl      the full URL to the datasets.yaml catalog
    ///                        (e.g. {@code "https://bucket.s3.amazonaws.com/path/datasets.yaml"})
    /// @param localCacheDir   the local directory to cache downloaded fvec/ivec files and the catalog
    /// @param checkForUpdates if true and a local catalog already exists, the remote catalog is
    ///                        fetched and compared; a warning is logged if they differ
    @SuppressWarnings("unchecked")
    public DataSetLoaderSimpleMFD(String catalogUrl, String localCacheDir, boolean checkForUpdates) {
        this.localCacheDir = Paths.get(localCacheDir);
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        // derive remote base path by stripping the filename from the catalog URL
        boolean isRemote = catalogUrl.startsWith("http://") || catalogUrl.startsWith("https://");
        int lastSlash = catalogUrl.lastIndexOf('/');
        if (lastSlash < 0) {
            throw new IllegalArgumentException("Catalog URL/path must contain a path separator: " + catalogUrl);
        }
        this.remoteBasePath = isRemote ? catalogUrl.substring(0, lastSlash + 1) : null;

        // check for a local catalog first
        Path localCatalog = this.localCacheDir.resolve(CATALOG_FILENAME);
        Map<String, Map<String, String>> localCatalogData = null;
        if (Files.exists(localCatalog)) {
            logger.info("Loading local dataset catalog from {}", localCatalog);
            localCatalogData = loadCatalogFromFile(localCatalog);
            logger.info("Local catalog contains {} datasets", localCatalogData.size());
        }

        if (isRemote) {
            if (localCatalogData != null) {
                // local catalog exists — use it, optionally check for updates
                this.catalog = localCatalogData;
                if (checkForUpdates) {
                    checkRemoteCatalogForUpdates(catalogUrl, localCatalogData);
                }
            } else {
                // no local catalog — must fetch from remote
                logger.info("No local catalog found, fetching from {}", catalogUrl);
                this.catalog = fetchRemoteCatalog(catalogUrl);
                saveCatalogLocally(localCatalog, catalogUrl);
                logger.info("Loaded and cached catalog with {} datasets", catalog.size());
            }
        } else {
            // local-only path
            if (localCatalogData != null) {
                this.catalog = localCatalogData;
            } else {
                // the catalogUrl itself is a local path
                logger.info("Loading dataset catalog from local path {}", catalogUrl);
                this.catalog = loadCatalogFromFile(Paths.get(catalogUrl));
                logger.info("Loaded catalog with {} datasets", catalog.size());
            }
        }
    }

    @Override
    public Optional<DataSetInfo> loadDataSet(String dataSetName) {
        var entry = catalog.get(dataSetName);
        if (entry == null) {
            logger.debug("Dataset '{}' not found in catalog", dataSetName);
            return Optional.empty();
        }

        var baseFile = entry.get("base");
        var queryFile = entry.get("query");
        var gtFile = entry.get("gt");
        if (baseFile == null || queryFile == null || gtFile == null) {
            logger.error("Dataset '{}' is missing required fields (base, query, gt) in catalog", dataSetName);
            return Optional.empty();
        }

        logger.info("Found dataset '{}' in catalog", dataSetName);

        // download missing files from remote, or verify they exist locally
        try {
            ensureFileAvailable(baseFile);
            ensureFileAvailable(queryFile);
            ensureFileAvailable(gtFile);
        } catch (Exception e) {
            throw new RuntimeException("Failed to obtain dataset files for " + dataSetName, e);
        }

        var props = metadata.getProperties(dataSetName)
                .orElseThrow(() -> new IllegalArgumentException(
                        "No metadata configured in dataset_metadata.yml for dataset: " + dataSetName));
        props.similarityFunction()
                .orElseThrow(() -> new IllegalArgumentException(
                        "No similarity_function configured in dataset_metadata.yml for dataset: " + dataSetName));

        return Optional.of(new DataSetInfo(props, () -> {
            var baseVectors = SiftLoader.readFvecs(localCacheDir.resolve(baseFile).toString());
            var queryVectors = SiftLoader.readFvecs(localCacheDir.resolve(queryFile).toString());
            var gtVectors = SiftLoader.readIvecs(localCacheDir.resolve(gtFile).toString());
            return DataSetUtils.processDataSet(dataSetName, props, baseVectors, queryVectors, gtVectors);
        }));
    }

    /// Ensures a dataset file is available locally. If the file already exists in the cache
    /// directory, it is used as-is. Otherwise, if a remote base path is configured, the file
    /// is downloaded via HTTP.
    private void ensureFileAvailable(String filename) throws IOException, InterruptedException {
        Path localPath = localCacheDir.resolve(filename);

        if (Files.exists(localPath)) {
            logger.debug("Using local file: {}", localPath);
            return;
        }

        if (remoteBasePath == null) {
            throw new IOException("File not found locally and no remote URL configured: " + localPath);
        }

        Files.createDirectories(localCacheDir);
        String url = remoteBasePath + filename;
        logger.info("Downloading {} -> {}", url, localPath);

        var request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .build();

        // download to a temp file first, then move atomically to avoid partial files
        Path tempFile = Files.createTempFile(localCacheDir, "download-", ".tmp");
        try {
            var response = httpClient.send(request, HttpResponse.BodyHandlers.ofFile(tempFile));
            if (response.statusCode() != 200) {
                throw new IOException("HTTP " + response.statusCode() + " downloading " + url);
            }
            long size = Files.size(tempFile);
            Files.move(tempFile, localPath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            logger.info("Downloaded {} ({} bytes)", filename, size);
        } catch (Exception e) {
            Files.deleteIfExists(tempFile);
            throw e;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, String>> loadCatalogFromFile(Path path) {
        try (InputStream in = Files.newInputStream(path)) {
            return new Yaml().load(in);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load catalog from " + path, e);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Map<String, String>> fetchRemoteCatalog(String catalogUrl) {
        try {
            var request = HttpRequest.newBuilder()
                    .uri(URI.create(catalogUrl))
                    .GET()
                    .build();
            var response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                throw new IOException("HTTP " + response.statusCode() + " fetching catalog: " + catalogUrl);
            }
            try (InputStream body = response.body()) {
                return new Yaml().load(body);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while fetching catalog: " + catalogUrl, e);
        } catch (IOException e) {
            throw new RuntimeException("Failed to fetch dataset catalog from " + catalogUrl, e);
        }
    }

    /// Saves the remote catalog to the local cache directory so subsequent runs can use it offline.
    private void saveCatalogLocally(Path localCatalog, String catalogUrl) {
        try {
            Files.createDirectories(localCacheDir);
            var request = HttpRequest.newBuilder()
                    .uri(URI.create(catalogUrl))
                    .GET()
                    .build();
            Path tempFile = Files.createTempFile(localCacheDir, "catalog-", ".tmp");
            var response = httpClient.send(request, HttpResponse.BodyHandlers.ofFile(tempFile));
            if (response.statusCode() != 200) {
                Files.deleteIfExists(tempFile);
                logger.warn("Failed to cache catalog locally (HTTP {})", response.statusCode());
            } else {
                Files.move(tempFile, localCatalog, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception e) {
            logger.warn("Failed to cache catalog locally: {}", e.getMessage());
        }
    }

    /// Fetches the remote catalog and compares it to the local one, logging a warning if they differ.
    private void checkRemoteCatalogForUpdates(String catalogUrl, Map<String, Map<String, String>> localCatalogData) {
        try {
            var remoteCatalogData = fetchRemoteCatalog(catalogUrl);
            if (!remoteCatalogData.equals(localCatalogData)) {
                logger.warn("Remote catalog at {} differs from local catalog. "
                        + "Local has {} datasets, remote has {} datasets. "
                        + "Consider updating your local copy.",
                        catalogUrl, localCatalogData.size(), remoteCatalogData.size());
            } else {
                logger.debug("Local catalog is up to date with remote");
            }
        } catch (Exception e) {
            logger.warn("Could not check remote catalog for updates: {}", e.getMessage());
        }
    }
}
