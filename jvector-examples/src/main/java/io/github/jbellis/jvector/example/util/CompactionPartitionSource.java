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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.auth.credentials.AnonymousCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.transfer.s3.S3TransferManager;
import software.amazon.awssdk.transfer.s3.model.DownloadFileRequest;
import software.amazon.awssdk.transfer.s3.model.FileDownload;
import software.amazon.awssdk.transfer.s3.progress.LoggingTransferListener;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Resolves and downloads pre-built compaction partitions from S3 so regression tests can compact
 * them directly instead of rebuilding partitions on every run.
 *
 * <p>Partitions live under a public, anonymously-readable bucket using the layout:
 * <pre>
 *   s3://&lt;bucket&gt;/&lt;datasetName&gt;/&lt;numPartitions&gt;-&lt;distribution&gt;-&lt;precision&gt;/per-source-graph-&lt;i&gt;
 * </pre>
 * e.g. {@code s3://vector-compaction-partitions/cap-1M/4-UNIFORM-FUSEDPQ/per-source-graph-0}.
 *
 * <p>Downloaded files are cached locally and reused across configs and reruns; a file is
 * re-downloaded only if it is missing or empty.
 *
 * <p>Configuration via environment variables:
 * <ul>
 *   <li>{@code COMPACTION_PARTITIONS_BUCKET} — S3 bucket name (default {@code vector-compaction-partitions})</li>
 *   <li>{@code COMPACTION_PARTITIONS_CACHE_DIR} — local cache root (default {@code dataset_cache/compaction-partitions})</li>
 * </ul>
 */
public final class CompactionPartitionSource {
    private static final Logger logger = LoggerFactory.getLogger(CompactionPartitionSource.class);

    private static final String DEFAULT_BUCKET = "vector-compaction-partitions";
    private static final String DEFAULT_CACHE_DIR = "dataset_cache/compaction-partitions";
    private static final String PARTITION_FILE_PREFIX = "per-source-graph-";

    private static volatile S3TransferManager transferManager;

    private CompactionPartitionSource() {}

    /** Returns the configured S3 bucket name for compaction partitions. */
    public static String bucket() {
        String env = System.getenv("COMPACTION_PARTITIONS_BUCKET");
        return (env != null && !env.isBlank()) ? env.trim() : DEFAULT_BUCKET;
    }

    /** Returns the local cache root where downloaded partitions are stored. */
    public static Path cacheRoot() {
        String env = System.getenv("COMPACTION_PARTITIONS_CACHE_DIR");
        return Path.of((env != null && !env.isBlank()) ? env.trim() : DEFAULT_CACHE_DIR);
    }

    /**
     * Ensures the {@code numPartitions} partition graphs for {@code datasetName}/{@code configDir}
     * are present in the local cache, downloading any that are missing, and returns their local paths
     * in partition order (0..numPartitions-1).
     *
     * @param datasetName   logical dataset name (e.g. {@code "cap-1M"})
     * @param configDir     partition config directory (e.g. {@code "4-UNIFORM-FUSEDPQ"})
     * @param numPartitions number of partition graphs to fetch
     */
    public static List<Path> ensurePartitions(String datasetName, String configDir, int numPartitions) throws IOException {
        Path localDir = cacheRoot().resolve(datasetName).resolve(configDir);
        Files.createDirectories(localDir);

        List<Path> paths = new ArrayList<>(numPartitions);
        for (int i = 0; i < numPartitions; i++) {
            String fileName = PARTITION_FILE_PREFIX + i;
            Path localPath = localDir.resolve(fileName);
            if (Files.exists(localPath) && Files.size(localPath) > 0) {
                logger.info("Using cached partition {}", localPath.toAbsolutePath());
            } else {
                String key = datasetName + "/" + configDir + "/" + fileName;
                logger.info("Downloading partition s3://{}/{} -> {}", bucket(), key, localPath.toAbsolutePath());
                downloadFileS3(bucket(), key, localPath);
            }
            paths.add(localPath);
        }
        return paths;
    }

    private static synchronized S3TransferManager getS3TransferManager() {
        if (transferManager == null) {
            S3AsyncClient s3Client = S3AsyncClient.crtBuilder()
                    .region(Region.US_EAST_1)
                    .credentialsProvider(AnonymousCredentialsProvider.create())
                    .targetThroughputInGbps(10.0)
                    .minimumPartSizeInBytes(8L * 1024 * 1024)
                    .build();
            transferManager = S3TransferManager.builder().s3Client(s3Client).build();
        }
        return transferManager;
    }

    private static void downloadFileS3(String bucket, String key, Path localPath) throws IOException {
        S3TransferManager tm = getS3TransferManager();
        DownloadFileRequest request = DownloadFileRequest.builder()
                .getObjectRequest(b -> b.bucket(bucket).key(key))
                .addTransferListener(LoggingTransferListener.create())
                .destination(localPath)
                .build();

        IOException lastError = null;
        for (int attempt = 1; attempt <= 3; attempt++) {
            try {
                FileDownload download = tm.downloadFile(request);
                download.completionFuture().join();
                if (Files.size(localPath) <= 0) {
                    throw new IOException("Downloaded empty file for s3://" + bucket + "/" + key);
                }
                return;
            } catch (Exception e) {
                lastError = new IOException("Download attempt " + attempt + " failed for s3://" + bucket + "/" + key, e);
                logger.error("Download attempt {} failed for s3://{}/{}: {}", attempt, bucket, key, e.getMessage());
                Files.deleteIfExists(localPath);
            }
        }
        throw new IOException("Failed to download s3://" + bucket + "/" + key + " after 3 attempts", lastError);
    }
}
