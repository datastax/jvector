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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.auth.credentials.AnonymousCredentialsProvider;
import software.amazon.awssdk.http.crt.AwsCrtAsyncHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3AsyncClientBuilder;
import software.amazon.awssdk.transfer.s3.S3TransferManager;
import software.amazon.awssdk.transfer.s3.model.CompletedFileDownload;
import software.amazon.awssdk.transfer.s3.model.DownloadFileRequest;
import software.amazon.awssdk.transfer.s3.model.FileDownload;
import software.amazon.awssdk.transfer.s3.progress.LoggingTransferListener;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * This dataset loader supports <i>multi-file</i> datasets which are comprised of several files as defined in
 * {@link DataSetLoaderMFD.MultiFileDatasource}.
 *
 * <p>The vector similarity function is determined by looking up the dataset name in
 * {@code dataset_metadata.yml} via {@link DataSetMetadataReader}. If no entry is found,
 * an error is thrown.
 */
public class DataSetLoaderMFD implements DataSetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DataSetLoaderMFD.class);

    private final static Set<String> infraDatasets = Set.of("dpr-1M", "dpr-10M", "cap-1M", "cap-6M", "cohere-english-v3-1M", "cohere-english-v3-10M");
    private static final String infraBucketName = "jvector-datasets-infratest";
    private static final String fvecDir = "fvec";
    private static final String bucketName = "astra-vector";
    private static final List<String> bucketNames = List.of(bucketName, infraBucketName);
    private static final DataSetMetadataReader metadata = DataSetMetadataReader.load();

    /**
     * {@inheritDoc}
     */
    public Optional<DataSetInfo> loadDataSet(String fileName) {
        return maybeDownloadFvecs(fileName).map(mfd -> {
            var props = metadata.getProperties(mfd.name)
                    .orElseThrow(() -> new IllegalArgumentException(
                            "No metadata configured in dataset_metadata.yml for MFD dataset: " + mfd.name));
            var vsf = props.similarityFunction()
                    .orElseThrow(() -> new IllegalArgumentException(
                            "No similarity_function configured in dataset_metadata.yml for MFD dataset: " + mfd.name));
            return new DataSetInfo(props, () -> mfd.load(vsf));
        });
    }

    /// Downloads the fvec/ivec files for the named dataset from S3 if not already present locally.
    ///
    /// @param name the logical dataset name
    /// @return the datasource descriptor, or empty if the name is not a known multi-file dataset
    private Optional<MultiFileDatasource> maybeDownloadFvecs(String name) {
        String bucket = infraDatasets.contains(name) ? infraBucketName : bucketName;
        var mfd = MultiFileDatasource.byName.get(name);
        if (mfd == null) {
            logger.debug("MultiFileDatasource not found for name: [" + name + "]");
            return Optional.empty();
        }
        logger.info("found dataset definition for {}", name);

        // TODO how to detect and recover from incomplete downloads?

        // get directory from paths in keys
        Path fvecPath = Paths.get(fvecDir);
        try {
            Files.createDirectories(fvecPath.resolve(mfd.directory()));
        } catch (IOException e) {
            throw new RuntimeException("Failed to create directory: " + fvecDir, e);
        }

        try (S3AsyncClient s3Client = s3AsyncClientBuilder().build()) {
            S3TransferManager tm = S3TransferManager.builder().s3Client(s3Client).build();
            for (var pathFragment : mfd.paths()) {
                Path localPath = fvecPath.resolve(pathFragment);
                if (Files.exists(localPath)) {
                    continue;
                }

                var urlPath = pathFragment.toString().replace('\\', '/');
                logger.info("Downloading dataset {} from {}", name, urlPath);
                DownloadFileRequest downloadFileRequest =
                        DownloadFileRequest.builder()
                                .getObjectRequest(b -> b.bucket(bucket).key(urlPath))
                                .addTransferListener(LoggingTransferListener.create())
                                .destination(Paths.get(localPath.toString()))
                                .build();

                // 3 retries
                boolean downloaded = false;
                for (int i = 0; i < 3; i++) {
                    try {
                        FileDownload downloadFile = tm.downloadFile(downloadFileRequest);
                        CompletedFileDownload downloadResult = downloadFile.completionFuture().join();
                        long downloadedSize = Files.size(localPath);

                        // Check if downloaded file size matches the expected size
                        if (downloadedSize != downloadResult.response().contentLength()) {
                            logger.error("Incomplete download (got {} of {} bytes). Retrying...",
                                    downloadedSize, downloadResult.response().contentLength());
                            Files.deleteIfExists(localPath);
                            continue;
                        }

                        // Validate the file header to catch corrupt downloads
                        if (!validateVecFileHeader(localPath)) {
                            logger.error("Downloaded file {} has an invalid header; deleting and retrying", urlPath);
                            Files.deleteIfExists(localPath);
                            continue;
                        }

                        logger.info("Downloaded file of length " + downloadedSize);
                        downloaded = true;
                        break;
                    } catch (Exception e) {
                        logger.error("Download attempt {} failed for {}: {}", i + 1, urlPath, e.getMessage());
                        Files.deleteIfExists(localPath);
                    }
                }
                if (!downloaded) {
                    throw new IOException("Failed to download " + urlPath + " after 3 attempts");
                }
            }
            tm.close();
        } catch (Exception e) {
            throw new RuntimeException("Error downloading data from S3: " + e.getMessage());
        }

        return Optional.of(mfd);
    }

    /// Reads the first 4 bytes of a vec file (fvecs or ivecs) and checks that the
    /// little-endian int32 dimension/count value is positive and reasonable.
    private static boolean validateVecFileHeader(Path path) {
        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile())))) {
            int dimension = Integer.reverseBytes(dis.readInt());
            return dimension > 0 && dimension <= 100_000;
        } catch (IOException e) {
            return false;
        }
    }

    /// Creates an S3 async client builder configured for anonymous access to US-EAST-1.
    private static S3AsyncClientBuilder s3AsyncClientBuilder() {
        return S3AsyncClient.builder()
                .region(Region.US_EAST_1)
                .httpClient(AwsCrtAsyncHttpClient.builder()
                        .maxConcurrency(16)
                        .build())
                .credentialsProvider(AnonymousCredentialsProvider.create());
    }

    /// Describes a dataset stored as three separate fvec/ivec files (base vectors, query
    /// vectors, and ground truth) in an S3 bucket. Known datasets are registered in {@link #byName}.
    public static class MultiFileDatasource {
        public final String name;
        public final Path basePath;
        public final Path queriesPath;
        public final Path groundTruthPath;
        private final static String DATASET_HASH = System.getenv("DATASET_HASH");

        public MultiFileDatasource(String name, String basePath, String queriesPath, String groundTruthPath) {
            this.name = name;
            this.basePath = Paths.get(basePath);
            this.queriesPath = Paths.get(queriesPath);
            this.groundTruthPath = Paths.get(groundTruthPath);
        }

        /// Returns the parent directory of the base vectors file.
        public Path directory() {
            return basePath.getParent();
        }

        /// Returns the three file paths (base, queries, ground truth) that comprise this dataset.
        public Iterable<Path> paths() {
            return List.of(basePath, queriesPath, groundTruthPath);
        }

        /// Reads the fvec/ivec files from disk and returns a scrubbed {@link DataSet}.
        ///
        /// @param similarityFunction the similarity function to associate with the dataset
        /// @return the loaded and scrubbed dataset
        public DataSet load(VectorSimilarityFunction similarityFunction) {
            var baseVectors = SiftLoader.readFvecs("fvec/" + basePath);
            var queryVectors = SiftLoader.readFvecs("fvec/" + queriesPath);
            var gtVectors = SiftLoader.readIvecs("fvec/" + groundTruthPath);
            return DataSetUtils.getScrubbedDataSet(name, similarityFunction, baseVectors, queryVectors, gtVectors);
        }

        public static Map<String, MultiFileDatasource> byName = new HashMap<>() {{
            put("degen-200k", new MultiFileDatasource("degen-200k",
                                                       "ada-degen/degen_base_vectors.fvec",
                                                       "ada-degen/degen_query_vectors.fvec",
                                                       "ada-degen/degen_ground_truth.ivec"));
            put("cohere-english-v3-100k", new MultiFileDatasource("cohere-english-v3-100k",
                                                                  "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec",
                                                                  "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec",
                                                                  "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100.ivec"));
            put("cohere-english-v3-1M", new MultiFileDatasource("cohere-english-v3-1M",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_base_1m_norm.fvecs",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_query_10k_norm.fvecs",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_gt_1m_ip_k100.ivecs"));
            put("cohere-english-v3-10M", new MultiFileDatasource("cohere-english-v3-10M",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_base_10m_norm.fvecs",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_query_10k_norm.fvecs",
                    DATASET_HASH + "/cohere/cohere_wikipedia_v3/cohere_wiki_en_flat_gt_10m_ip_k100.ivecs"));
            put("colbert-10M", new MultiFileDatasource("colbert-10M",
                                                       "wikipedia_squad/10M/colbertv2.0_128_base_vectors_10000000.fvec",
                                                       "wikipedia_squad/10M/colbertv2.0_128_query_vectors_100000.fvec",
                                                       "wikipedia_squad/10M/colbertv2.0_128_indices_b10000000_q100000_k100.ivec"));
            put("colbert-1M", new MultiFileDatasource("colbert-1M",
                                                       "wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec",
                                                       "wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec",
                                                       "wikipedia_squad/1M/colbertv2.0_128_indices_b1000000_q100000_k100.ivec"));
            put("nv-qa-v4-100k", new MultiFileDatasource("nv-qa-v4-100k",
                                                         "wikipedia_squad/100k/nvidia-nemo_1024_base_vectors_100000.fvec",
                                                         "wikipedia_squad/100k/nvidia-nemo_1024_query_vectors_10000.fvec",
                                                         "wikipedia_squad/100k/nvidia-nemo_1024_indices_b100000_q10000_k100.ivec"));
            put("openai-v3-large-3072-100k", new MultiFileDatasource("openai-v3-large-3072-100k",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_3072_100000_indices_query_10000.ivec"));
            put("openai-v3-large-1536-100k", new MultiFileDatasource("openai-v3-large-1536-100k",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_1536_100000_base_vectors.fvec",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_1536_100000_query_vectors_10000.fvec",
                                                                     "wikipedia_squad/100k/text-embedding-3-large_1536_100000_indices_query_10000.ivec"));
            put("openai-v3-small-100k", new MultiFileDatasource("openai-v3-small-100k",
                                                                "wikipedia_squad/100k/text-embedding-3-small_1536_100000_base_vectors.fvec",
                                                                "wikipedia_squad/100k/text-embedding-3-small_1536_100000_query_vectors_10000.fvec",
                                                                "wikipedia_squad/100k/text-embedding-3-small_1536_100000_indices_query_10000.ivec"));
            put("ada002-100k", new MultiFileDatasource("ada002-100k",
                                                       "wikipedia_squad/100k/ada_002_100000_base_vectors.fvec",
                                                       "wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec",
                                                       "wikipedia_squad/100k/ada_002_100000_indices_query_10000.ivec"));
            put("ada002-1M", new MultiFileDatasource("ada002-1M",
                                                     "wikipedia_squad/1M/ada_002_1000000_base_vectors.fvec",
                                                     "wikipedia_squad/1M/ada_002_1000000_query_vectors_10000.fvec",
                                                     "wikipedia_squad/1M/ada_002_1000000_indices_query_10000.ivec"));
            put("e5-small-v2-100k", new MultiFileDatasource("e5-small-v2-100k",
                                                            "wikipedia_squad/100k/intfloat_e5-small-v2_100000_base_vectors.fvec",
                                                            "wikipedia_squad/100k/intfloat_e5-small-v2_100000_query_vectors_10000.fvec",
                                                            "wikipedia_squad/100k/intfloat_e5-small-v2_100000_indices_query_10000.ivec"));
            put("e5-base-v2-100k", new MultiFileDatasource("e5-base-v2-100k",
                                                           "wikipedia_squad/100k/intfloat_e5-base-v2_100000_base_vectors.fvec",
                                                           "wikipedia_squad/100k/intfloat_e5-base-v2_100000_query_vectors_10000.fvec",
                                                           "wikipedia_squad/100k/intfloat_e5-base-v2_100000_indices_query_10000.ivec"));
            put("e5-large-v2-100k", new MultiFileDatasource("e5-large-v2-100k",
                                                            "wikipedia_squad/100k/intfloat_e5-large-v2_100000_base_vectors.fvec",
                                                            "wikipedia_squad/100k/intfloat_e5-large-v2_100000_query_vectors_10000.fvec",
                                                            "wikipedia_squad/100k/intfloat_e5-large-v2_100000_indices_query_10000.ivec"));
            put("gecko-100k", new MultiFileDatasource("gecko-100k",
                                                      "wikipedia_squad/100k/textembedding-gecko_100000_base_vectors.fvec",
                                                      "wikipedia_squad/100k/textembedding-gecko_100000_query_vectors_10000.fvec",
                                                      "wikipedia_squad/100k/textembedding-gecko_100000_indices_query_10000.ivec"));
            put("gecko-1M", new MultiFileDatasource("gecko-1M",
                    "wikipedia_squad/1M/textembedding-gecko_1000000_base_vectors.fvec",
                    "wikipedia_squad/1M/textembedding-gecko_1000000_query_vectors_10000.fvec",
                    "wikipedia_squad/1M/textembedding-gecko_1000000_indices_query_10000.ivec"));
            put("dpr-1M", new MultiFileDatasource("dpr-1M",
                    DATASET_HASH + "/dpr/c4-en_base_1M_norm_files0_2.fvecs",
                    DATASET_HASH + "/dpr/c4-en_query_10k_norm_files0_1.fvecs",
                    DATASET_HASH + "/dpr/dpr_1m_gt_norm_ip_k100.ivecs"));
            put("dpr-10M", new MultiFileDatasource("dpr-10M",
                    DATASET_HASH + "/dpr/c4-en_base_10M_norm_files0_2.fvecs",
                    DATASET_HASH + "/dpr/c4-en_query_10k_norm_files0_1.fvecs",
                    DATASET_HASH + "/dpr/dpr_10m_gt_norm_ip_k100.ivecs"));
            put("cap-1M", new MultiFileDatasource("cap-1M",
                    DATASET_HASH + "/cap/Caselaw_gte-Qwen2-1.5B_embeddings_base_1m_norm_shuffle.fvecs",
                    DATASET_HASH + "/cap/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs",
                    DATASET_HASH + "/cap/cap_1m_gt_norm_shuffle_ip_k100.ivecs"));
            put("cap-6M", new MultiFileDatasource("cap-6M",
                    DATASET_HASH + "/cap/Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs",
                    DATASET_HASH + "/cap/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs",
                    DATASET_HASH + "/cap/cap_6m_gt_norm_shuffle_ip_k100.ivecs"));
        }};
    }
}
