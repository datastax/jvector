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

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Set;

public class DownloadHelper {
    private static final String bucketName = "astra-vector";
    private static final String infraBucketName = "jvector-datasets-infratest";
    private static final List<String> bucketNames = List.of(bucketName, infraBucketName);

    private static final String fvecDir = "fvec";
    private final static Set<String> infraDatasets = Set.of("dpr-1M", "dpr-10M", "cap-1M", "cap-6M", "cohere-english-v3-1M", "cohere-english-v3-10M");

    private static S3AsyncClientBuilder s3AsyncClientBuilder() {
        return S3AsyncClient.builder()
                .region(Region.US_EAST_1)
                .httpClient(AwsCrtAsyncHttpClient.builder()
                        .maxConcurrency(1)
                        .build())
                .credentialsProvider(AnonymousCredentialsProvider.create());
    }

    public static MultiFileDatasource maybeDownloadFvecs(String name) {
        String bucket = infraDatasets.contains(name) ? infraBucketName : bucketName;
        var mfd = MultiFileDatasource.byName.get(name);
        if (mfd == null) {
            throw new IllegalArgumentException("Unknown dataset: " + name);
        }
        // TODO how to detect and recover from incomplete downloads?

        // get directory from paths in keys
        try {
            Files.createDirectories(Paths.get(fvecDir).resolve(mfd.directory()));
        } catch (IOException e) {
            System.err.println("Failed to create directory: " + e.getMessage());
        }

        try (S3AsyncClient s3Client = s3AsyncClientBuilder().build()) {
            S3TransferManager tm = S3TransferManager.builder().s3Client(s3Client).build();
            for (var pathFragment : mfd.paths()) {
                Path localPath = Paths.get(fvecDir).resolve(pathFragment);
                if (Files.exists(localPath)) {
                    continue;
                }

                var urlPath = pathFragment.toString().replace('\\', '/');
                System.out.println("Downloading dataset: " + name);
                DownloadFileRequest downloadFileRequest =
                        DownloadFileRequest.builder()
                                .getObjectRequest(b -> b.bucket(bucket).key(urlPath))
                                .addTransferListener(LoggingTransferListener.create())
                                .destination(Paths.get(localPath.toString()))
                                .build();

                // 3 retries
                for (int i = 0; i < 3; i++) {
                    FileDownload downloadFile = tm.downloadFile(downloadFileRequest);
                    CompletedFileDownload downloadResult = downloadFile.completionFuture().join();
                    long downloadedSize = Files.size(localPath);

                    // Check if downloaded file size matches the expected size
                    if (downloadedSize == downloadResult.response().contentLength()) {
                        System.out.println("Downloaded file of length " + downloadedSize);
                        break;  // Successfully downloaded
                    } else {
                        System.out.println("Incomplete download. Retrying...");
                    }
                }
            }
            tm.close();
        } catch (Exception e) {
            System.out.println("Error downloading data from S3: " + e.getMessage());
            System.exit(1);
        }

        return mfd;
    }

    public static void maybeDownloadHdf5(String datasetName) {
        Path path = Path.of(Hdf5Loader.HDF5_DIR);
        var localPath = path.resolve(datasetName);
        if (Files.exists(localPath)) {
            return;
        }

        // Download from https://ann-benchmarks.com/datasetName
        var url = "https://ann-benchmarks.com/" + datasetName;
        System.out.println("Downloading: " + url);

        HttpURLConnection connection;
        while (true) {
            int responseCode;
            try {
                connection = (HttpURLConnection) new URL(url).openConnection();
                responseCode = connection.getResponseCode();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            if (responseCode == HttpURLConnection.HTTP_MOVED_PERM || responseCode == HttpURLConnection.HTTP_MOVED_TEMP) {
                String newUrl = connection.getHeaderField("Location");
                System.out.println("Redirect detected to URL: " + newUrl);
                url = newUrl;
            } else {
                break;
            }
        }

        try (InputStream in = connection.getInputStream()) {
            Files.createDirectories(path);
            Files.copy(in, localPath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            System.out.println("Error downloading data: " + e.getMessage());
            System.exit(1);
        }
    }
}
