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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.transfer.s3.S3TransferManager;
import software.amazon.awssdk.transfer.s3.model.CompletedFileDownload;
import software.amazon.awssdk.transfer.s3.model.DownloadFileRequest;
import software.amazon.awssdk.transfer.s3.model.FileDownload;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.*;

import static org.junit.Assert.*;

public class TestDatasetDownloadPublication {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test(timeout = 10000) public void incompleteS3DownloadIsNotVisibleInCache() throws Exception {
        var cache = folder.newFolder().toPath();
        var downloads = new Downloads(cache);
        var executor = Executors.newSingleThreadExecutor();
        try {
            var result = executor.submit(() -> { downloads.ensure(); return null; });
            var pending = downloads.next();
            assertArrayEquals(new byte[] {1}, Files.readAllBytes(pending.path));
            assertFalse("A cache hit must not expose a partial download", Files.exists(downloads.target));
            pending.complete(new byte[] {1, 2, 3}, 3);
            result.get(5, TimeUnit.SECONDS);
            assertArrayEquals(new byte[] {1, 2, 3}, Files.readAllBytes(downloads.target));
            assertEquals(1, fileCount(cache));
        } finally { downloads.abort(); executor.shutdownNow(); }
    }

    @Test(timeout = 10000) public void failedConcurrentS3DownloadCannotDeletePublishedFile() throws Exception {
        var cache = folder.newFolder().toPath();
        var downloads = new Downloads(cache);
        var executor = Executors.newFixedThreadPool(2);
        try {
            var slow = executor.submit(() -> { downloads.ensure(); return null; });
            var first = downloads.next();
            var fast = executor.submit(() -> { downloads.ensure(); return null; });
            var second = downloads.next();
            assertNotEquals(first.path, second.path);
            second.complete(new byte[] {4, 5, 6}, 3);
            fast.get(5, TimeUnit.SECONDS);
            first.fail();
            downloads.next().fail();
            downloads.next().fail();
            var failure = assertThrows(ExecutionException.class, () -> slow.get(5, TimeUnit.SECONDS));
            assertTrue(failure.getCause() instanceof IOException);
            assertArrayEquals(new byte[] {4, 5, 6}, Files.readAllBytes(downloads.target));
            assertEquals(1, fileCount(cache));
        } finally { downloads.abort(); executor.shutdownNow(); }
    }

    @Test(timeout = 10000) public void invalidS3LengthIsNotPublishedAndRetriesAreCleanedUp() throws Exception {
        var cache = folder.newFolder().toPath();
        var downloads = new Downloads(cache);
        var executor = Executors.newSingleThreadExecutor();
        try {
            var result = executor.submit(() -> { downloads.ensure(); return null; });
            for (int attempt = 0; attempt < 3; attempt++) {
                downloads.next().complete(new byte[] {1}, 2);
            }
            var failure = assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
            assertTrue(failure.getCause() instanceof IOException);
            assertFalse(Files.exists(downloads.target));
            assertEquals(0, fileCount(cache));
        } finally { downloads.abort(); executor.shutdownNow(); }
    }

    private static long fileCount(Path directory) throws IOException {
        try (var files = Files.list(directory)) { return files.count(); }
    }

    private static class Pending {
        final Path path;
        final CompletableFuture<CompletedFileDownload> future = new CompletableFuture<>();
        Pending(Path path) { this.path = path; }
        void complete(byte[] content, long expectedSize) throws IOException {
            Files.write(path, content);
            future.complete(CompletedFileDownload.builder()
                    .response(GetObjectResponse.builder().contentLength(expectedSize).build()).build());
        }
        void fail() { future.completeExceptionally(new IOException("Injected download failure")); }
    }

    private static class Downloads {
        final Path cache, target;
        final DataSetLoaderSimpleMFD loader;
        final BlockingQueue<Pending> pending = new LinkedBlockingQueue<>();
        final ConcurrentLinkedQueue<Pending> all = new ConcurrentLinkedQueue<>();
        volatile boolean aborted;
        Downloads(Path cache) throws Exception {
            this.cache = cache;
            target = cache.resolve("vectors.fvecs");
            Path metadata = cache.getParent().resolve("metadata.yml");
            Files.writeString(metadata, "test-ds:\n  similarity_function: DOT_PRODUCT\n  load_behavior: NO_SCRUB\n");
            loader = new DataSetLoaderSimpleMFD(null, cache.toString(), false,
                    DataSetMetadataReader.load(metadata.toString()));
            // Replace only the network transport; exercise the real cache and retry logic.
            var manager = (S3TransferManager) Proxy.newProxyInstance(S3TransferManager.class.getClassLoader(),
                    new Class<?>[] {S3TransferManager.class}, (proxy, method, args) -> {
                if (!method.getName().equals("downloadFile")) throw new UnsupportedOperationException(method.getName());
                var request = (DownloadFileRequest) args[0];
                var download = new Pending(request.destination());
                Files.write(download.path, new byte[] {1});
                all.add(download);
                pending.add(download);
                if (aborted) download.fail();
                return Proxy.newProxyInstance(FileDownload.class.getClassLoader(),
                        new Class<?>[] {FileDownload.class}, (file, call, arguments) -> {
                    if (call.getName().equals("completionFuture")) return download.future;
                    throw new UnsupportedOperationException(call.getName());
                });
            });
            var field = DataSetLoaderSimpleMFD.class.getDeclaredField("s3TransferManager");
            field.setAccessible(true);
            field.set(loader, manager);
        }
        void abort() {
            aborted = true;
            all.forEach(Pending::fail);
        }
        Pending next() throws Exception {
            var download = pending.poll(5, TimeUnit.SECONDS);
            assertNotNull("Expected a download attempt", download);
            return download;
        }
        void ensure() throws Exception {
            var method = DataSetLoaderSimpleMFD.class.getDeclaredMethod("ensureFileAvailable", String.class, Path.class, String.class);
            method.setAccessible(true);
            try { method.invoke(loader, "vectors.fvecs", cache, "s3://bucket/"); }
            catch (InvocationTargetException failure) { throw (Exception) failure.getCause(); }
        }
    }
}
