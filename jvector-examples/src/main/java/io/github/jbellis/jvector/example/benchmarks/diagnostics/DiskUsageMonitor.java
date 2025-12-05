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

package io.github.jbellis.jvector.example.benchmarks.diagnostics;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Utility class for monitoring disk usage during benchmark execution.
 * Tracks total disk space used and number of files created.
 */
public class DiskUsageMonitor {

    /**
     * Captures disk usage statistics for a given directory
     */
    public DiskUsageSnapshot captureSnapshot(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return new DiskUsageSnapshot(0, 0);
        }

        AtomicLong totalSize = new AtomicLong(0);
        AtomicLong fileCount = new AtomicLong(0);

        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                totalSize.addAndGet(attrs.size());
                fileCount.incrementAndGet();
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
                // Skip files that can't be accessed
                return FileVisitResult.CONTINUE;
            }
        });

        return new DiskUsageSnapshot(totalSize.get(), fileCount.get());
    }

    /**
     * Logs the difference between two disk usage snapshots
     */
    public void logDifference(String phase, DiskUsageSnapshot before, DiskUsageSnapshot after) {
        long sizeDiff = after.totalBytes - before.totalBytes;
        long filesDiff = after.fileCount - before.fileCount;

        System.out.printf("[%s] Disk Usage Changes:%n", phase);
        System.out.printf("  Total Size: %s (change: %s)%n",
            formatBytes(after.totalBytes),
            formatBytesDiff(sizeDiff));
        System.out.printf("  File Count: %d (change: %+d)%n",
            after.fileCount, filesDiff);
    }

    /**
     * Formats bytes into a human-readable string
     */
    public static String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.2f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        } else {
            return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    /**
     * Formats byte difference with sign
     */
    private static String formatBytesDiff(long bytes) {
        String sign = bytes >= 0 ? "+" : "";
        return sign + formatBytes(Math.abs(bytes));
    }

    /**
     * Data class representing disk usage at a point in time
     */
    public static class DiskUsageSnapshot {
        public final long totalBytes;
        public final long fileCount;

        public DiskUsageSnapshot(long totalBytes, long fileCount) {
            this.totalBytes = totalBytes;
            this.fileCount = fileCount;
        }

        public DiskUsageSnapshot subtract(DiskUsageSnapshot other) {
            return new DiskUsageSnapshot(
                this.totalBytes - other.totalBytes,
                this.fileCount - other.fileCount
            );
        }
    }
}

// Made with Bob
