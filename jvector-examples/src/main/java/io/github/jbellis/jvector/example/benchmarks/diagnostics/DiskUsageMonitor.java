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
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static java.nio.file.StandardWatchEventKinds.*;

/**
 * Event-driven disk usage monitor that uses WatchService to track filesystem changes.
 * Maintains running totals updated incrementally, avoiding expensive directory traversals
 * on every snapshot. This minimizes I/O overhead and prevents monitoring from interfering
 * with benchmark measurements.
 * 
 * <p>Usage:
 * <pre>
 * try (DiskUsageMonitor monitor = new DiskUsageMonitor()) {
 *     monitor.start(directory);
 *     // ... run benchmarks ...
 *     DiskUsageSnapshot snapshot = monitor.captureSnapshot();
 * }
 * </pre>
 */
public class DiskUsageMonitor implements AutoCloseable {
    
    // Event processing
    private WatchService watchService;
    private Thread watchThread;
    private volatile boolean running;
    
    // Current state (thread-safe)
    private final AtomicLong totalBytes = new AtomicLong(0);
    private final AtomicLong fileCount = new AtomicLong(0);
    
    // Directory and file tracking
    private final Map<WatchKey, Path> watchKeyToPath = new ConcurrentHashMap<>();
    private final Map<Path, Long> fileSizeCache = new ConcurrentHashMap<>();
    private Path rootDirectory;
    
    // Monitoring state
    private volatile boolean started = false;
    
    /**
     * Starts monitoring the specified directory for filesystem changes.
     * Performs an initial scan to establish baseline, then monitors changes incrementally.
     * 
     * @param directory the directory to monitor
     * @throws IOException if unable to start monitoring
     * @throws IllegalStateException if already started
     */
    public void start(Path directory) throws IOException {
        if (started) {
            throw new IllegalStateException("Monitor already started");
        }
        
        if (!Files.exists(directory)) {
            // Directory doesn't exist yet, initialize with zero values
            started = true;
            return;
        }
        
        this.rootDirectory = directory;
        this.watchService = FileSystems.getDefault().newWatchService();
        
        // Perform initial scan to establish baseline
        performInitialScan(directory);
        
        // Register watchers recursively
        registerRecursive(directory);
        
        // Start event processing thread
        running = true;
        watchThread = new Thread(this::processEvents, "DiskUsageMonitor-" + directory.getFileName());
        watchThread.setDaemon(true);
        watchThread.start();
        
        started = true;
    }
    
    /**
     * Captures a snapshot of current disk usage.
     * This is an O(1) operation that returns cached values, unlike the previous
     * implementation which performed full directory traversal.
     * 
     * @return snapshot of current disk usage
     */
    public DiskUsageSnapshot captureSnapshot() {
        return new DiskUsageSnapshot(totalBytes.get(), fileCount.get());
    }
    
    /**
     * Captures disk usage for a directory without starting continuous monitoring.
     * This is a fallback method for compatibility with the old API.
     * 
     * @param directory the directory to scan
     * @return snapshot of disk usage
     * @throws IOException if unable to scan directory
     */
    public DiskUsageSnapshot captureSnapshot(Path directory) throws IOException {
        if (started && directory.equals(rootDirectory)) {
            // Use cached values if monitoring this directory
            return captureSnapshot();
        }
        
        // Fallback to one-time scan for compatibility
        return performOneTimeScan(directory);
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
     * Stops monitoring and releases resources.
     */
    @Override
    public void close() throws IOException {
        if (!started) {
            return;
        }
        
        running = false;
        
        if (watchThread != null) {
            watchThread.interrupt();
            try {
                watchThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        if (watchService != null) {
            watchService.close();
        }
        
        watchKeyToPath.clear();
        fileSizeCache.clear();
        started = false;
    }
    
    // ========== Private Implementation ==========
    
    /**
     * Performs initial directory scan to establish baseline metrics
     */
    private void performInitialScan(Path directory) throws IOException {
        AtomicLong size = new AtomicLong(0);
        AtomicLong count = new AtomicLong(0);
        
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                long fileSize = attrs.size();
                size.addAndGet(fileSize);
                count.incrementAndGet();
                fileSizeCache.put(file, fileSize);
                return FileVisitResult.CONTINUE;
            }
            
            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                // Skip files that can't be accessed
                return FileVisitResult.CONTINUE;
            }
        });
        
        totalBytes.set(size.get());
        fileCount.set(count.get());
    }
    
    /**
     * Performs one-time scan without caching (fallback for compatibility)
     */
    private DiskUsageSnapshot performOneTimeScan(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return new DiskUsageSnapshot(0, 0);
        }
        
        AtomicLong size = new AtomicLong(0);
        AtomicLong count = new AtomicLong(0);
        
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                size.addAndGet(attrs.size());
                count.incrementAndGet();
                return FileVisitResult.CONTINUE;
            }
            
            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });
        
        return new DiskUsageSnapshot(size.get(), count.get());
    }
    
    /**
     * Registers watchers for a directory and all its subdirectories
     */
    private void registerRecursive(Path directory) throws IOException {
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                WatchKey key = dir.register(watchService, ENTRY_CREATE, ENTRY_DELETE, ENTRY_MODIFY);
                watchKeyToPath.put(key, dir);
                return FileVisitResult.CONTINUE;
            }
            
            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });
    }
    
    /**
     * Event processing loop - runs in background thread
     */
    private void processEvents() {
        while (running) {
            WatchKey key;
            try {
                key = watchService.poll(100, TimeUnit.MILLISECONDS);
                if (key == null) {
                    continue;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (ClosedWatchServiceException e) {
                break;
            }
            
            Path dir = watchKeyToPath.get(key);
            if (dir == null) {
                key.reset();
                continue;
            }
            
            for (WatchEvent<?> event : key.pollEvents()) {
                WatchEvent.Kind<?> kind = event.kind();
                
                if (kind == OVERFLOW) {
                    // Event overflow - too many events, may need to rescan
                    continue;
                }
                
                @SuppressWarnings("unchecked")
                WatchEvent<Path> ev = (WatchEvent<Path>) event;
                Path filename = ev.context();
                Path fullPath = dir.resolve(filename);
                
                try {
                    if (kind == ENTRY_CREATE) {
                        handleCreate(fullPath);
                    } else if (kind == ENTRY_DELETE) {
                        handleDelete(fullPath);
                    } else if (kind == ENTRY_MODIFY) {
                        handleModify(fullPath);
                    }
                } catch (IOException e) {
                    // Log but continue processing other events
                    System.err.printf("Error processing event %s for %s: %s%n", 
                        kind.name(), fullPath, e.getMessage());
                }
            }
            
            boolean valid = key.reset();
            if (!valid) {
                watchKeyToPath.remove(key);
            }
        }
    }
    
    /**
     * Handles file/directory creation events
     */
    private void handleCreate(Path path) throws IOException {
        if (!Files.exists(path)) {
            return; // File may have been deleted before we could process
        }
        
        if (Files.isDirectory(path)) {
            // Register watcher for new directory
            WatchKey key = path.register(watchService, ENTRY_CREATE, ENTRY_DELETE, ENTRY_MODIFY);
            watchKeyToPath.put(key, path);
            
            // Scan new directory for existing files
            Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    long size = attrs.size();
                    fileSizeCache.put(file, size);
                    totalBytes.addAndGet(size);
                    fileCount.incrementAndGet();
                    return FileVisitResult.CONTINUE;
                }
                
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                    if (!dir.equals(path)) {
                        // Register watchers for subdirectories
                        WatchKey key = dir.register(watchService, ENTRY_CREATE, ENTRY_DELETE, ENTRY_MODIFY);
                        watchKeyToPath.put(key, dir);
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } else if (Files.isRegularFile(path)) {
            long size = Files.size(path);
            fileSizeCache.put(path, size);
            totalBytes.addAndGet(size);
            fileCount.incrementAndGet();
        }
    }
    
    /**
     * Handles file/directory deletion events
     */
    private void handleDelete(Path path) {
        Long size = fileSizeCache.remove(path);
        if (size != null) {
            totalBytes.addAndGet(-size);
            fileCount.decrementAndGet();
        }
        // Note: For directories, we rely on individual file deletion events
        // rather than trying to recursively process the deleted directory
    }
    
    /**
     * Handles file modification events
     */
    private void handleModify(Path path) throws IOException {
        if (!Files.exists(path) || !Files.isRegularFile(path)) {
            return;
        }
        
        long newSize = Files.size(path);
        Long oldSize = fileSizeCache.put(path, newSize);
        
        if (oldSize != null) {
            long delta = newSize - oldSize;
            totalBytes.addAndGet(delta);
        } else {
            // File wasn't in cache (shouldn't happen, but handle gracefully)
            totalBytes.addAndGet(newSize);
            fileCount.incrementAndGet();
        }
    }
    
    // ========== Utility Methods ==========
    
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

