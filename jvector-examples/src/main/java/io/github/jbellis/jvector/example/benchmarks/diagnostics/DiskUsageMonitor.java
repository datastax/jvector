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
    
    // Multi-directory tracking
    private final Map<String, DirectoryStats> directoryStats = new ConcurrentHashMap<>();
    
    // Directory and file tracking
    private final Map<WatchKey, Path> watchKeyToPath = new ConcurrentHashMap<>();
    private final Map<Path, DirectoryFileInfo> fileSizeCache = new ConcurrentHashMap<>();
    
    // Monitoring state
    private volatile boolean started = false;
    
    /**
     * Internal class to track statistics for a single directory
     */
    private static class DirectoryStats {
        final String label;
        final Path path;
        final AtomicLong totalBytes = new AtomicLong(0);
        final AtomicLong fileCount = new AtomicLong(0);
        
        DirectoryStats(String label, Path path) {
            this.label = label;
            this.path = path;
        }
    }
    
    /**
     * Internal class to track which directory a file belongs to
     */
    private static class DirectoryFileInfo {
        final String directoryLabel;
        final long size;
        
        DirectoryFileInfo(String directoryLabel, long size) {
            this.directoryLabel = directoryLabel;
            this.size = size;
        }
    }
    
    /**
     * Starts monitoring the specified directory for filesystem changes.
     * Performs an initial scan to establish baseline, then monitors changes incrementally.
     *
     * @param directory the directory to monitor
     * @throws IOException if unable to start monitoring
     * @throws IllegalStateException if already started
     * @deprecated Use {@link #startMonitoring(String, Path)} instead
     */
    @Deprecated
    public void start(Path directory) throws IOException {
        startMonitoring("default", directory);
    }
    
    /**
     * Starts monitoring a single labeled directory for filesystem changes.
     * Performs an initial scan to establish baseline, then monitors changes incrementally.
     *
     * @param label a label to identify this directory in reports
     * @param directory the directory to monitor
     * @throws IOException if unable to start monitoring
     * @throws IllegalStateException if already started
     */
    public void startMonitoring(String label, Path directory) throws IOException {
        if (started) {
            throw new IllegalStateException("Monitor already started. Use addDirectory() to add more directories.");
        }
        
        this.watchService = FileSystems.getDefault().newWatchService();
        
        addDirectory(label, directory);
        
        // Start event processing thread
        running = true;
        watchThread = new Thread(this::processEvents, "DiskUsageMonitor");
        watchThread.setDaemon(true);
        watchThread.start();
        
        started = true;
    }
    
    /**
     * Adds an additional directory to monitor. Must be called after startMonitoring().
     *
     * @param label a label to identify this directory in reports
     * @param directory the directory to monitor
     * @throws IOException if unable to monitor the directory
     * @throws IllegalStateException if not yet started
     */
    public void addDirectory(String label, Path directory) throws IOException {
        if (!started && watchService == null) {
            throw new IllegalStateException("Must call startMonitoring() before addDirectory()");
        }
        
        if (directoryStats.containsKey(label)) {
            throw new IllegalArgumentException("Directory with label '" + label + "' already being monitored");
        }
        
        DirectoryStats stats = new DirectoryStats(label, directory);
        directoryStats.put(label, stats);
        
        if (!Files.exists(directory)) {
            // Directory doesn't exist yet, initialize with zero values
            return;
        }
        
        // Perform initial scan to establish baseline
        performInitialScan(label, directory, stats);
        
        // Register watchers recursively
        registerRecursive(directory, label);
    }
    
    /**
     * Captures a snapshot of current disk usage across all monitored directories.
     * This is an O(1) operation that returns cached values.
     *
     * @return snapshot of current disk usage for all directories
     */
    public MultiDirectorySnapshot captureSnapshot() {
        Map<String, DiskUsageSnapshot> snapshots = new java.util.HashMap<>();
        for (Map.Entry<String, DirectoryStats> entry : directoryStats.entrySet()) {
            DirectoryStats stats = entry.getValue();
            snapshots.put(entry.getKey(), new DiskUsageSnapshot(stats.totalBytes.get(), stats.fileCount.get()));
        }
        return new MultiDirectorySnapshot(snapshots);
    }
    
    /**
     * Captures disk usage for a specific labeled directory.
     *
     * @param label the label of the directory to capture
     * @return snapshot of disk usage for the specified directory, or null if not found
     */
    public DiskUsageSnapshot captureSnapshot(String label) {
        DirectoryStats stats = directoryStats.get(label);
        if (stats == null) {
            return null;
        }
        return new DiskUsageSnapshot(stats.totalBytes.get(), stats.fileCount.get());
    }
    
    /**
     * Captures disk usage for a directory without starting continuous monitoring.
     * This is a fallback method for compatibility with the old API.
     *
     * @param directory the directory to scan
     * @return snapshot of disk usage
     * @throws IOException if unable to scan directory
     * @deprecated Use labeled monitoring instead
     */
    @Deprecated
    public DiskUsageSnapshot captureSnapshot(Path directory) throws IOException {
        // Check if this directory is being monitored
        for (Map.Entry<String, DirectoryStats> entry : directoryStats.entrySet()) {
            if (entry.getValue().path.equals(directory)) {
                return captureSnapshot(entry.getKey());
            }
        }
        
        // Fallback to one-time scan for compatibility
        return performOneTimeScan(directory);
    }
    
    /**
     * Logs the difference between two multi-directory snapshots
     */
    public void logDifference(String phase, MultiDirectorySnapshot before, MultiDirectorySnapshot after) {
        System.out.printf("[%s] Disk Usage Changes:%n", phase);
        
        for (String label : after.snapshots.keySet()) {
            DiskUsageSnapshot beforeSnap = before.snapshots.get(label);
            DiskUsageSnapshot afterSnap = after.snapshots.get(label);
            
            if (beforeSnap == null) {
                // New directory added
                System.out.printf("  [%s] (new): %s, %d files%n",
                    label,
                    formatBytes(afterSnap.totalBytes),
                    afterSnap.fileCount);
            } else {
                long sizeDiff = afterSnap.totalBytes - beforeSnap.totalBytes;
                long filesDiff = afterSnap.fileCount - beforeSnap.fileCount;
                
                System.out.printf("  [%s] Size: %s (change: %s), Files: %d (change: %+d)%n",
                    label,
                    formatBytes(afterSnap.totalBytes),
                    formatBytesDiff(sizeDiff),
                    afterSnap.fileCount,
                    filesDiff);
            }
        }
    }
    
    /**
     * Logs the difference between two single-directory snapshots (legacy method)
     * @deprecated Use {@link #logDifference(String, MultiDirectorySnapshot, MultiDirectorySnapshot)} instead
     */
    @Deprecated
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
    private void performInitialScan(String label, Path directory, DirectoryStats stats) throws IOException {
        AtomicLong size = new AtomicLong(0);
        AtomicLong count = new AtomicLong(0);
        
        Files.walkFileTree(directory, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                long fileSize = attrs.size();
                size.addAndGet(fileSize);
                count.incrementAndGet();
                fileSizeCache.put(file, new DirectoryFileInfo(label, fileSize));
                return FileVisitResult.CONTINUE;
            }
            
            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                // Skip files that can't be accessed
                return FileVisitResult.CONTINUE;
            }
        });
        
        stats.totalBytes.set(size.get());
        stats.fileCount.set(count.get());
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
        
        Files.walkFileTree(directory, new SimpleFileVisitor<>() {
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
    private void registerRecursive(Path directory, String label) throws IOException {
        Files.walkFileTree(directory, new SimpleFileVisitor<>() {
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
     * Determines which directory label a path belongs to
     */
    private String getDirectoryLabel(Path path) {
        for (Map.Entry<String, DirectoryStats> entry : directoryStats.entrySet()) {
            if (path.startsWith(entry.getValue().path)) {
                return entry.getKey();
            }
        }
        return null;
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
        
        String label = getDirectoryLabel(path);
        if (label == null) {
            return; // Path not under any monitored directory
        }
        
        DirectoryStats stats = directoryStats.get(label);
        
        if (Files.isDirectory(path)) {
            // Register watcher for new directory
            WatchKey key = path.register(watchService, ENTRY_CREATE, ENTRY_DELETE, ENTRY_MODIFY);
            watchKeyToPath.put(key, path);
            
            // Scan new directory for existing files
            Files.walkFileTree(path, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    long size = attrs.size();
                    fileSizeCache.put(file, new DirectoryFileInfo(label, size));
                    stats.totalBytes.addAndGet(size);
                    stats.fileCount.incrementAndGet();
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
            fileSizeCache.put(path, new DirectoryFileInfo(label, size));
            stats.totalBytes.addAndGet(size);
            stats.fileCount.incrementAndGet();
        }
    }
    
    /**
     * Handles file/directory deletion events
     */
    private void handleDelete(Path path) {
        DirectoryFileInfo info = fileSizeCache.remove(path);
        if (info != null) {
            DirectoryStats stats = directoryStats.get(info.directoryLabel);
            if (stats != null) {
                stats.totalBytes.addAndGet(-info.size);
                stats.fileCount.decrementAndGet();
            }
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
        
        String label = getDirectoryLabel(path);
        if (label == null) {
            return; // Path not under any monitored directory
        }
        
        DirectoryStats stats = directoryStats.get(label);
        long newSize = Files.size(path);
        DirectoryFileInfo oldInfo = fileSizeCache.put(path, new DirectoryFileInfo(label, newSize));
        
        if (oldInfo != null) {
            long delta = newSize - oldInfo.size;
            stats.totalBytes.addAndGet(delta);
        } else {
            // File wasn't in cache (shouldn't happen, but handle gracefully)
            stats.totalBytes.addAndGet(newSize);
            stats.fileCount.incrementAndGet();
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
    /**
     * Data class representing disk usage across multiple directories
     */
    public static class MultiDirectorySnapshot {
        public final Map<String, DiskUsageSnapshot> snapshots;

        public MultiDirectorySnapshot(Map<String, DiskUsageSnapshot> snapshots) {
            this.snapshots = new java.util.HashMap<>(snapshots);
        }

        /**
         * Get the snapshot for a specific directory label
         */
        public DiskUsageSnapshot get(String label) {
            return snapshots.get(label);
        }

        /**
         * Get total bytes across all directories
         */
        public long getTotalBytes() {
            return snapshots.values().stream()
                    .mapToLong(s -> s.totalBytes)
                    .sum();
        }

        /**
         * Get total file count across all directories
         */
        public long getTotalFileCount() {
            return snapshots.values().stream()
                    .mapToLong(s -> s.fileCount)
                    .sum();
        }

        /**
         * Subtract another multi-directory snapshot from this one
         */
        public MultiDirectorySnapshot subtract(MultiDirectorySnapshot other) {
            Map<String, DiskUsageSnapshot> result = new java.util.HashMap<>();
            for (Map.Entry<String, DiskUsageSnapshot> entry : snapshots.entrySet()) {
                String label = entry.getKey();
                DiskUsageSnapshot thisSnap = entry.getValue();
                DiskUsageSnapshot otherSnap = other.snapshots.get(label);
                if (otherSnap != null) {
                    result.put(label, thisSnap.subtract(otherSnap));
                } else {
                    result.put(label, thisSnap);
                }
            }
            return new MultiDirectorySnapshot(result);
        }
    }
}
