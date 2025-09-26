package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.StatusSink;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.StatusTracker;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * A metrics-collecting task sink that captures detailed performance and progress statistics
 * for task tracking and analysis. This sink is designed for production environments where
 * task performance monitoring, SLA tracking, and operational metrics are important.
 *
 * <p>This sink provides comprehensive metrics including:
 * <ul>
 *   <li>Task execution timing (start, end, duration)</li>
 *   <li>Progress statistics (updates, averages, current values)</li>
 *   <li>Aggregate counts (total tasks, finished tasks, active tasks)</li>
 *   <li>Performance analysis (average durations, update frequencies)</li>
 *   <li>Detailed reporting with task breakdowns</li>
 * </ul>
 *
 * <p>All metrics are thread-safe and can be safely accessed during concurrent task execution.
 * The sink uses atomic operations and concurrent data structures to ensure accuracy under load.
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Basic Metrics Collection</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 * TrackerScope scope = new TrackerScope("batch-processing");
 * scope.addSink(metrics);
 *
 * // Process multiple tasks
 * try (Tracker<Task1> t1 = scope.track(task1);
 *      Tracker<Task2> t2 = scope.track(task2)) {
 *
 *     CompletableFuture.allOf(
 *         CompletableFuture.runAsync(task1::execute),
 *         CompletableFuture.runAsync(task2::execute)
 *     ).join();
 * }
 *
 * // Analyze results
 * System.out.println("Tasks completed: " + metrics.getTotalTasksFinished());
 * System.out.println("Average duration: " + metrics.getAverageTaskDuration() + "ms");
 * }</pre>
 *
 * <h3>Individual Task Metrics</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 *
 * try (Tracker<DataProcessor> tracker = Tracker.withInstrumented(processor, metrics)) {
 *     processor.processLargeDataset();
 *
 *     // Get detailed metrics for this specific task
 *     MetricsTaskSink.TaskMetrics taskMetrics = metrics.getMetrics(tracker);
 *     System.out.println("Task duration: " + taskMetrics.getDuration() + "ms");
 *     System.out.println("Progress updates: " + taskMetrics.getUpdateCount());
 *     System.out.println("Final progress: " + taskMetrics.getLastProgress() * 100 + "%");
 * }
 * }</pre>
 *
 * <h3>Performance Monitoring and Alerting</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 * // ... run tasks
 *
 * // Check for performance issues
 * double avgDuration = metrics.getAverageTaskDuration();
 * if (avgDuration > 30000) { // 30 seconds
 *     logger.warn("Tasks running slower than expected: " + avgDuration + "ms average");
 * }
 *
 * // Monitor active task count
 * long activeTasks = metrics.getActiveTaskCount();
 * if (activeTasks > 100) {
 *     logger.warn("High number of active tasks: " + activeTasks);
 * }
 * }</pre>
 *
 * <h3>Detailed Reporting</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 * // ... run tasks
 *
 * // Generate comprehensive report
 * String report = metrics.generateReport();
 * System.out.println(report);
 *
 * // Output:
 * // === Task Metrics Report ===
 * // Total tasks started: 15
 * // Total tasks finished: 12
 * // Active tasks: 3
 * // Total updates: 847
 * // Average task duration: 2547.33 ms
 * //
 * // Task Details:
 * //   - data-processing:
 * //     Duration: 3200 ms
 * //     Updates: 64
 * //     Progress: 100.0%
 * //     Status: Finished
 * }</pre>
 *
 * <h3>Integration with Monitoring Systems</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 * // ... configure and run tasks
 *
 * // Periodically export metrics to external systems
 * ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
 * scheduler.scheduleAtFixedRate(() -> {
 *     // Export to Prometheus, StatsD, CloudWatch, etc.
 *     exportToMonitoringSystem(
 *         "tasks.started", metrics.getTotalTasksStarted(),
 *         "tasks.finished", metrics.getTotalTasksFinished(),
 *         "tasks.active", metrics.getActiveTaskCount(),
 *         "tasks.avg_duration", metrics.getAverageTaskDuration()
 *     );
 * }, 0, 60, TimeUnit.SECONDS);
 * }</pre>
 *
 * <h3>Memory Management</h3>
 * <pre>{@code
 * MetricsTaskSink metrics = new MetricsTaskSink();
 * // ... run tasks
 *
 * // Clean up finished task metrics to prevent memory leaks
 * Map<Tracker<?>, MetricsTaskSink.TaskMetrics> allMetrics = metrics.getAllMetrics();
 * allMetrics.entrySet().removeIf(entry -> {
 *     MetricsTaskSink.TaskMetrics taskMetrics = entry.getValue();
 *     return taskMetrics.isFinished() &&
 *            (System.currentTimeMillis() - taskMetrics.getEndTime()) > Duration.ofHours(1).toMillis();
 * });
 * }</pre>
 *
 * <h2>TaskMetrics Class</h2>
 * <p>The {@link TaskMetrics} inner class provides detailed statistics for individual tasks:</p>
 * <ul>
 *   <li><strong>Timing:</strong> Start time, end time, duration (including running tasks)</li>
 *   <li><strong>Progress:</strong> Update count, average progress, last progress value</li>
 *   <li><strong>Status:</strong> Task name, completion status</li>
 * </ul>
 *
 * <h2>Thread Safety and Performance</h2>
 * <p>This sink is designed for high-throughput environments with multiple concurrent tasks:</p>
 * <ul>
 *   <li>Thread-safe using {@link AtomicLong} and {@link DoubleAdder} for counters</li>
 *   <li>{@link ConcurrentHashMap} for metrics storage with minimal lock contention</li>
 *   <li>Low overhead per task update (O(1) operations)</li>
 *   <li>Safe for concurrent read/write access from multiple threads</li>
 * </ul>
 *
 * <h2>Memory Considerations</h2>
 * <p>The sink retains metrics for all tracked tasks until explicitly cleared:</p>
 * <ul>
 *   <li>Use {@link #removeMetrics(StatusTracker)} to clean up individual tasks</li>
 *   <li>Use {@link #clearMetrics()} to reset all metrics</li>
 *   <li>Implement periodic cleanup for long-running applications</li>
 *   <li>Monitor memory usage in high-volume scenarios</li>
 * </ul>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see TrackerScope
 * @since 4.0.0
 */
public class MetricsStatusSink implements StatusSink {

    public static class TaskMetrics {
        private final AtomicLong startTime = new AtomicLong();
        private final AtomicLong endTime = new AtomicLong();
        private final AtomicLong updateCount = new AtomicLong();
        private final DoubleAdder totalProgress = new DoubleAdder();
        private volatile double lastProgress = 0.0;
        private volatile String taskName;

        public long getStartTime() {
            return startTime.get();
        }

        public long getEndTime() {
            return endTime.get();
        }

        public long getDuration() {
            long end = endTime.get();
            if (end == 0) {
                return System.currentTimeMillis() - startTime.get();
            }
            return end - startTime.get();
        }

        public long getUpdateCount() {
            return updateCount.get();
        }

        public double getAverageProgress() {
            long count = updateCount.get();
            return count > 0 ? totalProgress.sum() / count : 0.0;
        }

        public double getLastProgress() {
            return lastProgress;
        }

        public String getTaskName() {
            return taskName;
        }

        public boolean isFinished() {
            return endTime.get() > 0;
        }
    }

    private final Map<StatusTracker<?>, TaskMetrics> metricsMap = new ConcurrentHashMap<>();
    private final AtomicLong totalTasksStarted = new AtomicLong();
    private final AtomicLong totalTasksFinished = new AtomicLong();
    private final AtomicLong totalUpdates = new AtomicLong();

    @Override
    public void taskStarted(StatusTracker<?> task) {
        TaskMetrics metrics = new TaskMetrics();
        metrics.startTime.set(System.currentTimeMillis());
        metrics.taskName = getTaskName(task);
        metricsMap.put(task, metrics);
        totalTasksStarted.incrementAndGet();
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        TaskMetrics metrics = metricsMap.get(task);
        if (metrics != null) {
            metrics.updateCount.incrementAndGet();
            metrics.totalProgress.add(status.progress);
            metrics.lastProgress = status.progress;
        }
        totalUpdates.incrementAndGet();
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        TaskMetrics metrics = metricsMap.get(task);
        if (metrics != null) {
            metrics.endTime.set(System.currentTimeMillis());
        }
        totalTasksFinished.incrementAndGet();
    }

    public TaskMetrics getMetrics(StatusTracker<?> task) {
        if (task == null) {
            return null;
        }
        return metricsMap.get(task);
    }

    public Map<StatusTracker<?>, TaskMetrics> getAllMetrics() {
        return new ConcurrentHashMap<>(metricsMap);
    }

    public void clearMetrics() {
        metricsMap.clear();
    }

    public void removeMetrics(StatusTracker<?> task) {
        metricsMap.remove(task);
    }

    public long getTotalTasksStarted() {
        return totalTasksStarted.get();
    }

    public long getTotalTasksFinished() {
        return totalTasksFinished.get();
    }

    public long getTotalUpdates() {
        return totalUpdates.get();
    }

    public long getActiveTaskCount() {
        return metricsMap.values().stream()
                .filter(m -> !m.isFinished())
                .count();
    }

    public double getAverageTaskDuration() {
        long finishedCount = 0;
        long totalDuration = 0;

        for (TaskMetrics metrics : metricsMap.values()) {
            if (metrics.isFinished()) {
                finishedCount++;
                totalDuration += metrics.getDuration();
            }
        }

        return finishedCount > 0 ? (double) totalDuration / finishedCount : 0.0;
    }

    public String generateReport() {
        StringBuilder report = new StringBuilder();
        report.append("=== Task Metrics Report ===\n");
        report.append("Total tasks started: ").append(totalTasksStarted.get()).append("\n");
        report.append("Total tasks finished: ").append(totalTasksFinished.get()).append("\n");
        report.append("Active tasks: ").append(getActiveTaskCount()).append("\n");
        report.append("Total updates: ").append(totalUpdates.get()).append("\n");
        report.append("Average task duration: ").append(String.format("%.2f ms", getAverageTaskDuration())).append("\n");

        report.append("\nTask Details:\n");
        for (Map.Entry<StatusTracker<?>, TaskMetrics> entry : metricsMap.entrySet()) {
            TaskMetrics metrics = entry.getValue();
            report.append("  - ").append(metrics.getTaskName()).append(":\n");
            report.append("    Duration: ").append(metrics.getDuration()).append(" ms\n");
            report.append("    Updates: ").append(metrics.getUpdateCount()).append("\n");
            report.append("    Progress: ").append(String.format("%.1f%%", metrics.getLastProgress() * 100)).append("\n");
            report.append("    Status: ").append(metrics.isFinished() ? "Finished" : "Running").append("\n");
        }

        return report.toString();
    }

    private String getTaskName(StatusTracker<?> task) {
        Object tracked = task.getTracked();
        // SimpleTask moved to test package - use reflection for getName()
        try {
            java.lang.reflect.Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall through to default toString
        }
        return tracked.toString();
    }
}