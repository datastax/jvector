package io.github.jbellis.jvector.status;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A generic monitoring system that continuously polls and tracks the status of any object
 * implementing task-like behavior. The monitor runs on a dedicated daemon thread and provides
 * real-time status updates through configurable callbacks.
 *
 * <p>This monitor provides:
 * <ul>
 *   <li>Continuous polling of task status at configurable intervals</li>
 *   <li>Automatic detection of status changes with callback notifications</li>
 *   <li>Lifecycle management with start/stop controls</li>
 *   <li>Thread-safe operation with proper resource cleanup</li>
 *   <li>Automatic completion detection and monitoring termination</li>
 *   <li>Generic type support for monitoring any object type</li>
 * </ul>
 *
 * <p>The monitor automatically stops when the tracked task reaches a terminal state
 * (SUCCESS, FAILED, or CANCELLED) and provides proper thread cleanup via daemon threads.
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Basic Task Monitoring</h3>
 * <pre>{@code
 * // Monitor a simple task with default 100ms polling
 * // Example with a custom task object implementing TaskStatus.Provider
 * MyTask task = new MyTask("data-processing");
 * TaskMonitor<MyTask> monitor = new TaskMonitor<>(
 *     task,
 *     MyTask::getTaskStatus
 * );
 *
 * monitor.start();
 * // Monitor runs automatically until task completes
 * // ...
 * monitor.stop(); // Clean shutdown when done
 * }</pre>
 *
 * <h3>Custom Polling Interval</h3>
 * <pre>{@code
 * // Monitor with custom 500ms polling interval
 * TaskMonitor<MyTask> monitor = new TaskMonitor<>(
 *     myTask,
 *     task -> getTaskStatus(task),
 *     Duration.ofMillis(500)
 * );
 *
 * monitor.start();
 * }</pre>
 *
 * <h3>Monitoring with Status Change Callbacks</h3>
 * <pre>{@code
 * TaskMonitor<ProcessingJob> monitor = new TaskMonitor<>(
 *     job,
 *     j -> j.getStatus(),
 *     Duration.ofMillis(200),
 *     status -> {
 *         System.out.printf("Job %s: %.1f%% complete (%s)%n",
 *             status.tracked.getName(),
 *             status.progress * 100,
 *             status.runstate);
 *
 *         if (status.runstate == TaskStatus.RunState.FAILED) {
 *             handleFailure(status.tracked);
 *         }
 *     }
 * );
 *
 * monitor.start();
 * }</pre>
 *
 * <h3>Batch Monitoring Pattern</h3>
 * <pre>{@code
 * List<TaskMonitor<ProcessingTask>> monitors = new ArrayList<>();
 * for (ProcessingTask task : tasks) {
 *     TaskMonitor<ProcessingTask> monitor = new TaskMonitor<>(
 *         task,
 *         t -> t.getCurrentStatus(),
 *         Duration.ofMillis(100),
 *         status -> updateProgress(task.getId(), status)
 *     );
 *     monitors.add(monitor);
 *     monitor.start();
 * }
 *
 * // All monitors run concurrently
 * // Cleanup when batch is complete
 * monitors.forEach(TaskMonitor::stop);
 * }</pre>
 *
 * <h2>Threading and Lifecycle</h2>
 * <p>Each TaskMonitor creates its own dedicated daemon thread for polling. The monitor
 * automatically handles thread lifecycle, including:</p>
 * <ul>
 *   <li>Thread creation on start()</li>
 *   <li>Proper thread interruption and cleanup on stop()</li>
 *   <li>Automatic termination when tasks complete</li>
 *   <li>Exception handling and recovery during polling</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>This class is thread-safe. The monitor can be safely started and stopped from
 * multiple threads, and status callbacks are invoked serially from the monitoring thread.</p>
 *
 * @param <T> the type of object being monitored
 * @see StatusUpdate
 * @see StatusUpdate
 * @since 4.0.0
 */
public class StatusMonitor<T> {
    private final T tracked;
    private final Function<T, StatusUpdate<T>> statusFunction;
    private final Duration pollInterval;
    private final Consumer<StatusUpdate<T>> statusChangeCallback;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    private volatile StatusUpdate<T> currentStatus;
    private ExecutorService executor;
    private CompletableFuture<Void> monitoringTask;

    public StatusMonitor(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        this(tracked, statusFunction, Duration.ofMillis(100), null);
    }

    public StatusMonitor(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval) {
        this(tracked, statusFunction, pollInterval, null);
    }

    public StatusMonitor(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval, Consumer<StatusUpdate<T>> statusChangeCallback) {
        this.tracked = tracked;
        this.statusFunction = statusFunction;
        this.pollInterval = pollInterval;
        this.statusChangeCallback = statusChangeCallback;
        this.currentStatus = statusFunction.apply(tracked);
    }

    public synchronized void start() {
        if (running.compareAndSet(false, true) && !shutdown.get()) {
            // Notify callback of initial status
            if (statusChangeCallback != null && currentStatus != null) {
                try {
                    statusChangeCallback.accept(currentStatus);
                } catch (Exception e) {
                    System.err.println("Error in initial status callback: " + e.getMessage());
                }
            }

            executor = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "TaskMonitor-" + getTaskName());
                t.setDaemon(true);
                return t;
            });

            monitoringTask = CompletableFuture.runAsync(this::monitorLoop, executor);
        }
    }

    public synchronized void stop() {
        if (shutdown.compareAndSet(false, true)) {
            running.set(false);

            if (monitoringTask != null) {
                monitoringTask.cancel(true);
            }

            if (executor != null) {
                executor.shutdown();
                try {
                    if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                        executor.shutdownNow();
                    }
                } catch (InterruptedException e) {
                    executor.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    public StatusUpdate<T> getCurrentStatus() {
        return currentStatus;
    }

    public boolean isRunning() {
        return running.get() && !shutdown.get();
    }

    public boolean isShutdown() {
        return shutdown.get();
    }

    private void monitorLoop() {
        try {
            while (running.get() && !shutdown.get() && !Thread.currentThread().isInterrupted()) {
                try {
                    StatusUpdate<T> newStatus = statusFunction.apply(tracked);
                    if (newStatus != null) {
                        StatusUpdate<T> previousStatus = currentStatus;
                        currentStatus = newStatus;

                        if (statusChangeCallback != null && !statusEquals(previousStatus, newStatus)) {
                            try {
                                statusChangeCallback.accept(newStatus);
                            } catch (Exception e) {
                                System.err.println("Error in status change callback: " + e.getMessage());
                            }
                        }
                    }

                    if (isTaskComplete(newStatus)) {
                        break;
                    }

                    Thread.sleep(pollInterval.toMillis());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    System.err.println("Error polling task status: " + e.getMessage());
                    try {
                        Thread.sleep(pollInterval.toMillis());
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        } finally {
            running.set(false);
        }
    }

    private boolean statusEquals(StatusUpdate<T> status1, StatusUpdate<T> status2) {
        if (status1 == null && status2 == null) return true;
        if (status1 == null || status2 == null) return false;
        return status1.progress == status2.progress && status1.runstate == status2.runstate;
    }

    private boolean isTaskComplete(StatusUpdate<T> status) {
        if (status == null) {
            return false;
        }
        return status.runstate == StatusUpdate.RunState.SUCCESS ||
               status.runstate == StatusUpdate.RunState.FAILED ||
               status.runstate == StatusUpdate.RunState.CANCELLED;
    }

    private String getTaskName() {
        // Use reflection to find getName() method
        try {
            java.lang.reflect.Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall through to default toString
        }
        return tracked.getClass().getSimpleName() + "@" + Integer.toHexString(tracked.hashCode());
    }
}