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

package io.github.jbellis.jvector.status;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * A lightweight task monitoring wrapper that provides passive status polling and sink
 * notifications for any task object. The tracker runs a background monitoring thread
 * that continuously polls the task status and notifies registered sinks of changes.
 *
 * <p>This class provides:
 * <ul>
 *   <li>Automatic background status monitoring with configurable polling intervals</li>
 *   <li>Multiple sink notification support for status changes</li>
 *   <li>Lifecycle management with proper resource cleanup</li>
 *   <li>Support for both instrumented objects and custom status functions</li>
 *   <li>Thread-safe operation and concurrent access</li>
 * </ul>
 *
 * <p><strong>CRITICAL:</strong> This class MUST be used with try-with-resources to ensure
 * proper cleanup and prevent resource leaks. Failing to close trackers will result in
 * background threads remaining active indefinitely.
 *
 * <h2>Usage Patterns:</h2>
 *
 * <h3>Basic Tracking via TrackerScope (Recommended)</h3>
 * <pre>{@code
 * TrackerScope scope = new TrackerScope("batch-processing");
 * scope.addSink(new ConsoleTaskSink());
 *
 * try (Tracker<MyTask> tracker = scope.track(myTask)) {
 *     // Task executes automatically in background
 *     // Status updates sent to console sink
 * } // Automatically closed, background thread stopped
 * }</pre>
 *
 * <h3>Direct Creation for Instrumented Objects</h3>
 * <pre>{@code
 * // For objects implementing TaskStatus.Provider
 * try (Tracker<InstrumentedTask> tracker = Tracker.withInstrumented(task, consoleSink)) {
 *     task.execute(); // Status automatically monitored
 * }
 * }</pre>
 *
 * <h3>Custom Status Function</h3>
 * <pre>{@code
 * // For objects that don't implement TaskStatus.Provider
 * Function<ProcessingJob, TaskStatus<ProcessingJob>> statusFunc = job ->
 *     new TaskStatus<>(job.getCompletionRatio(), job.getCurrentState());
 *
 * try (Tracker<ProcessingJob> tracker = Tracker.withFunctors(job, statusFunc, sink)) {
 *     job.start(); // Background monitoring begins
 * }
 * }</pre>
 *
 * <h3>Multiple Sinks and Custom Polling</h3>
 * <pre>{@code
 * List<TaskSink> sinks = Arrays.asList(
 *     new ConsoleTaskSink(),
 *     new MetricsTaskSink(),
 *     new LoggerTaskSink()
 * );
 *
 * try (Tracker<DataProcessor> tracker = Tracker.withInstrumented(
 *         processor,
 *         Duration.ofMillis(250), // Poll every 250ms
 *         sinks)) {
 *     processor.processData();
 * }
 * }</pre>
 *
 * <h3>Dynamic Sink Management</h3>
 * <pre>{@code
 * try (Tracker<LongRunningTask> tracker = Tracker.withInstrumented(task)) {
 *     // Start with no sinks
 *     task.start();
 *
 *     // Add monitoring during execution
 *     tracker.addSink(new ConsoleTaskSink());
 *
 *     if (task.needsDetailedLogging()) {
 *         tracker.addSink(new LoggerTaskSink());
 *     }
 * }
 * }</pre>
 *
 * <h2>Factory Methods:</h2>
 * <p>Two main factory patterns are available:</p>
 * <ul>
 *   <li><strong>withInstrumented()</strong> - For objects implementing {@link StatusUpdate.Provider}</li>
 *   <li><strong>withFunctors()</strong> - For objects with custom status extraction functions</li>
 * </ul>
 *
 * <h2>Thread Safety and Lifecycle</h2>
 * <p>Trackers are thread-safe and manage their own background monitoring threads:</p>
 * <ul>
 *   <li>Each tracker creates a dedicated daemon thread for status polling</li>
 *   <li>Sink notifications are serialized and happen on the monitoring thread</li>
 *   <li>Adding/removing sinks is thread-safe and takes effect immediately</li>
 *   <li>Closing the tracker immediately stops the background thread</li>
 * </ul>
 *
 * <h2>Best Practices</h2>
 * <ul>
 *   <li>Always use try-with-resources for automatic cleanup</li>
 *   <li>Create trackers through {@link TrackerScope} when possible for better configuration management</li>
 *   <li>Keep polling intervals reasonable (100-1000ms) to balance responsiveness and overhead</li>
 *   <li>Handle exceptions in sink implementations to prevent monitoring disruption</li>
 *   <li>Use appropriate sink types for different monitoring needs (console, metrics, logging)</li>
 * </ul>
 *
 * @param <T> the type of object being tracked
 * @see TrackerScope
 * @see StatusMonitor
 * @see StatusUpdate
 * @see StatusSink
 * @since 4.0.0
 */
public class StatusTracker<T> implements AutoCloseable {

    private final T tracked;
    private final Function<T, StatusUpdate<T>> statusFunction;
    private final StatusMonitor<T> monitor;
    private final List<StatusSink> sinks;
    private volatile boolean closed = false;
    private final StatusTracker<?> parent;
    private TrackerScope scope; // The scope that created this tracker
    private final List<StatusTracker<?>> children = new CopyOnWriteArrayList<>(); // Child trackers

    // ThreadLocal to track the currently active tracker in this thread
    // This allows child trackers to automatically discover their parent
    private static final ThreadLocal<StatusTracker<?>> currentTracker = new ThreadLocal<>();

    private StatusTracker(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        this(tracked, statusFunction, Duration.ofMillis(100), new ArrayList<>());
    }

    private StatusTracker(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval) {
        this(tracked, statusFunction, pollInterval, new ArrayList<>());
    }

    private StatusTracker(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval, List<StatusSink> sinks) {
        this.tracked = tracked;
        this.statusFunction=statusFunction;
        this.sinks = new CopyOnWriteArrayList<>(sinks);

        // Capture parent tracker from ThreadLocal
        this.parent = currentTracker.get();

        // Set this as the current tracker for any children created in the same thread
        StatusTracker<?> previousTracker = currentTracker.get();
        currentTracker.set(this);

        try {
            this.monitor = new StatusMonitor<>(tracked, statusFunction, pollInterval, this::handleStatusChange);

            notifyTaskStarted();
            this.monitor.start();
        } finally {
            // Restore the previous tracker
            currentTracker.set(previousTracker);
        }
    }

    private void handleStatusChange(StatusUpdate<T> newStatus) {
        if (!sinks.isEmpty()) {
            for (StatusSink sink : sinks) {
                try {
                    sink.taskUpdate(this, newStatus);
                } catch (Exception e) {
                    System.err.println("Error notifying sink of status change: " + e.getMessage());
                }
            }
        }
    }

    public static <U extends StatusUpdate.Provider<U>> StatusTracker<U> withInstrumented(U tracked) {
        return new StatusTracker<>(tracked, StatusUpdate.Provider::getTaskStatus);
    }

    public static <U extends StatusUpdate.Provider<U>> StatusTracker<U> withInstrumented(U tracked, StatusSink... sinks) {
        return new StatusTracker<>(tracked, StatusUpdate.Provider::getTaskStatus, Duration.ofMillis(100), Arrays.asList(sinks));
    }

    public static <U extends StatusUpdate.Provider<U>> StatusTracker<U> withInstrumented(U tracked, List<StatusSink> sinks) {
        return new StatusTracker<>(tracked, StatusUpdate.Provider::getTaskStatus, Duration.ofMillis(100), sinks);
    }

    public static <U extends StatusUpdate.Provider<U>> StatusTracker<U> withInstrumented(U tracked, Duration pollInterval, StatusSink... sinks) {
        return new StatusTracker<>(tracked, StatusUpdate.Provider::getTaskStatus, pollInterval, Arrays.asList(sinks));
    }

    public static <U extends StatusUpdate.Provider<U>> StatusTracker<U> withInstrumented(U tracked, Duration pollInterval, List<StatusSink> sinks) {
        return new StatusTracker<>(tracked, StatusUpdate.Provider::getTaskStatus, pollInterval, sinks);
    }

    public static <T> StatusTracker<T> withFunctors(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        return new StatusTracker<>(tracked, statusFunction);
    }

    public static <T> StatusTracker<T> withFunctors(T tracked, Function<T, StatusUpdate<T>> statusFunction, StatusSink... sinks) {
        return new StatusTracker<>(tracked, statusFunction, Duration.ofMillis(100), Arrays.asList(sinks));
    }

    public static <T> StatusTracker<T> withFunctors(T tracked, Function<T, StatusUpdate<T>> statusFunction, List<StatusSink> sinks) {
        return new StatusTracker<>(tracked, statusFunction, Duration.ofMillis(100), sinks);
    }

    public static <T> StatusTracker<T> withFunctors(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval, StatusSink... sinks) {
        return new StatusTracker<>(tracked, statusFunction, pollInterval, Arrays.asList(sinks));
    }

    public static <T> StatusTracker<T> withFunctors(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval, List<StatusSink> sinks) {
        return new StatusTracker<>(tracked, statusFunction, pollInterval, sinks);
    }

    public void addSink(StatusSink sink) {
        if (sink != null) {
            sinks.add(sink);
            // Notify the newly added sink that the task has started
            try {
                sink.taskStarted(this);
            } catch (Exception e) {
                System.err.println("Error notifying sink of task start: " + e.getMessage());
            }
        }
    }

    public void removeSink(StatusSink sink) {
        if (sinks.remove(sink)) {
            // Notify the removed sink that the task has finished for it
            try {
                sink.taskFinished(this);
            } catch (Exception e) {
                System.err.println("Error notifying sink of task finish: " + e.getMessage());
            }
        }
    }

    public List<StatusSink> getSinks() {
        return new ArrayList<>(sinks);
    }

    private void notifyTaskStarted() {
        for (StatusSink sink : sinks) {
            try {
                sink.taskStarted(this);
            } catch (Exception e) {
                System.err.println("Error notifying sink of task start: " + e.getMessage());
            }
        }
    }

    private void notifyTaskFinished() {
        for (StatusSink sink : sinks) {
            try {
                sink.taskFinished(this);
            } catch (Exception e) {
                System.err.println("Error notifying sink of task finish: " + e.getMessage());
            }
        }
    }

    public T getTracked() {
        return tracked;
    }

    /**
     * Gets the parent tracker of this tracker, if any.
     *
     * @return the parent tracker, or null if this is a root tracker
     */
    public StatusTracker<?> getParent() {
        return parent;
    }

    /**
     * Creates a child tracker for an instrumented object.
     * The child will inherit sinks and configuration from this tracker's scope.
     * The child tracker will have this tracker as its parent.
     *
     * @param childTracked the object to track
     * @return a new child StatusTracker
     */
    public <U extends StatusUpdate.Provider<U>> StatusTracker<U> createChild(U childTracked) {
        // Set this as current tracker so the child will have us as parent
        StatusTracker<?> previous = currentTracker.get();
        currentTracker.set(this);
        try {
            // Create the child tracker with the same sinks
            StatusTracker<U> child = StatusTracker.withInstrumented(childTracked,
                Duration.ofMillis(100), sinks);

            // Set the scope if we have one
            if (scope != null) {
                child.setScope(scope);
            }

            // Add to our children list
            children.add(child);

            // Register with scope if it tracks active trackers
            if (scope != null) {
                scope.getActiveTrackers().add(child);
            }

            return child;
        } finally {
            currentTracker.set(previous);
        }
    }

    /**
     * Creates a child tracker with a custom status function.
     *
     * @param childTracked the object to track
     * @param statusFunction function to get status from the object
     * @return a new child StatusTracker
     */
    public <U> StatusTracker<U> createChild(U childTracked, Function<U, StatusUpdate<U>> statusFunction) {
        // Set this as current tracker so the child will have us as parent
        StatusTracker<?> previous = currentTracker.get();
        currentTracker.set(this);
        try {
            // Create the child tracker with the same sinks
            StatusTracker<U> child = StatusTracker.withFunctors(childTracked, statusFunction,
                Duration.ofMillis(100), sinks);

            // Set the scope if we have one
            if (scope != null) {
                child.setScope(scope);
            }

            // Add to our children list
            children.add(child);

            // Register with scope if it tracks active trackers
            if (scope != null) {
                scope.getActiveTrackers().add(child);
            }

            return child;
        } finally {
            currentTracker.set(previous);
        }
    }

    /**
     * Gets the child trackers of this tracker.
     *
     * @return list of child trackers
     */
    public List<StatusTracker<?>> getChildren() {
        return new ArrayList<>(children);
    }

    /**
     * Sets the scope that created this tracker.
     * Used internally by TrackerScope for lifecycle management.
     *
     * @param scope the scope that created this tracker
     */
    void setScope(TrackerScope scope) {
        this.scope = scope;
    }

    /**
     * Gets the current active tracker for this thread.
     * This is used to automatically establish parent-child relationships.
     *
     * @return the current tracker, or null if no tracker is active
     */
    public static StatusTracker<?> getCurrentTracker() {
        return currentTracker.get();
    }

    /**
     * Sets the current active tracker for this thread.
     * This should be managed internally by the tracker lifecycle.
     *
     * @param tracker the tracker to set as current
     */
    static void setCurrentTracker(StatusTracker<?> tracker) {
        currentTracker.set(tracker);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;

            // Ensure we're removed from the current tracker if we're still set
            if (currentTracker.get() == this) {
                currentTracker.set(parent);
            }

            // Remove from scope if we have one
            if (scope != null) {
                scope.removeTracker(this);
            }

            if (monitor != null) {
                monitor.stop();
            }
            notifyTaskFinished();
            // Clear sink references to help GC
            sinks.clear();
        }
    }

    /**
     * Executes the given runnable with this tracker as the current tracker.
     * This ensures proper parent-child relationships for any trackers created
     * within the runnable.
     *
     * @param runnable the code to execute
     */
    public void executeWithContext(Runnable runnable) {
        StatusTracker<?> previous = currentTracker.get();
        currentTracker.set(this);
        try {
            runnable.run();
        } finally {
            currentTracker.set(previous);
        }
    }

    /**
     * Executes the given supplier with this tracker as the current tracker.
     * This ensures proper parent-child relationships for any trackers created
     * within the supplier.
     *
     * @param <R> the return type
     * @param supplier the code to execute
     * @return the result of the supplier
     */
    public <R> R executeWithContext(java.util.function.Supplier<R> supplier) {
        StatusTracker<?> previous = currentTracker.get();
        currentTracker.set(this);
        try {
            return supplier.get();
        } finally {
            currentTracker.set(previous);
        }
    }

    public StatusUpdate<T> getStatus() {
        return monitor != null ? monitor.getCurrentStatus() : null;
    }

    /**
     * Returns the elapsed time in milliseconds since the task started running,
     * or 0 if the task hasn't started running yet.
     */
    public long getElapsedRunningTime() {
        return monitor != null ? monitor.getElapsedRunningTime() : 0;
    }

    /**
     * Returns the time (in milliseconds) when the task transitioned to RUNNING state,
     * or null if the task hasn't started running yet.
     */
    public Long getRunningStartTime() {
        return monitor != null ? monitor.getRunningStartTime() : null;
    }


}
