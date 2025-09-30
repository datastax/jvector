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
        this.monitor = new StatusMonitor<>(tracked, statusFunction, pollInterval, this::handleStatusChange);

        notifyTaskStarted();
        this.monitor.start();
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

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (monitor != null) {
                monitor.stop();
            }
            notifyTaskFinished();
            // Clear sink references to help GC
            sinks.clear();
        }
    }

    public StatusUpdate<T> getStatus() {
        return monitor != null ? monitor.getCurrentStatus() : null;
    }


}
