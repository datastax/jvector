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

import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;

import java.lang.reflect.Method;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * Represents a leaf node in the task tracking hierarchy managed by a {@link StatusContext}.
 * Trackers are responsible for observing actual work units and reporting their progress and state.
 * Unlike {@link StatusScope}, trackers have progress/state and cannot have children.
 *
 * <p><strong>Key Responsibilities:</strong></p>
 * <ul>
 *   <li><strong>Observation:</strong> Periodically calls the status function to observe the tracked object
 *       via {@link #refreshAndGetStatus()}, which is invoked by {@link StatusMonitor}</li>
 *   <li><strong>Caching:</strong> Stores the most recent status observation for query and timing purposes
 *       without requiring re-observation of the tracked object</li>
 *   <li><strong>Timing:</strong> Tracks task execution timing including start time, running duration,
 *       and accumulated running time across multiple RUNNING states</li>
 *   <li><strong>Scope Membership:</strong> Optionally belongs to a {@link StatusScope} for
 *       organizational hierarchy</li>
 * </ul>
 *
 * <p><strong>Architectural Flow:</strong></p>
 * <ol>
 *   <li>{@link StatusMonitor} periodically calls {@link #refreshAndGetStatus()} on this tracker</li>
 *   <li>Tracker invokes its status function to observe the tracked object</li>
 *   <li>Tracker caches the observed status and updates timing information</li>
 *   <li>{@link StatusContext} receives the status and routes it to all registered {@link StatusSink}s</li>
 *   <li>Status flows unidirectionally: Task → Tracker → Monitor → Context → Sinks</li>
 * </ol>
 *
 * <p><strong>Usage Pattern:</strong></p>
 * <pre>{@code
 * try (StatusContext context = new StatusContext("operation");
 *      StatusScope scope = context.createScope("DataProcessing")) {
 *
 *     // Create trackers as leaf nodes
 *     StatusTracker<LoadTask> loader = scope.trackTask(new LoadTask());
 *     StatusTracker<ProcessTask> processor = scope.trackTask(new ProcessTask());
 *
 *     // Trackers report progress automatically
 *     // Cannot create children - use scopes for hierarchy
 * }
 * }</pre>
 *
 * <p><strong>Thread Safety:</strong></p>
 * <ul>
 *   <li><strong>Status observation:</strong> {@link #refreshAndGetStatus()} is called only by the {@link StatusMonitor}
 *       thread, ensuring single-threaded access to the status function</li>
 *   <li><strong>Cached status:</strong> Read via volatile field, safe for concurrent access by multiple threads</li>
 *   <li><strong>Timing data:</strong> Protected by internal synchronization lock ({@code timingLock}), which is
 *       guaranteed non-null (final field initialized inline). All timing mutations occur within synchronized blocks,
 *       preventing race conditions between the monitor thread and close operations</li>
 *   <li><strong>Public methods:</strong> {@link #getStatus()}, {@link #getTracked()}, {@link #close()} are thread-safe</li>
 *   <li><strong>Close operation:</strong> Idempotent and safe to call from any thread. Uses {@code timingLock}
 *       synchronization to safely coordinate with concurrent monitor polling</li>
 * </ul>
 *
 * <p>This class should not be instantiated directly. Use {@link StatusContext#track} methods
 * or {@link StatusScope#trackTask} methods to create trackers.
 *
 * @param <T> the type of object being tracked
 * @see StatusScope
 * @see StatusContext
 * @see StatusMonitor
 * @see StatusSink
 * @see StatusUpdate
 * @since 4.0.0
 */
public final class StatusTracker<T> implements AutoCloseable {

    private final StatusContext context;
    private final StatusScope parentScope;
    private final Function<T, StatusUpdate<T>> statusFunction;
    private final T tracked;
    private final boolean ownsScope;

    private volatile boolean closed = false;
    private volatile StatusUpdate<T> lastStatus;

    // Synchronization lock for timing data. Guaranteed non-null (final, initialized inline).
    // Protects timing fields from concurrent access by the StatusMonitor thread and close() calls.
    private final Object timingLock = new Object();
    private volatile Long runningStartTime;
    private volatile Long firstRunningStartTime;
    private volatile long accumulatedRunTimeMillis;

    /**
     * Creates a standalone tracker with its own default StatusScope and StatusContext.
     * This is the simplest way to track a task when you don't need to configure
     * the context or organize multiple tasks.
     *
     * <p>Example usage:
     * <pre>{@code
     * try (StatusTracker<MyTask> tracker = new StatusTracker<>(new MyTask())) {
     *     // Add a sink to see progress
     *     tracker.getContext().addSink(new ConsoleLoggerSink());
     *
     *     // Execute the task
     *     tracker.getTracked().execute();
     * }
     * }</pre>
     *
     * @param tracked the task implementing StatusSource to track
     */
    public StatusTracker(T tracked) {
        this(tracked, null);
    }

    /**
     * Creates a standalone tracker with a custom status function, using its own
     * default StatusScope and StatusContext.
     *
     * @param tracked the object to track
     * @param statusFunction function to extract status from the tracked object (null to use StatusSource interface)
     */
    @SuppressWarnings("unchecked")
    public StatusTracker(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        this.tracked = Objects.requireNonNull(tracked, "tracked");

        // Determine status function
        if (statusFunction == null) {
            if (!(tracked instanceof StatusSource)) {
                throw new IllegalArgumentException("Task must implement StatusSource or provide a statusFunction");
            }
            StatusSource<?> source = (StatusSource<?>) tracked;
            this.statusFunction = t -> (StatusUpdate<T>) source.getTaskStatus();
        } else {
            this.statusFunction = statusFunction;
        }

        // Create our own scope (which creates its own context)
        String taskName = extractTaskNameFromObject(tracked);
        this.parentScope = new StatusScope("tracker-" + taskName);
        this.context = parentScope.getContext();
        this.ownsScope = true;

        // Register ourselves with the scope
        parentScope.addChildTask(this);

        // Register with monitor
        StatusUpdate<T> initial = refreshAndGetStatus();
        context.registerTracker(this, initial);
    }

    StatusTracker(StatusContext context,
                  StatusScope parentScope,
                  T tracked,
                  Function<T, StatusUpdate<T>> statusFunction) {
        this.context = Objects.requireNonNull(context, "context");
        this.tracked = Objects.requireNonNull(tracked, "tracked");
        this.statusFunction = Objects.requireNonNull(statusFunction, "statusFunction");

        // If no parent scope provided, create a default one
        if (parentScope == null) {
            String taskName = extractTaskNameFromObject(tracked);
            this.parentScope = context.createScope("auto-scope-" + taskName);
            this.ownsScope = true;
        } else {
            this.parentScope = parentScope;
            this.ownsScope = false;
        }
        // Note: parentScope relationship is managed by StatusScope.trackTask() or auto-created above
    }

    /**
     * Helper method to extract task name from an object (non-static version for constructor use).
     */
    private static String extractTaskNameFromObject(Object tracked) {
        try {
            Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall back to toString
        }
        return tracked.toString();
    }

    /**
     * Observes the tracked object by invoking the status function and caches the result.
     * This method is called by {@link StatusMonitor} at configured poll intervals.
     * The cached status is used for timing calculations and can be retrieved without
     * re-observation via {@link #getLastStatus()}.
     *
     * @return the newly observed status from the tracked object
     */
    StatusUpdate<T> refreshAndGetStatus() {
        StatusUpdate<T> status = statusFunction.apply(tracked);
        this.lastStatus = status;
        updateTiming(status);
        return status;
    }

    /**
     * Returns the last cached status without re-observing the tracked object.
     * This method is used internally for efficient status retrieval when fresh
     * observation is not required.
     *
     * @return the most recently cached status, or null if no status has been observed yet
     */
    StatusUpdate<T> getLastStatus() {
        return lastStatus;
    }


    /**
     * Returns whether this tracker has been closed.
     *
     * @return true if {@link #close()} has been called, false otherwise
     */
    boolean isClosed() {
        return closed;
    }

    /**
     * Returns the object being tracked by this tracker.
     *
     * @return the tracked object
     */
    public T getTracked() {
        return tracked;
    }

    /**
     * Returns the current status of the tracked object. If no status has been
     * cached yet, this method will perform an immediate observation by calling
     * {@link #refreshAndGetStatus()}.
     *
     * @return the current status of the tracked object
     */
    public StatusUpdate<T> getStatus() {
        StatusUpdate<T> current = lastStatus;
        if (current == null) {
            current = refreshAndGetStatus();
        }
        return current;
    }

    /**
     * Returns the scope that contains this tracker. For trackers created with an explicit scope,
     * returns that scope. For trackers created without a scope, returns the automatically
     * created scope.
     *
     * @return the scope containing this tracker
     */
    public StatusScope getScope() {
        return parentScope;
    }

    /**
     * Returns the parent scope if this tracker belongs to a scope, or null otherwise.
     * @deprecated Use {@link #getScope()} instead
     *
     * @return the parent scope, or null if this tracker doesn't belong to a scope
     */
    @Deprecated
    public StatusScope getParentScope() {
        return parentScope;
    }

    /**
     * Closes this tracker, unregistering it from monitoring. This method is idempotent
     * and safe to call multiple times.
     * <p>
     * When closed:
     * <ul>
     *   <li>The tracker performs a final status observation to capture the latest state</li>
     *   <li>The tracker is unregistered from {@link StatusMonitor} (no more polling)</li>
     *   <li>Running time is finalized for timing calculations</li>
     *   <li>The tracker is removed from its parent scope (if any)</li>
     *   <li>{@link StatusContext#onTrackerClosed} is invoked to complete cleanup and notify sinks</li>
     *   <li>If the tracker owns its scope (auto-created), the scope is also closed</li>
     * </ul>
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        // Perform final status observation to capture the latest state before closing
        refreshAndGetStatus();

        finalizeRunningTime(System.currentTimeMillis());
        context.onTrackerClosed(this);

        // If we own the scope (it was auto-created), close it as well
        if (ownsScope && parentScope != null) {
            parentScope.close();
        }
    }

    /**
     * Returns the context that manages this tracker.
     *
     * @return the owning context
     */
    public StatusContext getContext() {
        return context;
    }

    /**
     * Returns the total accumulated running time for this tracker in milliseconds.
     * This includes all time spent in the RUNNING state, even across multiple
     * transitions to/from RUNNING. For tasks currently running, includes the
     * time elapsed since the current RUNNING state began.
     *
     * @return accumulated running time in milliseconds
     */
    public long getElapsedRunningTime() {
        synchronized (timingLock) {
            long total = accumulatedRunTimeMillis;
            if (runningStartTime != null) {
                total += Math.max(0, System.currentTimeMillis() - runningStartTime);
            }
            return total;
        }
    }

    /**
     * Returns the timestamp when this tracker first entered the RUNNING state,
     * or null if it has never been RUNNING.
     *
     * @return the first running start timestamp in milliseconds since epoch, or null
     */
    public Long getRunningStartTime() {
        return firstRunningStartTime;
    }

    /**
     * Updates timing information based on the current status. This method is called
     * automatically by {@link #refreshAndGetStatus()} after observing the tracked object.
     * <p>
     * Timing transitions:
     * <ul>
     *   <li>PENDING → RUNNING: Records first and current running start times</li>
     *   <li>RUNNING → SUCCESS/FAILED/CANCELLED: Finalizes accumulated running time</li>
     *   <li>Other transitions: No timing changes</li>
     * </ul>
     *
     * @param status the observed status containing runstate information
     */
    private void updateTiming(StatusUpdate<T> status) {
        if (status == null || status.runstate == null) {
            return;
        }

        synchronized (timingLock) {
            switch (status.runstate) {
                case RUNNING:
                    if (runningStartTime == null) {
                        runningStartTime = status.timestamp;
                        if (firstRunningStartTime == null) {
                            firstRunningStartTime = runningStartTime;
                        }
                    }
                    break;
                case SUCCESS:
                case FAILED:
                case CANCELLED:
                    finalizeRunningTimeLocked(status.timestamp);
                    break;
                default:
                    // No-op for PENDING and other non-running states
                    break;
            }
        }
    }

    private void finalizeRunningTime(long timestamp) {
        synchronized (timingLock) {
            finalizeRunningTimeLocked(timestamp);
        }
    }

    private void finalizeRunningTimeLocked(long timestamp) {
        if (runningStartTime == null) {
            return;
        }

        accumulatedRunTimeMillis += Math.max(0, timestamp - runningStartTime);
        runningStartTime = null;
    }

    /**
     * Extracts a human-readable name from the tracked object.
     * Attempts to call a getName() method via reflection, falling back to toString().
     *
     * @param tracker the tracker whose tracked object should be named
     * @return the extracted name
     */
    public static String extractTaskName(StatusTracker<?> tracker) {
        Object tracked = tracker.getTracked();
        try {
            Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall back to toString
        }
        return tracked.toString();
    }

    /**
     * Returns a one-line summary of the tracker's current state.
     * Format: "TaskName [progress%] state (elapsed time)"
     *
     * <p>Example output:
     * <ul>
     *   <li>"DataLoader [45.2%] RUNNING (1234ms)"</li>
     *   <li>"DataProcessor [100.0%] SUCCESS (2567ms)"</li>
     *   <li>"ValidationTask [0.0%] PENDING (0ms)"</li>
     * </ul>
     *
     * @return a formatted string summarizing the tracker's status
     */
    @Override
    public String toString() {
        StatusUpdate<T> status = getStatus();
        String taskName = extractTaskName(this);
        double progressPercent = status.progress * 100.0;
        long elapsed = getElapsedRunningTime();

        return String.format("%s [%.1f%%] %s (%dms)",
                taskName,
                progressPercent,
                status.runstate,
                elapsed);
    }
}
