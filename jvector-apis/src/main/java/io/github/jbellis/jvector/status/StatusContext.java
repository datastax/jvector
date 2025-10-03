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

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Central coordinator for task tracking that manages scopes, trackers, sinks, and the monitoring infrastructure.
 * A context represents a cohesive tracking scope (such as a batch operation or application subsystem)
 * and owns exactly one {@link StatusMonitor} instance plus a collection of {@link StatusSink}s.
 *
 * <p><strong>Architectural Model:</strong></p>
 * <p>The API enforces a clear separation between organizational structure and work execution:</p>
 * <ul>
 *   <li><strong>{@link TrackerScope}:</strong> Organizational containers with no progress/state
 *       <ul>
 *         <li>Created via {@link #createScope(String)}</li>
 *         <li>Can contain child scopes (nested organization)</li>
 *         <li>Can contain task trackers (actual work)</li>
 *         <li>Completion checked via {@link TrackerScope#isComplete()}</li>
 *       </ul>
 *   </li>
 *   <li><strong>{@link StatusTracker}:</strong> Leaf nodes representing actual work
 *       <ul>
 *         <li>Created via {@link TrackerScope#trackTask} or {@link #track}</li>
 *         <li>Have progress and state (PENDING, RUNNING, SUCCESS, etc.)</li>
 *         <li>Cannot have children - purely leaf nodes</li>
 *         <li>Report status via {@link StatusSource#getTaskStatus()}</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * <p><strong>Key Responsibilities:</strong></p>
 * <ul>
 *   <li><strong>Scope Creation:</strong> Factory for {@link TrackerScope} via {@link #createScope}</li>
 *   <li><strong>Tracker Creation:</strong> Factory for {@link StatusTracker} via {@code track()} methods</li>
 *   <li><strong>Monitor Ownership:</strong> Owns a single {@link StatusMonitor} that polls all trackers</li>
 *   <li><strong>Sink Management:</strong> Maintains a collection of sinks that receive status updates</li>
 *   <li><strong>Status Routing:</strong> Routes status updates from monitor to all registered sinks</li>
 *   <li><strong>Lifecycle Management:</strong> Coordinates cleanup of scopes, trackers, monitor, and sinks</li>
 * </ul>
 *
 * <p><strong>Data Flow:</strong></p>
 * <ol>
 *   <li>User creates scope via {@code createScope()}</li>
 *   <li>Scope creates trackers via {@code trackTask()}</li>
 *   <li>Context registers tracker with its {@link StatusMonitor}</li>
 *   <li>Monitor periodically polls tracker via {@link StatusTracker#refreshAndGetStatus()}</li>
 *   <li>Tracker observes its object and caches the status</li>
 *   <li>Monitor forwards status to context via {@link #pushStatus}</li>
 *   <li>Context routes status to all registered {@link StatusSink}s</li>
 *   <li>Status flows unidirectionally: Task → Tracker → Monitor → Context → Sinks</li>
 * </ol>
 *
 * <p><strong>Usage Example:</strong></p>
 * <pre>{@code
 * try (StatusContext context = new StatusContext("data-pipeline")) {
 *     context.addSink(new ConsolePanelSink.builder().build());
 *     context.addSink(new MetricsStatusSink());
 *
 *     // Create organizational scope
 *     try (TrackerScope ingestionScope = context.createScope("Ingestion");
 *          TrackerScope processingScope = context.createScope("Processing")) {
 *
 *         // Add actual work as leaf tasks
 *         StatusTracker<LoadTask> loader = ingestionScope.trackTask(new LoadTask());
 *         StatusTracker<TransformTask> transformer = processingScope.trackTask(new TransformTask());
 *
 *         // Execute tasks...
 *         loader.getTracked().execute();
 *         transformer.getTracked().execute();
 *
 *         // Check scope completion
 *         boolean ingestionDone = ingestionScope.isComplete();
 *     }
 * }
 * }</pre>
 *
 * <p>This class implements {@link StatusSink} to receive events from the monitor and forward them
 * to registered sinks. It also implements {@link AutoCloseable} to ensure proper cleanup of resources.
 *
 * @see TrackerScope
 * @see StatusTracker
 * @see StatusMonitor
 * @see StatusSink
 * @since 4.0.0
 */
public class StatusContext implements AutoCloseable, StatusSink {

    private final String name;
    private final Duration defaultPollInterval;
    private final CopyOnWriteArrayList<StatusSink> sinks;
    private final CopyOnWriteArrayList<StatusTracker<?>> activeTrackers;
    private final CopyOnWriteArrayList<TrackerScope> activeScopes;
    private final StatusMonitor monitor;
    private volatile boolean closed = false;

    /**
     * Creates a new context with the specified name and default configuration
     * (100ms poll interval, no sinks).
     *
     * @param name the name of this context for identification purposes
     */
    public StatusContext(String name) {
        this(name, Duration.ofMillis(100), List.of());
    }

    /**
     * Creates a new context with the specified name and poll interval, with no sinks.
     *
     * @param name the name of this context for identification purposes
     * @param defaultPollInterval the default interval between status observations
     */
    public StatusContext(String name, Duration defaultPollInterval) {
        this(name, defaultPollInterval, List.of());
    }

    /**
     * Creates a new context with the specified name and initial sinks, using the
     * default poll interval of 100ms.
     *
     * @param name the name of this context for identification purposes
     * @param sinks initial collection of sinks to register
     */
    public StatusContext(String name, List<StatusSink> sinks) {
        this(name, Duration.ofMillis(100), sinks);
    }

    /**
     * Creates a new context with full configuration.
     *
     * @param name the name of this context for identification purposes
     * @param defaultPollInterval the default interval between status observations
     * @param sinks initial collection of sinks to register
     */
    public StatusContext(String name, Duration defaultPollInterval, List<StatusSink> sinks) {
        this.name = Objects.requireNonNull(name, "name");
        this.defaultPollInterval = Objects.requireNonNullElse(defaultPollInterval, Duration.ofMillis(100));
        this.sinks = new CopyOnWriteArrayList<>(Objects.requireNonNullElse(sinks, List.of()));
        this.activeTrackers = new CopyOnWriteArrayList<>();
        this.activeScopes = new CopyOnWriteArrayList<>();
        this.monitor = new StatusMonitor(this);
    }

    /**
     * Creates a root-level organizational scope for grouping related tasks.
     * Scopes provide hierarchical organization without having their own progress or state.
     *
     * @param name the name of the scope
     * @return a new TrackerScope registered with this context
     */
    public TrackerScope createScope(String name) {
        checkNotClosed();
        TrackerScope scope = new TrackerScope(this, null, name);
        activeScopes.add(scope);
        scopeStarted(scope);
        return scope;
    }

    TrackerScope createChildScope(TrackerScope parent, String name) {
        checkNotClosed();
        TrackerScope scope = new TrackerScope(this, parent, name);
        activeScopes.add(scope);
        scopeStarted(scope);
        return scope;
    }

    private void scopeStarted(TrackerScope scope) {
        notifySinks(sink -> {
            // Use reflection to avoid circular dependency
            try {
                var method = sink.getClass().getMethod("scopeStarted", TrackerScope.class);
                method.invoke(sink, scope);
            } catch (NoSuchMethodException e) {
                // Sink doesn't support scopes - ignore
            } catch (Exception e) {
                // Log but don't fail
            }
        }, "notifying sink of scope start");
    }

    /**
     * Creates a root-level tracker for an object implementing {@link StatusSource}.
     * Uses the context's default poll interval.
     *
     * @param tracked the object to track
     * @param <U> the type of object being tracked
     * @return a new tracker registered with this context's monitor
     * @deprecated StatusTrackers must belong to a TrackerScope. Use {@link TrackerScope#trackTask} instead.
     *             Create a scope first with {@link #createScope(String)}, then track tasks within it.
     */
    @Deprecated
    public <U extends StatusSource<U>> StatusTracker<U> track(U tracked) {
        throw new UnsupportedOperationException(
            "StatusTrackers must belong to a TrackerScope. " +
            "Use createScope(name).trackTask(tracked) instead.");
    }

    /**
     * Creates a root-level tracker for an object implementing {@link StatusSource}
     * with a custom poll interval.
     *
     * @param tracked the object to track
     * @param pollInterval the interval between status observations
     * @param <U> the type of object being tracked
     * @return a new tracker registered with this context's monitor
     * @deprecated StatusTrackers must belong to a TrackerScope. Use {@link TrackerScope#trackTask} instead.
     *             Create a scope first with {@link #createScope(String)}, then track tasks within it.
     */
    @Deprecated
    public <U extends StatusSource<U>> StatusTracker<U> track(U tracked, Duration pollInterval) {
        throw new UnsupportedOperationException(
            "StatusTrackers must belong to a TrackerScope. " +
            "Use createScope(name).trackTask(tracked, pollInterval) instead.");
    }

    /**
     * Creates a root-level tracker for any object using a custom status function.
     * Uses the context's default poll interval.
     *
     * @param tracked the object to track
     * @param statusFunction function to extract status from the tracked object
     * @param <T> the type of object being tracked
     * @return a new tracker registered with this context's monitor
     * @deprecated StatusTrackers must belong to a TrackerScope. Use {@link TrackerScope#trackTask} instead.
     *             Create a scope first with {@link #createScope(String)}, then track tasks within it.
     */
    @Deprecated
    public <T> StatusTracker<T> track(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        throw new UnsupportedOperationException(
            "StatusTrackers must belong to a TrackerScope. " +
            "Use createScope(name).trackTask(tracked, statusFunction) instead.");
    }

    /**
     * Creates a root-level tracker for any object using a custom status function and poll interval.
     *
     * @param tracked the object to track
     * @param statusFunction function to extract status from the tracked object
     * @param pollInterval the interval between status observations
     * @param <T> the type of object being tracked
     * @return a new tracker registered with this context's monitor
     * @deprecated StatusTrackers must belong to a TrackerScope. Use {@link TrackerScope#trackTask} instead.
     *             Create a scope first with {@link #createScope(String)}, then track tasks within it.
     */
    @Deprecated
    public <T> StatusTracker<T> track(T tracked,
                                      Function<T, StatusUpdate<T>> statusFunction,
                                      Duration pollInterval) {
        throw new UnsupportedOperationException(
            "StatusTrackers must belong to a TrackerScope. " +
            "Use createScope(name).trackTask(tracked, statusFunction, pollInterval) instead.");
    }

    // Package-private methods for TrackerScope to create tasks
    <U extends StatusSource<U>> StatusTracker<U> trackInScope(TrackerScope scope, U tracked) {
        return trackInScope(scope, tracked, StatusSource::getTaskStatus, null);
    }

    <U extends StatusSource<U>> StatusTracker<U> trackInScope(TrackerScope scope, U tracked, Duration pollInterval) {
        return trackInScope(scope, tracked, StatusSource::getTaskStatus, pollInterval);
    }

    <T> StatusTracker<T> trackInScope(TrackerScope scope,
                                      T tracked,
                                      Function<T, StatusUpdate<T>> statusFunction) {
        return trackInScope(scope, tracked, statusFunction, null);
    }

    <T> StatusTracker<T> trackInScope(TrackerScope scope,
                                      T tracked,
                                      Function<T, StatusUpdate<T>> statusFunction,
                                      Duration pollInterval) {
        return createTracker(scope, tracked, statusFunction, pollInterval);
    }

    private <T> StatusTracker<T> createTracker(TrackerScope scope,
                                               T tracked,
                                               Function<T, StatusUpdate<T>> statusFunction,
                                               Duration requestedInterval) {
        checkNotClosed();
        Objects.requireNonNull(scope, "scope - StatusTrackers must belong to a TrackerScope");

        Duration effectiveInterval = Objects.requireNonNullElse(requestedInterval, defaultPollInterval);

        if (scope.getContext() != this) {
            throw new IllegalArgumentException("Scope belongs to a different StatusContext");
        }

        StatusTracker<T> tracker = new StatusTracker<>(this, scope, tracked, statusFunction, effectiveInterval);
        StatusUpdate<T> initial = tracker.refreshAndGetStatus();
        activeTrackers.add(tracker);
        monitor.register(tracker, effectiveInterval, initial);
        taskStarted(tracker);
        return tracker;
    }

    /**
     * Adds a sink to receive status updates for all trackers in this context.
     * The sink will immediately begin receiving events for existing trackers
     * and all future trackers.
     *
     * @param sink the sink to add, ignored if null or already registered
     * @throws IllegalStateException if this context has been closed
     */
    public void addSink(StatusSink sink) {
        checkNotClosed();
        if (sink != null && !sinks.contains(sink)) {
            sinks.add(sink);
        }
    }

    /**
     * Removes a sink from this context. After removal, the sink will no longer
     * receive status updates.
     *
     * @param sink the sink to remove
     * @throws IllegalStateException if this context has been closed
     */
    public void removeSink(StatusSink sink) {
        checkNotClosed();
        sinks.remove(sink);
    }

    /**
     * Returns a snapshot of all registered sinks. The returned list is a defensive
     * copy and will not reflect subsequent additions or removals.
     *
     * @return an immutable snapshot of registered sinks
     */
    public List<StatusSink> getSinks() {
        return new ArrayList<>(sinks);
    }

    /**
     * Returns the number of currently active (not closed) trackers.
     *
     * @return the count of active trackers
     */
    public int getActiveTrackerCount() {
        return activeTrackers.size();
    }

    /**
     * Returns a snapshot of all active trackers. The returned list is a defensive
     * copy and will not reflect subsequent tracker additions or closures.
     *
     * @return an immutable snapshot of active trackers
     */
    public List<StatusTracker<?>> getActiveTrackers() {
        return new ArrayList<>(activeTrackers);
    }

    /**
     * Returns the name of this context.
     *
     * @return the context name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the default poll interval used when creating trackers without
     * an explicit interval.
     *
     * @return the default poll interval
     */
    public Duration getDefaultPollInterval() {
        return defaultPollInterval;
    }

    /**
     * Callback invoked when a tracker is closed. Unregisters the tracker from
     * monitoring and removes it from the active tracker list.
     *
     * @param tracker the tracker that was closed
     */
    void onTrackerClosed(StatusTracker<?> tracker) {
        monitor.unregister(tracker);
        activeTrackers.remove(tracker);
        taskFinished(tracker);

        // Remove from parent scope if it has one
        TrackerScope parentScope = tracker.getParentScope();
        if (parentScope != null) {
            parentScope.removeChildTask(tracker);
        }
    }

    /**
     * Callback invoked when a scope is closed. Removes the scope from
     * the active scope list.
     *
     * @param scope the scope that was closed
     */
    void onScopeClosed(TrackerScope scope) {
        activeScopes.remove(scope);
        scopeFinished(scope);
    }

    private void scopeFinished(TrackerScope scope) {
        notifySinks(sink -> {
            // Use reflection to avoid circular dependency
            try {
                var method = sink.getClass().getMethod("scopeFinished", TrackerScope.class);
                method.invoke(sink, scope);
            } catch (NoSuchMethodException e) {
                // Sink doesn't support scopes - ignore
            } catch (Exception e) {
                // Log but don't fail
            }
        }, "notifying sink of scope finish");
    }

    private List<StatusSink> snapshotSinks() {
        return new ArrayList<>(sinks);
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("StatusContext '" + name + "' has been closed");
        }
    }

    /**
     * Routes a status update from the monitor to all registered sinks.
     * This method implements the unidirectional status flow: the monitor observes
     * the tracked object, the tracker caches the status, and this method forwards
     * it to all sinks for processing.
     * <p>
     * Called by {@link StatusMonitor} after {@link StatusTracker#refreshAndGetStatus()}.
     *
     * @param tracker the tracker reporting the status
     * @param status the observed status
     * @param <T> the type of object being tracked
     */
    <T> void pushStatus(StatusTracker<T> tracker, StatusUpdate<T> status) {
        if (tracker == null || status == null) {
            return;
        }

        // Status flows unidirectionally: Monitor observes → Context routes → Sinks receive
        // (Tracker already cached the status in refreshAndGetStatus())
        taskUpdate(tracker, status);
    }

    /**
     * Notifies all registered sinks that a task has started tracking.
     * This method is part of the {@link StatusSink} interface and is called
     * when a new tracker is created.
     *
     * @param task the tracker that started
     */
    @Override
    public void taskStarted(StatusTracker<?> task) {
        notifySinks(sink -> sink.taskStarted(task), "notifying sink of task start");
    }

    /**
     * Notifies all registered sinks of a status update.
     * This method is part of the {@link StatusSink} interface and is called
     * by {@link #pushStatus} when the monitor provides a new status observation.
     *
     * @param task the tracker reporting the update
     * @param status the new status
     */
    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        notifySinks(sink -> sink.taskUpdate(task, status), "notifying sink of status change");
    }

    /**
     * Notifies all registered sinks that a task has finished.
     * This method is part of the {@link StatusSink} interface and is called
     * when a tracker is closed.
     *
     * @param task the tracker that finished
     */
    @Override
    public void taskFinished(StatusTracker<?> task) {
        notifySinks(sink -> sink.taskFinished(task), "notifying sink of task finish");
    }

    /**
     * Helper method to safely notify all sinks, catching and logging any exceptions
     * to prevent one failing sink from affecting others.
     *
     * @param sinkAction the action to perform on each sink
     * @param errorContext description of the action for error messages
     */
    private void notifySinks(Consumer<StatusSink> sinkAction, String errorContext) {
        for (StatusSink sink : snapshotSinks()) {
            try {
                sinkAction.accept(sink);
            } catch (Exception e) {
                System.err.println("Error " + errorContext + ": " + e.getMessage());
            }
        }
    }

    /**
     * Closes this context and all associated resources. This method:
     * <ol>
     *   <li>Closes all active trackers (which notifies sinks)</li>
     *   <li>Stops the monitoring thread</li>
     *   <li>Clears all registered sinks</li>
     * </ol>
     * <p>
     * This method is idempotent and safe to call multiple times.
     * After closing, attempts to create new trackers or modify sinks will throw
     * {@link IllegalStateException}.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        // Close trackers first so they can signal sinks before monitors are torn down.
        for (StatusTracker<?> tracker : new ArrayList<>(activeTrackers)) {
            try {
                tracker.close();
            } catch (Exception e) {
                System.err.println("Error closing tracker: " + e.getMessage());
            }
        }
        activeTrackers.clear();

        monitor.close();
        sinks.clear();
    }

    /**
     * Returns whether this context has been closed.
     *
     * @return true if {@link #close()} has been called, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Returns the path identifier for this context (name prefixed with slash).
     *
     * @return the path identifier
     */
    public String getPath() {
        return "/" + name;
    }
}
