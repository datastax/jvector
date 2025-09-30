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
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * TrackerScope manages configuration and lifecycle for creating Tracker instances.
 *
 * Scopes can be organized in a tree structure where child scopes inherit
 * configuration from parent scopes. This allows for hierarchical organization
 * of tracking configuration while keeping individual Tracker instances lightweight
 * and short-lived.
 *
 * TrackerScope instances are long-lived and hold:
 * - TaskSinks for output/monitoring
 * - Default polling intervals
 * - Child scopes for hierarchical organization
 *
 * Tracker instances are short-lived and should be used with try-with-resources.
 */
public class TrackerScope implements AutoCloseable {

    private final String name;
    private final TrackerScope parent;
    private final List<StatusSink> sinks;
    private final List<StatusSink> inheritedSinks;
    private final Duration defaultPollInterval;
    private final ConcurrentHashMap<String, TrackerScope> children;
    private final CopyOnWriteArrayList<StatusTracker<?>> activeStatusTrackers;
    private volatile boolean closed = false;

    // Root scope constructor
    public TrackerScope(String name) {
        this(name, null, Duration.ofMillis(100), new ArrayList<>());
    }

    // Root scope with custom interval
    public TrackerScope(String name, Duration defaultPollInterval) {
        this(name, null, defaultPollInterval, new ArrayList<>());
    }

    // Root scope with sinks
    public TrackerScope(String name, List<StatusSink> sinks) {
        this(name, null, Duration.ofMillis(100), sinks);
    }

    // Root scope with interval and sinks
    public TrackerScope(String name, Duration defaultPollInterval, List<StatusSink> sinks) {
        this(name, null, defaultPollInterval, sinks);
    }

    // Child scope constructor
    private TrackerScope(String name, TrackerScope parent, Duration defaultPollInterval, List<StatusSink> sinks) {
        this.name = name;
        this.parent = parent;
        this.defaultPollInterval = defaultPollInterval != null ? defaultPollInterval : Duration.ofMillis(100);
        this.sinks = new CopyOnWriteArrayList<>(sinks);
        this.children = new ConcurrentHashMap<>();
        this.activeStatusTrackers = new CopyOnWriteArrayList<>();

        // Inherit sinks from parent
        if (parent != null) {
            this.inheritedSinks = new ArrayList<>(parent.getAllSinks());
        } else {
            this.inheritedSinks = new ArrayList<>();
        }
    }

    /**
     * Create a child scope that inherits this scope's configuration.
     * Child scopes inherit all parent sinks but can add their own.
     */
    public TrackerScope createChildScope(String childName) {
        return createChildScope(childName, defaultPollInterval, new ArrayList<>());
    }

    public TrackerScope createChildScope(String childName, List<StatusSink> additionalSinks) {
        return createChildScope(childName, defaultPollInterval, additionalSinks);
    }

    public TrackerScope createChildScope(String childName, Duration pollInterval, List<StatusSink> additionalSinks) {
        checkNotClosed();
        TrackerScope child = new TrackerScope(childName, this, pollInterval, additionalSinks);
        children.put(childName, child);
        return child;
    }

    /**
     * Create a Tracker for an instrumented object (implements TaskStatus.Provider).
     * The Tracker will use this scope's sinks and configuration.
     *
     * MUST be used with try-with-resources:
     * <pre>
     * try (Tracker&lt;MyTask&gt; tracker = scope.track(myTask)) {
     *     // Task executes
     * }
     * </pre>
     */
    public <U extends StatusUpdate.Provider<U>> StatusTracker<U> track(U tracked) {
        checkNotClosed();
        List<StatusSink> allSinks = getAllSinks();

        // Create tracker without automatic parent assignment
        // Parent-child relationships should be established explicitly via createChild()
        StatusTracker<U> statusTracker = StatusTracker.withInstrumented(tracked, defaultPollInterval, allSinks);

        activeStatusTrackers.add(statusTracker);
        // Set the scope so the tracker can remove itself when closing
        statusTracker.setScope(this);
        return statusTracker;
    }

    /**
     * Create a Tracker with a custom status function.
     * The Tracker will use this scope's sinks and configuration.
     */
    public <T> StatusTracker<T> track(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        checkNotClosed();
        List<StatusSink> allSinks = getAllSinks();

        // Create tracker without automatic parent assignment
        // Parent-child relationships should be established explicitly via createChild()
        StatusTracker<T> statusTracker = allSinks.isEmpty()
                ? StatusTracker.withFunctors(tracked, statusFunction)
                : StatusTracker.withFunctors(tracked, statusFunction, defaultPollInterval, allSinks);

        activeStatusTrackers.add(statusTracker);
        // Set the scope so the tracker can remove itself when closing
        statusTracker.setScope(this);
        return statusTracker;
    }

    /**
     * Create a Tracker with a custom polling interval.
     */
    public <U extends StatusUpdate.Provider<U>> StatusTracker<U> track(U tracked, Duration pollInterval) {
        checkNotClosed();
        List<StatusSink> allSinks = getAllSinks();

        // Create tracker without automatic parent assignment
        // Parent-child relationships should be established explicitly via createChild()
        StatusTracker<U> statusTracker = allSinks.isEmpty()
                ? StatusTracker.withInstrumented(tracked)
                : StatusTracker.withInstrumented(tracked, pollInterval, allSinks);

        activeStatusTrackers.add(statusTracker);
        // Set the scope so the tracker can remove itself when closing
        statusTracker.setScope(this);
        return statusTracker;
    }

    /**
     * Create a Tracker with a custom status function and polling interval.
     */
    public <T> StatusTracker<T> track(T tracked, Function<T, StatusUpdate<T>> statusFunction, Duration pollInterval) {
        checkNotClosed();
        List<StatusSink> allSinks = getAllSinks();

        // Create tracker without automatic parent assignment
        // Parent-child relationships should be established explicitly via createChild()
        StatusTracker<T> statusTracker = allSinks.isEmpty()
                ? StatusTracker.withFunctors(tracked, statusFunction)
                : StatusTracker.withFunctors(tracked, statusFunction, pollInterval, allSinks);

        activeStatusTrackers.add(statusTracker);
        // Set the scope so the tracker can remove itself when closing
        statusTracker.setScope(this);
        return statusTracker;
    }

    /**
     * Add a sink to this scope. It will be used by all Trackers created from this scope
     * and inherited by child scopes.
     */
    public void addSink(StatusSink sink) {
        checkNotClosed();
        if (sink != null && !sinks.contains(sink)) {
            sinks.add(sink);
        }
    }

    /**
     * Remove a sink from this scope.
     */
    public void removeSink(StatusSink sink) {
        checkNotClosed();
        sinks.remove(sink);
    }

    /**
     * Get all sinks for this scope (own + inherited).
     */
    public List<StatusSink> getAllSinks() {
        List<StatusSink> allSinks = new ArrayList<>();
        allSinks.addAll(inheritedSinks);
        allSinks.addAll(sinks);
        return allSinks;
    }

    /**
     * Get only the sinks defined at this scope level.
     */
    public List<StatusSink> getOwnSinks() {
        return new ArrayList<>(sinks);
    }

    public String getName() {
        return name;
    }

    public TrackerScope getParent() {
        return parent;
    }

    public TrackerScope getChild(String childName) {
        return children.get(childName);
    }

    public List<TrackerScope> getChildren() {
        return new ArrayList<>(children.values());
    }

    public Duration getDefaultPollInterval() {
        return defaultPollInterval;
    }

    // createTask method removed - SimpleTask moved to test package
    // Use your own task objects implementing TaskStatus.Provider instead

    /**
     * Gets the current number of active trackers in this scope.
     *
     * @return the number of active trackers
     */
    public int getActiveTrackerCount() {
        return activeStatusTrackers.size();
    }

    /**
     * Gets a copy of all active trackers in this scope.
     *
     * @return list of active trackers
     */
    public List<StatusTracker<?>> getActiveTrackers() {
        return new ArrayList<>(activeStatusTrackers);
    }

    /**
     * Internal method to remove a tracker from active tracking.
     */
    void removeTracker(StatusTracker<?> statusTracker) {
        activeStatusTrackers.remove(statusTracker);
    }

    /**
     * Get the full path of this scope in the tree.
     */
    public String getPath() {
        if (parent == null) {
            return "/" + name;
        }
        return parent.getPath() + "/" + name;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("TrackerScope '" + getPath() + "' has been closed");
        }
    }

    /**
     * Close this scope and all child scopes.
     * After closing, no new Trackers can be created from this scope.
     * Existing Trackers continue to function until they are closed.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;

            // Close all active trackers
            for (StatusTracker<?> statusTracker : activeStatusTrackers) {
                try {
                    statusTracker.close();
                } catch (Exception e) {
                    System.err.println("Error closing tracker: " + e.getMessage());
                }
            }
            activeStatusTrackers.clear();

            // Close all child scopes
            for (TrackerScope child : children.values()) {
                try {
                    child.close();
                } catch (Exception e) {
                    System.err.println("Error closing child scope " + child.getPath() + ": " + e.getMessage());
                }
            }

            children.clear();
            sinks.clear();
        }
    }

    public boolean isClosed() {
        return closed;
    }


    @Override
    public String toString() {
        return String.format("TrackerScope[path=%s, sinks=%d, children=%d, closed=%s]",
            getPath(), getAllSinks().size(), children.size(), closed);
    }

}