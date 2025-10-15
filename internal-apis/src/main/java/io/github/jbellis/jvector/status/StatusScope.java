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

import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * An organizational container for grouping related tasks in a hierarchical structure.
 * Unlike {@link StatusTracker}, a StatusScope has no progress or state of its own -
 * it serves purely as an umbrella for organizing child tasks and nested scopes.
 *
 * <p><strong>Architectural Role:</strong></p>
 * <ul>
 *   <li><strong>Scopes</strong> are organizational nodes that can contain:
 *     <ul>
 *       <li>Child scopes (for nested organization)</li>
 *       <li>Task trackers (actual work units)</li>
 *     </ul>
 *   </li>
 *   <li><strong>Task Trackers</strong> are leaf nodes representing actual work:
 *     <ul>
 *       <li>Have progress and state (PENDING, RUNNING, SUCCESS, etc.)</li>
 *       <li>Cannot have children (enforced by design)</li>
 *       <li>Report status updates via {@link StatusSource}</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * <p><strong>Hierarchy Example:</strong></p>
 * <pre>
 * DataPipeline (Scope)
 *   ├─ Ingestion (Scope)
 *   │    ├─ LoadCSV (Task: 45% complete)
 *   │    └─ ValidateSchema (Task: 100% complete)
 *   └─ Processing (Scope)
 *        ├─ Transform (Task: 30% complete)
 *        └─ Index (Task: PENDING)
 * </pre>
 *
 * <p><strong>Usage Example with StatusContext:</strong></p>
 * <pre>{@code
 * try (StatusContext context = new StatusContext("pipeline");
 *      StatusScope dataScope = context.createScope("DataPipeline")) {
 *
 *     // Create nested organizational scopes
 *     StatusScope ingestionScope = dataScope.createChildScope("Ingestion");
 *     StatusScope processingScope = dataScope.createChildScope("Processing");
 *
 *     // Add actual tasks as leaf nodes
 *     StatusTracker<LoadTask> loadTracker = ingestionScope.trackTask(new LoadTask());
 *     StatusTracker<ValidateTask> validateTracker = ingestionScope.trackTask(new ValidateTask());
 *
 *     // Execute tasks...
 * }
 * }</pre>
 *
 * <p><strong>Usage Example with Default Context:</strong></p>
 * <pre>{@code
 * // Create a standalone scope with its own default context
 * try (StatusScope scope = new StatusScope("my-work")) {
 *     // Access the auto-created context if needed
 *     StatusContext context = scope.getContext();
 *     context.addSink(new ConsoleLoggerSink());
 *
 *     // Create tasks directly
 *     StatusTracker<Task> tracker = scope.trackTask(new Task());
 *     tracker.getTracked().execute();
 * }
 * }</pre>
 *
 * <p><strong>Completion Semantics:</strong></p>
 * A scope is considered "complete" when all its children (both nested scopes and tasks) are complete.
 * This provides a natural aggregation of completion state without needing the scope itself to track progress.
 *
 * <p><strong>Thread Safety:</strong></p>
 * <ul>
 *   <li><strong>All public methods are thread-safe</strong> and can be called concurrently</li>
 *   <li><strong>Child management:</strong> Uses {@link CopyOnWriteArrayList} for thread-safe child collections</li>
 *   <li><strong>Concurrent scope creation:</strong> Multiple threads can create child scopes simultaneously</li>
 *   <li><strong>Concurrent tracker creation:</strong> Multiple threads can create task trackers simultaneously</li>
 *   <li><strong>Completion checking:</strong> {@link #isComplete()} is safe to call while children are being added/closed</li>
 *   <li><strong>Close operation:</strong> Idempotent and safe to call from any thread</li>
 *   <li><strong>Parent-child relationships:</strong> Maintained atomically via CopyOnWriteArrayList operations</li>
 * </ul>
 *
 * @see StatusTracker
 * @see StatusContext
 * @since 4.0.0
 */
public final class StatusScope implements AutoCloseable {

    private final StatusContext context;
    private final StatusScope parent;
    private final String name;
    private final List<StatusScope> childScopes = new CopyOnWriteArrayList<>();
    private final List<StatusTracker<?>> childTasks = new CopyOnWriteArrayList<>();
    private volatile boolean closed = false;
    private final boolean ownsContext;

    /**
     * Creates a root StatusScope with its own default StatusContext.
     * The context will be automatically closed when this scope is closed.
     * This is useful for simple cases where you don't need to configure
     * the context explicitly.
     *
     * @param name the name of the scope (also used as the context name)
     */
    public StatusScope(String name) {
        this.context = new StatusContext(name);
        this.parent = null;
        this.name = name;
        this.ownsContext = true;
        context.registerScope(this);
    }

    /**
     * Package-private constructor. Use {@link StatusContext#createScope(String)} or
     * {@link #createChildScope(String)} to create scopes within an existing context.
     */
    StatusScope(StatusContext context, StatusScope parent, String name) {
        this.context = context;
        this.parent = parent;
        this.name = name;
        this.ownsContext = false;

        if (parent != null) {
            parent.childScopes.add(this);
        }
    }

    /**
     * Creates a nested organizational scope under this scope.
     *
     * @param name the name of the child scope
     * @return a new StatusScope as a child of this scope
     */
    public StatusScope createChildScope(String name) {
        checkNotClosed();
        return context.createChildScope(this, name);
    }

    /**
     * Creates a task tracker for an object implementing {@link StatusSource}.
     * The task becomes a leaf node under this scope.
     *
     * @param tracked the object to track
     * @param <U> the type of object being tracked
     * @return a new StatusTracker as a child of this scope
     */
    public <U extends StatusSource<U>> StatusTracker<U> trackTask(U tracked) {
        checkNotClosed();
        StatusTracker<U> tracker = context.trackInScope(this, tracked);
        childTasks.add(tracker);
        return tracker;
    }

    /**
     * Creates a task tracker using a custom status function.
     *
     * @param tracked the object to track
     * @param statusFunction function to extract status from the tracked object
     * @param <T> the type of object being tracked
     * @return a new StatusTracker as a child of this scope
     */
    public <T> StatusTracker<T> trackTask(T tracked, Function<T, StatusUpdate<T>> statusFunction) {
        checkNotClosed();
        StatusTracker<T> tracker = context.trackInScope(this, tracked, statusFunction);
        childTasks.add(tracker);
        return tracker;
    }

    /**
     * Returns the name of this scope.
     *
     * @return the scope name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the parent scope, or null if this is a root scope.
     *
     * @return the parent scope or null
     */
    public StatusScope getParent() {
        return parent;
    }

    /**
     * Returns the context that owns this scope.
     * For scopes created with the default constructor, this returns
     * the auto-created context.
     *
     * @return the owning context
     */
    public StatusContext getContext() {
        return context;
    }

    /**
     * Returns a snapshot of all child scopes.
     *
     * @return an immutable list of child scopes
     */
    public List<StatusScope> getChildScopes() {
        return new ArrayList<>(childScopes);
    }

    /**
     * Returns a snapshot of all child tasks.
     *
     * @return an immutable list of child task trackers
     */
    public List<StatusTracker<?>> getChildTasks() {
        return new ArrayList<>(childTasks);
    }

    /**
     * Returns whether all children (scopes and tasks) are complete.
     * A scope is complete when:
     * <ul>
     *   <li>All child scopes are complete, AND</li>
     *   <li>All child tasks are finished (have finishTime > 0)</li>
     * </ul>
     *
     * @return true if this scope and all descendants are complete
     */
    public boolean isComplete() {
        // Check all child scopes are complete
        for (StatusScope childScope : childScopes) {
            if (!childScope.isComplete()) {
                return false;
            }
        }

        // Check all child tasks are finished
        for (StatusTracker<?> task : childTasks) {
            if (!task.isClosed()) {
                return false;
            }
        }

        return true;
    }

    /**
     * Returns whether this scope has been closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Closes this scope, removing it from its parent's child list.
     * If this scope owns its context (created via the default constructor),
     * the context will also be closed, which closes all associated resources.
     * Child scopes and tasks are not automatically closed.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        if (parent != null) {
            parent.childScopes.remove(this);
        }

        context.onScopeClosed(this);

        // If we own the context, close it as well
        if (ownsContext) {
            context.close();
        }
    }

    /**
     * Called by child tracker when it closes to remove itself from this scope's child list.
     */
    void removeChildTask(StatusTracker<?> tracker) {
        childTasks.remove(tracker);
    }

    /**
     * Package-private method to add a tracker to this scope's child list.
     * Used by StatusTracker's public constructor when creating a standalone tracker.
     */
    void addChildTask(StatusTracker<?> tracker) {
        childTasks.add(tracker);
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("StatusScope '" + name + "' has been closed");
        }
    }

    /**
     * Returns a summary string showing the scope name and task counts.
     * Format: "ScopeName (active: N, done: M)"
     */
    @Override
    public String toString() {
        long active = childTasks.stream().filter(t -> !t.isClosed()).count();
        long done = childTasks.stream().filter(StatusTracker::isClosed).count();

        // Recursively count from child scopes
        for (StatusScope childScope : childScopes) {
            active += childScope.childTasks.stream().filter(t -> !t.isClosed()).count();
            done += childScope.childTasks.stream().filter(StatusTracker::isClosed).count();
        }

        if (active == 0 && done == 0) {
            return name; // No tasks yet
        }
        return String.format("%s (active: %d, done: %d)", name, active, done);
    }
}
