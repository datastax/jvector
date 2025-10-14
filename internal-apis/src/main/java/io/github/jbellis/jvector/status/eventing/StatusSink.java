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

package io.github.jbellis.jvector.status.eventing;

import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.sinks.LoggerStatusSink;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.sinks.NoopStatusSink;

/**
 * A contract for objects that receive task lifecycle events and status updates.
 * TaskSink implementations define how task monitoring information is processed,
 * stored, or displayed. This interface enables flexible monitoring strategies
 * by allowing multiple sinks to observe the same tasks.
 *
 * <p>The sink pattern allows for:
 * <ul>
 *   <li>Multiple concurrent monitoring outputs (console, logs, metrics, etc.)</li>
 *   <li>Pluggable monitoring strategies without changing task code</li>
 *   <li>Centralized event processing and filtering</li>
 *   <li>Integration with external monitoring systems</li>
 * </ul>
 *
 * <h2>Lifecycle Events</h2>
 * <p>Sinks receive three types of events in order:</p>
 * <ol>
 *   <li><strong>taskStarted:</strong> Called once when task tracking begins</li>
 *   <li><strong>taskUpdate:</strong> Called zero or more times as task progresses</li>
 *   <li><strong>taskFinished:</strong> Called once when task tracking ends</li>
 * </ol>
 *
 * <h2>Implementation Examples:</h2>
 *
 * <h3>Basic Console Sink</h3>
 * <pre>{@code
 * public class SimpleConsoleSink implements TaskSink {
 *     @Override
 *     public void taskStarted(Tracker<?> task) {
 *         System.out.println("Started: " + getTaskName(task));
 *     }
 *
 *     @Override
 *     public void taskUpdate(Tracker<?> task, TaskStatus<?> status) {
 *         System.out.printf("Update: %s [%.1f%%] %s%n",
 *             getTaskName(task), status.progress * 100, status.runstate);
 *     }
 *
 *     @Override
 *     public void taskFinished(Tracker<?> task) {
 *         System.out.println("Finished: " + getTaskName(task));
 *     }
 *
 *     private String getTaskName(Tracker<?> task) {
 *         return task.getTracked().toString();
 *     }
 * }
 * }</pre>
 *
 * <h3>Filtering Sink</h3>
 * <pre>{@code
 * public class FilteringSink implements TaskSink {
 *     private final TaskSink delegate;
 *     private final Predicate<Tracker<?>> filter;
 *
 *     public FilteringSink(TaskSink delegate, Predicate<Tracker<?>> filter) {
 *         this.delegate = delegate;
 *         this.filter = filter;
 *     }
 *
 *     @Override
 *     public void taskStarted(Tracker<?> task) {
 *         if (filter.test(task)) {
 *             delegate.taskStarted(task);
 *         }
 *     }
 *
 *     @Override
 *     public void taskUpdate(Tracker<?> task, TaskStatus<?> status) {
 *         if (filter.test(task) && status.progress > 0.1) { // Only > 10%
 *             delegate.taskUpdate(task, status);
 *         }
 *     }
 *
 *     @Override
 *     public void taskFinished(Tracker<?> task) {
 *         if (filter.test(task)) {
 *             delegate.taskFinished(task);
 *         }
 *     }
 * }
 * }</pre>
 *
 * <h3>Composite Sink</h3>
 * <pre>{@code
 * public class CompositeSink implements TaskSink {
 *     private final List<TaskSink> sinks;
 *
 *     public CompositeSink(TaskSink... sinks) {
 *         this.sinks = Arrays.asList(sinks);
 *     }
 *
 *     @Override
 *     public void taskStarted(Tracker<?> task) {
 *         sinks.forEach(sink -> {
 *             try {
 *                 sink.taskStarted(task);
 *             } catch (Exception e) {
 *                 handleSinkError(sink, e);
 *             }
 *         });
 *     }
 *
 *     // Similar for taskUpdate and taskFinished...
 * }
 * }</pre>
 *
 * <h2>Built-in Implementations</h2>
 * <p>The framework provides several ready-to-use sink implementations:</p>
 * <ul>
 *   <li>{@link ConsoleLoggerSink} - Human-readable console output</li>
 *   <li>{@link LoggerStatusSink} - Java logging integration</li>
 *   <li>{@link MetricsStatusSink} - Performance metrics collection</li>
 *   <li>{@link NoopStatusSink} - No-operation for testing</li>
 * </ul>
 *
 * <h2>Integration Patterns</h2>
 *
 * <h3>StatusContext Configuration</h3>
 * <pre>{@code
 * StatusContext context = new StatusContext("data-processing");
 * context.addSink(new ConsoleTaskSink());
 * context.addSink(new MetricsTaskSink());
 * context.addSink(new LoggerTaskSink("app.tasks"));
 * }</pre>
 *
 * <h3>Direct Tracker Usage</h3>
 * <pre>{@code
 * List<TaskSink> sinks = Arrays.asList(
 *     new ConsoleTaskSink(),
 *     new MetricsTaskSink()
 * );
 * try (Tracker<MyTask> tracker = Tracker.withInstrumented(task, sinks)) {
 *     task.execute();
 * }
 * }</pre>
 *
 * <h2>Error Handling</h2>
 * <p>Sink implementations should handle errors gracefully:</p>
 * <ul>
 *   <li>Catch and log exceptions rather than propagating them</li>
 *   <li>Continue processing even if individual operations fail</li>
 *   <li>Provide fallback behavior for resource unavailability</li>
 *   <li>Consider circuit breaker patterns for external integrations</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>Sink implementations must be thread-safe as they may receive concurrent
 * calls from multiple tracker threads. Use appropriate synchronization or
 * concurrent data structures as needed.</p>
 *
 * <h2>Performance Considerations</h2>
 * <p>Since sink methods are called on the monitoring thread:</p>
 * <ul>
 *   <li>Keep method implementations fast to avoid blocking task monitoring</li>
 *   <li>Use asynchronous processing for expensive operations (I/O, network)</li>
 *   <li>Consider batching updates for high-frequency scenarios</li>
 *   <li>Implement back-pressure mechanisms if needed</li>
 * </ul>
 *
 * @see StatusTracker
 * @see StatusUpdate
 * @see StatusContext
 * @since 4.0.0
 */
public interface StatusSink {
    /**
     * Called when task tracking begins.
     * This is the first method called for any tracked task.
     *
     * @param task the tracker instance that started monitoring
     */
    void taskStarted(StatusTracker<?> task);

    /**
     * Called when the task status changes.
     * This method may be called zero or more times during task execution.
     *
     * @param task the tracker instance reporting the update
     * @param status the current task status with progress and state information
     */
    void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status);

    /**
     * Called when task tracking ends.
     * This is the final method called for any tracked task.
     *
     * @param task the tracker instance that finished monitoring
     */
    void taskFinished(StatusTracker<?> task);

    /**
     * Called when a scope starts. Scopes provide hierarchical organization
     * of related tasks without having their own progress or state.
     * <p>
     * This method has a default no-op implementation to maintain backward
     * compatibility with sinks that don't need scope notifications.
     *
     * @param scope the scope that started
     * @since 4.0.0
     */
    default void scopeStarted(TrackerScope scope) {
        // No-op by default - sinks can override to handle scope events
    }

    /**
     * Called when a scope finishes. This is called after all tasks and
     * child scopes within this scope have completed.
     * <p>
     * This method has a default no-op implementation to maintain backward
     * compatibility with sinks that don't need scope notifications.
     *
     * @param scope the scope that finished
     * @since 4.0.0
     */
    default void scopeFinished(TrackerScope scope) {
        // No-op by default - sinks can override to handle scope events
    }
}
