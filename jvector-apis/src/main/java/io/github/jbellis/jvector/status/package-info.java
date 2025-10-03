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

/**
 * A comprehensive task tracking and monitoring framework designed for concurrent applications.
 * This package provides live status tracking capabilities with flexible monitoring strategies,
 * hierarchical task organization, and pluggable output systems.
 *
 * <h2>Design Goals</h2>
 * <p>The framework is built around several core principles:</p>
 * <ul>
 *   <li><strong>Real-time Feedback:</strong> Enable tactile user feedback on the status of ongoing tasks
 *       through multiple output channels (console, logging, metrics)</li>
 *   <li><strong>Performance Monitoring:</strong> Provide efficient instrumentation for internal JVector use
 *       with minimal overhead and comprehensive task lifecycle tracking</li>
 *   <li><strong>Hierarchical Organization:</strong> Support complex task structures including batch processing,
 *       nested task groups, and DAG-based execution patterns</li>
 *   <li><strong>Pluggable Architecture:</strong> Enable consumer-agnostic monitoring through flexible sink
 *       implementations and configurable scope management</li>
 * </ul>
 *
 * <h2>Core Architecture</h2>
 *
 * <h3>Primary Components</h3>
 * <ul>
 *   <li><strong>{@link io.github.jbellis.jvector.status.StatusTracker}</strong> - The main monitoring wrapper
 *       that provides background polling and sink notification for any task object</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.StatusContext}</strong> - Configuration and
 *       lifecycle management for creating tracker instances; also owns the monitoring threads</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.eventing.StatusSink}</strong> - Processing interface for
 *       task lifecycle events, enabling multiple concurrent monitoring strategies</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.StatusMonitor}</strong> - Background polling engine
 *       that continuously monitors task status and detects changes</li>
 * </ul>
 *
 * <h3>Supporting Types</h3>
 * <ul>
 *   <li><strong>{@link io.github.jbellis.jvector.status.eventing.StatusUpdate}</strong> - Immutable status
 *       snapshots containing progress, state, and timestamp information for monitoring</li>
 * </ul>
 *
 * <h2>Built-in Implementations</h2>
 *
 * <h3>Monitoring Sinks</h3>
 * <ul>
 *   <li><strong>{@link io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink}</strong> - Human-readable
 *       console output with progress bars and configurable formatting</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.sinks.LoggerStatusSink}</strong> - Java logging
 *       framework integration with configurable levels and hierarchies</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.sinks.MetricsStatusSink}</strong> - Performance
 *       metrics collection with detailed statistics and reporting</li>
 *   <li><strong>{@link io.github.jbellis.jvector.status.sinks.NoopStatusSink}</strong> - Zero-overhead
 *       no-operation sink for testing and performance-critical scenarios</li>
 * </ul>
 *
 * <h2>Essential Features</h2>
 *
 * <h3>Lifecycle Consistency</h3>
 * <p>The framework ensures tracking state never goes stale with respect to tracked tasks:</p>
 * <ul>
 *   <li>Automatic resource cleanup through {@link java.lang.AutoCloseable} integration</li>
 *   <li>Thread-safe lifecycle management with atomic state transitions</li>
 *   <li>Proper cleanup on task completion, failure, or cancellation</li>
 *   <li>Background thread management with daemon thread patterns</li>
 * </ul>
 *
 * <h3>Hierarchical Task Support</h3>
 * <p>Support for complex task structures beyond single task awareness:</p>
 * <ul>
 *   <li>Nested task groups with inherited configuration and sink management</li>
 *   <li>Batch processing patterns with concurrent task execution</li>
 *   <li>DAG (Directed Acyclic Graph) compatibility for complex workflows</li>
 *   <li>Scope-based configuration inheritance with override capabilities</li>
 * </ul>
 *
 * <h3>Consumer-Agnostic Design</h3>
 * <p>Flexible monitoring strategies through indirect observer wiring:</p>
 * <ul>
 *   <li>Multiple concurrent monitoring outputs (console, logs, metrics, external systems)</li>
 *   <li>Pluggable sink architecture for custom monitoring implementations</li>
 *   <li>Configuration-driven monitoring behavior without code changes</li>
 *   <li>Environment-specific monitoring strategies (dev, staging, production)</li>
 * </ul>
 *
 * <h2>Advanced Features</h2>
 *
 * <h3>Performance Optimization</h3>
 * <ul>
 *   <li>Configurable polling intervals to balance responsiveness and overhead</li>
 *   <li>Lock-free data structures for high-concurrency scenarios</li>
 *   <li>Memory-efficient design with minimal per-task overhead</li>
 *   <li>Zero-cost abstractions when monitoring is disabled</li>
 * </ul>
 *
 * <h3>Integration Flexibility</h3>
 * <ul>
 *   <li>Support for both instrumented objects (implementing TaskStatus.Provider)
 *       and custom status extraction functions</li>
 *   <li>Dynamic sink addition/removal during task execution</li>
 *   <li>Compatibility with existing task frameworks and execution models</li>
 *   <li>Minimal intrusion on existing codebases</li>
 * </ul>
 *
 * <h3>Monitoring Capabilities</h3>
 * <ul>
 *   <li>Real-time progress tracking with customizable update frequencies</li>
 *   <li>Automatic completion detection and monitoring termination</li>
 *   <li>Exception handling and error recovery in monitoring code</li>
 *   <li>Performance metrics collection and analysis</li>
 * </ul>
 *
 * <h2>Usage Patterns</h2>
 *
 * <h3>Simple Task Monitoring</h3>
 * <pre>{@code
 * // Basic tracking with console output
 * try (Tracker<MyTask> tracker = Tracker.withInstrumented(task, new ConsoleTaskSink())) {
 *     task.execute();
 * } // Automatic cleanup
 * }</pre>
 *
 * <h3>Context-Based Configuration</h3>
 * <pre>{@code
 * // Hierarchical configuration with multiple sinks
 * StatusContext context = new StatusContext("data-processing");
 * context.addSink(new ConsoleTaskSink());
 * context.addSink(new MetricsTaskSink());
 *
 * try (StatusTracker<DataProcessor> tracker = context.track(processor)) {
 *     processor.processLargeDataset();
 * }
 * }</pre>
 *
 * <h3>Batch Processing</h3>
 * <pre>{@code
 * // Multiple concurrent tasks with scope management
 * try (StatusContext context = new StatusContext("batch-job")) {
 *     List<StatusTracker<MyTask>> trackers = createTasks(context);
 *
 *     CompletableFuture.allOf(
 *         trackers.stream()
 *              .map(tracker -> CompletableFuture.runAsync(() -> processTask(tracker)))
 *              .toArray(CompletableFuture[]::new)
 *     ).join();
 * } // All trackers automatically cleaned up
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>All components in this package are designed for thread-safe operation:</p>
 * <ul>
 *   <li>Concurrent access to task collections and monitoring state</li>
 *   <li>Atomic operations for lifecycle management</li>
 *   <li>Proper synchronization in sink implementations</li>
 *   <li>Safe cleanup during concurrent task execution</li>
 * </ul>
 *
 * @since 4.0.0
 * @see io.github.jbellis.jvector.status.StatusTracker
 * @see io.github.jbellis.jvector.status.StatusContext
 * @see io.github.jbellis.jvector.status.eventing.StatusSink
 */
package io.github.jbellis.jvector.status;

import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.sinks.LoggerStatusSink;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.sinks.NoopStatusSink;
