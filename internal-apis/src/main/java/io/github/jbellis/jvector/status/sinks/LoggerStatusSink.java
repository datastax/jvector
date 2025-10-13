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

package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Objects;

/**
 * A task sink that integrates with Log4j 2 to record task progress
 * and lifecycle events. This sink is ideal for production environments where task
 * monitoring needs to be integrated with existing logging infrastructure and
 * centralized log management systems.
 *
 * <p>This sink provides:
 * <ul>
 *   <li>Integration with the Log4j 2 logging framework</li>
 *   <li>Configurable log levels for different environments</li>
 *   <li>Structured log messages with task names and progress</li>
 *   <li>Support for custom loggers and logger hierarchies</li>
 *   <li>Automatic task name extraction and formatting</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Basic Logging with Default Logger</h3>
 * <pre>{@code
 * // Uses the class name as logger name with INFO level
 * TaskSink loggerSink = new LoggerStatusSink("io.myapp.TaskProcessor");
 *
 * try (StatusTracker<DataProcessor> tracker = StatusTracker.withInstrumented(processor, loggerSink)) {
 *     processor.processData();
 *     // Log output:
 *     // INFO: Task started: data-processing
 *     // INFO: Task update: data-processing [45.0%] - RUNNING
 *     // INFO: Task finished: data-processing
 * }
 * }</pre>
 *
 * <h3>Custom Logger and Level</h3>
 * <pre>{@code
 * Logger customLogger = LogManager.getLogger("app.background.tasks");
 * TaskSink debugSink = new LoggerStatusSink(customLogger, Level.DEBUG);
 *
 * try (StatusTracker<BackgroundJob> tracker = StatusTracker.withInstrumented(job, debugSink)) {
*     job.execute(); // Debug level logging
* }
* }</pre>
 *
 * <h3>Production Environment Setup</h3>
 * <pre>{@code
 * // Production configuration with WARNING level for critical tasks
 * TaskSink productionSink = new LoggerStatusSink("production.critical.tasks", Level.WARN);
 *
 * StatusContext context = new StatusContext("critical-operations");
 * context.addSink(productionSink);
 *
 * try (StatusTracker<CriticalTask> tracker = context.track(criticalTask)) {
 *     criticalTask.execute(); // Only logs at WARNING level
 * }
 * }</pre>
 *
 * <h3>Multiple Loggers for Different Components</h3>
 * <pre>{@code
 * // Different loggers for different subsystems
 * TaskSink databaseSink = new LoggerStatusSink("app.database.operations", Level.INFO);
 * TaskSink networkSink = new LoggerStatusSink("app.network.operations", Level.INFO);
 * TaskSink fileSystemSink = new LoggerStatusSink("app.filesystem.operations", Level.DEBUG);
 *
 * // Use appropriate sink based on task type
 * try (StatusTracker<DatabaseTask> dbTracker = StatusTracker.withInstrumented(dbTask, databaseSink);
 *      StatusTracker<NetworkTask> netTracker = StatusTracker.withInstrumented(netTask, networkSink);
 *      StatusTracker<FileTask> fileTracker = StatusTracker.withInstrumented(fileTask, fileSystemSink)) {
 *
 *     CompletableFuture.allOf(
 *         CompletableFuture.runAsync(dbTask::execute),
 *         CompletableFuture.runAsync(netTask::execute),
 *         CompletableFuture.runAsync(fileTask::execute)
 *     ).join();
 * }
 * }</pre>
 *
 * <h3>Integration with Existing Logger Hierarchy</h3>
 * <pre>{@code
 * // Leverage existing logger configuration
 * Logger rootLogger = LogManager.getLogger("com.mycompany.myapp");
 * TaskSink appSink = new LoggerStatusSink(rootLogger, Level.INFO);
 *
 * // Child logger inherits parent configuration
 * TaskSink moduleSpecificSink = new LoggerStatusSink("com.mycompany.myapp.processing");
 *
 * try (StatusTracker<AppTask> tracker = StatusTracker.withInstrumented(task, appSink)) {
 *     task.run();
 * }
 * }</pre>
 *
 * <h2>Log Message Format</h2>
 * <p>The sink produces structured log messages with this format:</p>
 * <ul>
 *   <li><strong>Task Started:</strong> "Task started: [task-name]"</li>
 *   <li><strong>Task Update:</strong> "Task update: [task-name] [XX.X%] - [run-state]"</li>
 *   <li><strong>Task Finished:</strong> "Task finished: [task-name]"</li>
 * </ul>
 *
 * <h2>Logger Integration Benefits</h2>
 * <p>Using this sink provides several advantages in production environments:</p>
 * <ul>
 *   <li>Centralized log management through existing logging infrastructure</li>
 *   <li>Configurable output through Log4j 2 configuration or programmatic setup</li>
 *   <li>Integration with log aggregation systems (ELK, Splunk, etc.)</li>
 *   <li>Level-based filtering for different environments (dev, staging, prod)</li>
 *   <li>Thread safety provided by Log4j 2</li>
 * </ul>
 *
 * <h2>Best Practices</h2>
 * <ul>
 *   <li>Use hierarchical logger names for better organization (e.g., "app.module.component")</li>
 *   <li>Choose appropriate log levels (INFO for normal operations, DEBUG for detailed tracing)</li>
 *   <li>Configure appenders and layouts to match your logging infrastructure</li>
 *   <li>Consider using different loggers for different types of tasks</li>
 *   <li>Test log output in different environments to ensure proper configuration</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>This sink is thread-safe through Log4j 2, which handles concurrent access to
 * loggers and their appenders.</p>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see StatusContext
 * @see Logger
 * @since 4.0.0
 */
public class LoggerStatusSink implements StatusSink {

    private final Logger logger;
    private final Level level;

    public LoggerStatusSink() {
        this(LogManager.getLogger(LoggerStatusSink.class));
    }

    public LoggerStatusSink(Logger logger) {
        this(logger, Level.INFO);
    }

    public LoggerStatusSink(Logger logger, Level level) {
        this.logger = Objects.requireNonNull(logger, "logger");
        this.level = Objects.requireNonNullElse(level, Level.INFO);
    }

    public LoggerStatusSink(String loggerName) {
        this(LogManager.getLogger(loggerName));
    }

    public LoggerStatusSink(String loggerName, Level level) {
        this(LogManager.getLogger(loggerName), level);
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
        String taskName = StatusTracker.extractTaskName(task);
        log("Task started: " + taskName);
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        String taskName = StatusTracker.extractTaskName(task);
        double progress = status.progress * 100;

        log(String.format("Task update: %s [%.1f%%] - %s", taskName, progress, status.runstate));
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        String taskName = StatusTracker.extractTaskName(task);
        log("Task finished: " + taskName);
    }

    private void log(String message) {
        Level effectiveLevel = Objects.requireNonNullElse(level, Level.INFO);
        if (logger.isEnabled(effectiveLevel)) {
            logger.log(effectiveLevel, message);
        }
    }
}
