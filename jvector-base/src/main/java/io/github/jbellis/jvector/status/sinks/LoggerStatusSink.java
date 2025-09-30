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

import io.github.jbellis.jvector.status.StatusSink;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.TrackerScope;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.slf4j.LoggerFactory;

/**
 * A task sink that integrates with Logback logging framework to record task progress
 * and lifecycle events. This sink is ideal for production environments where task
 * monitoring needs to be integrated with existing logging infrastructure and
 * centralized log management systems.
 *
 * <p>This sink provides:
 * <ul>
 *   <li>Integration with Java's standard logging framework</li>
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
 * TaskSink loggerSink = new LoggerTaskSink("io.myapp.TaskProcessor");
 *
 * try (Tracker<DataProcessor> tracker = Tracker.withInstrumented(processor, loggerSink)) {
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
 * Logger customLogger = Logger.getLogger("app.background.tasks");
 * TaskSink debugSink = new LoggerTaskSink(customLogger, Level.FINE);
 *
 * // Configure logger for debug output
 * customLogger.setLevel(Level.FINE);
 * customLogger.addHandler(new ConsoleHandler());
 *
 * try (Tracker<BackgroundJob> tracker = Tracker.withInstrumented(job, debugSink)) {
 *     job.execute(); // Debug level logging
 * }
 * }</pre>
 *
 * <h3>Production Environment Setup</h3>
 * <pre>{@code
 * // Production configuration with WARNING level for critical tasks
 * TaskSink productionSink = new LoggerTaskSink("production.critical.tasks", Level.WARNING);
 *
 * TrackerScope criticalScope = new TrackerScope("critical-operations");
 * criticalScope.addSink(productionSink);
 *
 * try (Tracker<CriticalTask> tracker = criticalScope.track(criticalTask)) {
 *     criticalTask.execute(); // Only logs at WARNING level
 * }
 * }</pre>
 *
 * <h3>Multiple Loggers for Different Components</h3>
 * <pre>{@code
 * // Different loggers for different subsystems
 * TaskSink databaseSink = new LoggerTaskSink("app.database.operations", Level.INFO);
 * TaskSink networkSink = new LoggerTaskSink("app.network.operations", Level.INFO);
 * TaskSink fileSystemSink = new LoggerTaskSink("app.filesystem.operations", Level.FINE);
 *
 * // Use appropriate sink based on task type
 * try (Tracker<DatabaseTask> dbTracker = Tracker.withInstrumented(dbTask, databaseSink);
 *      Tracker<NetworkTask> netTracker = Tracker.withInstrumented(netTask, networkSink);
 *      Tracker<FileTask> fileTracker = Tracker.withInstrumented(fileTask, fileSystemSink)) {
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
 * Logger rootLogger = Logger.getLogger("com.mycompany.myapp");
 * TaskSink appSink = new LoggerTaskSink(rootLogger, Level.INFO);
 *
 * // Child logger inherits parent configuration
 * TaskSink moduleSpecificSink = new LoggerTaskSink("com.mycompany.myapp.processing");
 *
 * try (Tracker<AppTask> tracker = Tracker.withInstrumented(task, appSink)) {
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
 *   <li>Configurable output through standard logging.properties or programmatic setup</li>
 *   <li>Integration with log aggregation systems (ELK, Splunk, etc.)</li>
 *   <li>Level-based filtering for different environments (dev, staging, prod)</li>
 *   <li>Thread safety through Java's logging framework</li>
 * </ul>
 *
 * <h2>Best Practices</h2>
 * <ul>
 *   <li>Use hierarchical logger names for better organization (e.g., "app.module.component")</li>
 *   <li>Choose appropriate log levels (INFO for normal operations, FINE for debugging)</li>
 *   <li>Configure handlers and formatters to match your logging infrastructure</li>
 *   <li>Consider using different loggers for different types of tasks</li>
 *   <li>Test log output in different environments to ensure proper configuration</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>This sink is thread-safe through Logback's logging framework, which handles
 * concurrent access to loggers and their appenders.</p>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see TrackerScope
 * @see Logger
 * @since 4.0.0
 */
public class LoggerStatusSink implements StatusSink {

    private final Logger logger;
    private final Level level;

    public LoggerStatusSink(Logger logger) {
        this(logger, Level.INFO);
    }

    public LoggerStatusSink(Logger logger, Level level) {
        this.logger = logger;
        this.level = level;
    }

    public LoggerStatusSink(String loggerName) {
        this((Logger) LoggerFactory.getLogger(loggerName), Level.INFO);
    }

    public LoggerStatusSink(String loggerName, Level level) {
        this((Logger) LoggerFactory.getLogger(loggerName), level);
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
        String taskName = getTaskName(task);
        log("Task started: " + taskName);
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        String taskName = getTaskName(task);
        double progress = status.progress * 100;

        log(String.format("Task update: %s [%.1f%%] - %s", taskName, progress, status.runstate));
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        String taskName = getTaskName(task);
        log("Task finished: " + taskName);
    }

    private void log(String message) {
        // Use Logback's level-specific methods
        if (level == Level.TRACE) {
            logger.trace(message);
        } else if (level == Level.DEBUG) {
            logger.debug(message);
        } else if (level == Level.INFO) {
            logger.info(message);
        } else if (level == Level.WARN) {
            logger.warn(message);
        } else if (level == Level.ERROR) {
            logger.error(message);
        } else {
            // Default to info for any other level
            logger.info(message);
        }
    }

    private String getTaskName(StatusTracker<?> task) {
        Object tracked = task.getTracked();
        // SimpleTask moved to test package - use reflection for getName()
        try {
            java.lang.reflect.Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall through to default toString
        }
        return tracked.toString();
    }
}