package io.github.jbellis.jvector.status.sinks;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.status.StatusSink;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TestableTask;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.StatusTracker;
import org.junit.Test;
import org.junit.jupiter.api.Tag;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.StreamHandler;

import static org.junit.Assert.*;

/**
 * Core TaskSink functionality tests with performance scenarios.
 *
 * <p>This test suite validates all built-in TaskSink implementations and their behaviors:
 * <ul>
 *   <li><strong>{@link NoopStatusSink}:</strong> No-operation sink for high-performance scenarios</li>
 *   <li><strong>{@link ConsoleStatusSink}:</strong> Human-readable console output with progress bars</li>
 *   <li><strong>{@link LoggerStatusSink}:</strong> Java logging framework integration</li>
 *   <li><strong>{@link MetricsStatusSink}:</strong> Performance metrics collection and statistics</li>
 * </ul>
 *
 * <h2>Test Coverage Areas:</h2>
 * <ul>
 *   <li>Basic sink functionality (task lifecycle events)</li>
 *   <li>Output format validation and correctness</li>
 *   <li>Configuration options and customization</li>
 *   <li>Error handling and resilience</li>
 *   <li>Concurrent access and thread safety</li>
 *   <li>Integration with Tracker and status monitoring</li>
 * </ul>
 *
 * <h2>Sink-Specific Testing:</h2>
 * <ul>
 *   <li><strong>NoopTaskSink:</strong> Zero-overhead performance and singleton behavior</li>
 *   <li><strong>ConsoleTaskSink:</strong> Output formatting, progress bars, timestamps</li>
 *   <li><strong>LoggerTaskSink:</strong> Logger integration, log levels, message formats</li>
 *   <li><strong>MetricsTaskSink:</strong> Statistics calculation, performance tracking, averages</li>
 * </ul>
 *
 * <p><strong>Integration Testing:</strong> This class also tests sink coordination patterns
 * including multiple sinks working together, sink error recovery, and dynamic sink management.
 *
 * @see StatusSink for the base interface
 * @see TrackerWithSinksTest for advanced Tracker-Sink integration
 * @see StatusTrackingIntegrationTest for end-to-end sink usage
 * @since 4.0.0
 */
@Tag("Core")
@Tag("Performance")
public class StatusSinkTest extends RandomizedTest {

    /**
     * Tests NoopTaskSink functionality and singleton behavior.
     *
     * <p><strong>Purpose:</strong> Validates that NoopTaskSink provides a true
     * no-operation implementation that can handle all TaskSink operations
     * safely without side effects, and that it correctly implements singleton pattern.
     *
     * <p><strong>Why Important:</strong> NoopTaskSink is used in high-performance
     * scenarios where sink overhead must be minimized. This test ensures
     * it provides zero-overhead operation and proper singleton behavior.
     *
     * <p><strong>Coverage:</strong> Singleton pattern verification, no-operation
     * behavior for all TaskSink methods, safety with all task lifecycle events.
     *
     * @see #testMultipleSinksConcurrently() for NoopTaskSink integration with other sinks
     * @see StatusTrackingIntegrationTest#testMixedSinkBehavior() for noop sink in mixed scenarios
     */
    @Test
    public void testNoopTaskSink() {
        StatusSink sink = NoopStatusSink.getInstance();

        assertNotNull(sink);
        assertSame(sink, NoopStatusSink.getInstance());

        TrackerScope scope = new TrackerScope("test-scope");
        TestableTask task = new TestableTask("test-task");
        task.setProgress(0.5);

        // Create a simple tracker for testing
        class TestTask implements StatusUpdate.Provider<TestTask> {
            @Override
            public StatusUpdate<TestTask> getTaskStatus() {
                return new StatusUpdate<TestTask>(0.5, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "test-task";
            }
        }

        TestTask testTask = new TestTask();
        try (StatusTracker<TestTask> statusTracker = StatusTracker.withInstrumented(testTask)) {
            sink.taskStarted(statusTracker);
            StatusUpdate<TestTask> status = testTask.getTaskStatus();
            sink.taskUpdate(statusTracker, status);
            sink.taskFinished(statusTracker);
        }

        scope.close();
    }

    /**
     * Tests LoggerTaskSink integration with Java logging framework.
     *
     * <p><strong>Purpose:</strong> Validates that LoggerTaskSink correctly integrates
     * with the Java logging framework, producing properly formatted log messages
     * for all task lifecycle events with appropriate log levels.
     *
     * <p><strong>Why Important:</strong> LoggerTaskSink enables integration with
     * existing application logging infrastructure. This test ensures proper
     * message formatting and logging framework integration.
     *
     * <p><strong>Coverage:</strong> Logger integration, message formatting,
     * log level handling, task lifecycle event logging, progress information inclusion.
     *
     * @see #testLoggerTaskSinkWithDifferentLoggers() for multiple logger scenarios
     * @see StatusTrackingIntegrationTest for logger sink in integration scenarios
     */
    @Test
    public void testLoggerTaskSink() {
        ByteArrayOutputStream logStream = new ByteArrayOutputStream();
        Logger testLogger = Logger.getLogger("test-logger");
        StreamHandler handler = new StreamHandler(logStream, new java.util.logging.SimpleFormatter());
        testLogger.addHandler(handler);
        testLogger.setLevel(Level.INFO);
        testLogger.setUseParentHandlers(false);

        LoggerStatusSink sink = new LoggerStatusSink(testLogger, Level.INFO);

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task = new TestableTask("test-task");

        // Create a simple tracker for testing
        class LogTestTask implements StatusUpdate.Provider<LogTestTask> {
            @Override
            public StatusUpdate<LogTestTask> getTaskStatus() {
                return new StatusUpdate<LogTestTask>(0.75, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "test-task";
            }
        }

        LogTestTask testTask = new LogTestTask();
        try (StatusTracker<LogTestTask> statusTracker = StatusTracker.withInstrumented(testTask)) {
            sink.taskStarted(statusTracker);
            handler.flush();
            String startLog = logStream.toString();
            assertTrue(startLog.contains("Task started: test-task"));

            logStream.reset();
            StatusUpdate<LogTestTask> status = testTask.getTaskStatus();
            sink.taskUpdate(statusTracker, status);
            handler.flush();
            String updateLog = logStream.toString();
            assertTrue(updateLog.contains("Task update: test-task"));
            assertTrue(updateLog.contains("75.0%"));

            logStream.reset();
            sink.taskFinished(statusTracker);
        }
        handler.flush();
        String finishLog = logStream.toString();
        assertTrue(finishLog.contains("Task finished: test-task"));

        scope.close();
    }

    /**
     * Tests LoggerTaskSink with different logger configurations and log levels.
     *
     * <p><strong>Purpose:</strong> Validates that LoggerTaskSink can work with
     * different logger instances and configurations, including custom loggers
     * and different log levels, demonstrating flexibility in logging setup.
     *
     * <p><strong>Why Important:</strong> Applications may have complex logging
     * configurations with different loggers for different components. This
     * ensures LoggerTaskSink works in varied logging environments.
     *
     * <p><strong>Coverage:</strong> Multiple logger instances, different log levels,
     * logger configuration flexibility, constructor variations.
     *
     * @see #testLoggerTaskSink() for basic logger integration
     * @see #testMultipleSinksConcurrently() for logger sink with other sink types
     */
    @Test
    public void testLoggerTaskSinkWithDifferentLoggers() {
        LoggerStatusSink sink1 = new LoggerStatusSink("logger1");
        LoggerStatusSink sink2 = new LoggerStatusSink("logger2", Level.WARNING);

        assertNotNull(sink1);
        assertNotNull(sink2);

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task = new TestableTask("test-task");

        // Create a simple tracker for testing
        class TestTask implements StatusUpdate.Provider<TestTask> {
            @Override
            public StatusUpdate<TestTask> getTaskStatus() {
                return new StatusUpdate<TestTask>(0.0, StatusUpdate.RunState.PENDING);
            }
            @Override
            public String toString() {
                return "test-task";
            }
        }

        TestTask testTask = new TestTask();
        try (StatusTracker<TestTask> statusTracker = StatusTracker.withInstrumented(testTask)) {
            sink1.taskStarted(statusTracker);
            sink2.taskStarted(statusTracker);
        }

        scope.close();
    }

    /**
     * Tests ConsoleTaskSink output formatting and progress bar functionality.
     *
     * <p><strong>Purpose:</strong> Validates that ConsoleTaskSink produces
     * correctly formatted console output with progress bars, task names,
     * progress percentages, and visual indicators for human-readable monitoring.
     *
     * <p><strong>Why Important:</strong> ConsoleTaskSink provides user-facing
     * progress visualization. This test ensures output is properly formatted,
     * visually appealing, and contains all necessary information.
     *
     * <p><strong>Coverage:</strong> Progress bar rendering, percentage display,
     * task name formatting, visual symbols (▶, ✓), progress bar characters (█, ░).
     *
     * @see #testConsoleTaskSinkWithTimestamp() for timestamp functionality
     * @see #testConsoleTaskSinkEdgeCases() for edge case handling
     * @see StatusTrackingIntegrationTest#testCompleteWorkflowWithInstrumentedTasks() for console output validation
     */
    @Test
    public void testConsoleTaskSink() {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream testOutput = new PrintStream(outputStream);

        ConsoleStatusSink sink = new ConsoleStatusSink(testOutput, false, true);

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task = new TestableTask("console-task");

        // Create a simple tracker for testing
        class ConsoleTestTask implements StatusUpdate.Provider<ConsoleTestTask> {
            private double progress = 0.0;
            private String message;

            void setProgress(double progress, String message) {
                this.progress = progress;
                this.message = message;
            }

            @Override
            public StatusUpdate<ConsoleTestTask> getTaskStatus() {
                return new StatusUpdate<ConsoleTestTask>(progress, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "console-task";
            }
        }

        ConsoleTestTask testTask = new ConsoleTestTask();
        try (StatusTracker<ConsoleTestTask> statusTracker = StatusTracker.withInstrumented(testTask)) {
            sink.taskStarted(statusTracker);
            String startOutput = outputStream.toString();
            assertTrue(startOutput.contains("▶ Started: console-task"));

            outputStream.reset();
            testTask.setProgress(0.6, "processing");
            StatusUpdate<ConsoleTestTask> status = testTask.getTaskStatus();
            sink.taskUpdate(statusTracker, status);
            String updateOutput = outputStream.toString();
            assertTrue(updateOutput.contains("console-task"));
            assertTrue(updateOutput.contains("60.0%"));
            assertTrue(updateOutput.contains("["));
            assertTrue(updateOutput.contains("█"));
            assertTrue(updateOutput.contains("░"));

            outputStream.reset();
            sink.taskFinished(statusTracker);
        }
        String finishOutput = outputStream.toString();
        assertTrue(finishOutput.contains("✓ Finished: console-task"));

        scope.close();
    }

    /**
     * Tests ConsoleTaskSink timestamp functionality and configuration options.
     *
     * <p><strong>Purpose:</strong> Validates that ConsoleTaskSink can include
     * timestamps in output when configured to do so, and that timestamp
     * and progress bar options work independently and correctly.
     *
     * <p><strong>Why Important:</strong> Timestamps are crucial for debugging
     * and performance analysis. This test ensures timestamp functionality
     * works correctly and independently of other formatting options.
     *
     * <p><strong>Coverage:</strong> Timestamp inclusion, configuration option
     * independence, timestamp formatting, selective feature enabling/disabling.
     *
     * @see #testConsoleTaskSink() for basic console output functionality
     * @see #testConsoleTaskSinkEdgeCases() for comprehensive edge case testing
     */
    @Test
    public void testConsoleTaskSinkWithTimestamp() {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream testOutput = new PrintStream(outputStream);

        ConsoleStatusSink sink = new ConsoleStatusSink(testOutput, true, false);

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task = new TestableTask("timestamp-task");

        // Create a simple tracker for testing
        class TimestampTestTask implements StatusUpdate.Provider<TimestampTestTask> {
            private double progress = 0.0;

            void setProgress(double progress) {
                this.progress = progress;
            }

            @Override
            public StatusUpdate<TimestampTestTask> getTaskStatus() {
                return new StatusUpdate<>(progress, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "timestamp-task";
            }
        }

        TimestampTestTask testTask = new TimestampTestTask();
        try (StatusTracker<TimestampTestTask> statusTracker = StatusTracker.withInstrumented(testTask)) {
            sink.taskStarted(statusTracker);
            String output = outputStream.toString();
            assertTrue(output.contains("["));
            assertTrue(output.contains("]"));
            assertTrue(output.contains("▶ Started: timestamp-task"));

            outputStream.reset();
            testTask.setProgress(0.4);
            StatusUpdate<TimestampTestTask> status = testTask.getTaskStatus();
            sink.taskUpdate(statusTracker, status);
            String updateOutput = outputStream.toString();
            assertTrue(updateOutput.contains("["));
            assertTrue(updateOutput.contains("40.0%"));
            assertFalse(updateOutput.contains("█"));
        }

        scope.close();
    }

    /**
     * Tests MetricsTaskSink comprehensive statistics collection and calculation.
     *
     * <p><strong>Purpose:</strong> Validates that MetricsTaskSink correctly tracks
     * all task statistics including counts, durations, progress averages,
     * and individual task metrics with proper lifecycle management.
     *
     * <p><strong>Why Important:</strong> MetricsTaskSink provides quantitative
     * insights into task performance. This test ensures accurate statistics
     * collection and calculation for performance monitoring and analysis.
     *
     * <p><strong>Coverage:</strong> Task counting, duration tracking, progress
     * averaging, individual task metrics, report generation, active task management.
     *
     * @see #testMetricsTaskSinkOperations() for metrics management operations
     * @see #testMetricsTaskSinkAverages() for average calculation testing
     * @see #testMetricsTaskSinkConcurrency() for concurrent metrics collection
     */
    @Test
    public void testMetricsTaskSink() throws InterruptedException {
        MetricsStatusSink sink = new MetricsStatusSink();

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task1 = new TestableTask("metrics-task-1");
        TestableTask task2 = new TestableTask("metrics-task-2");

        assertEquals(0, sink.getTotalTasksStarted());
        assertEquals(0, sink.getTotalTasksFinished());
        assertEquals(0, sink.getTotalUpdates());

        // Create simple trackers for testing
        class MetricsTestTask implements StatusUpdate.Provider<MetricsTestTask> {
            private final String name;
            private double progress = 0.0;
            private String message;

            MetricsTestTask(String name) {
                this.name = name;
            }

            void setProgress(double progress, String message) {
                this.progress = progress;
                this.message = message;
            }

            @Override
            public StatusUpdate<MetricsTestTask> getTaskStatus() {
                return new StatusUpdate<>(progress, StatusUpdate.RunState.RUNNING);
            }

            @Override
            public String toString() {
                return name;
            }
        }

        MetricsTestTask metricsTask1 = new MetricsTestTask("metrics-task-1");
        MetricsTestTask metricsTask2 = new MetricsTestTask("metrics-task-2");

        try (StatusTracker<MetricsTestTask> statusTracker1 = StatusTracker.withInstrumented(metricsTask1);
             StatusTracker<MetricsTestTask> statusTracker2 = StatusTracker.withInstrumented(metricsTask2)) {

            sink.taskStarted(statusTracker1);
            sink.taskStarted(statusTracker2);

            // Small delay to ensure duration > 0
            Thread.sleep(10);

            assertEquals(2, sink.getTotalTasksStarted());
            assertEquals(0, sink.getTotalTasksFinished());
            assertEquals(2, sink.getActiveTaskCount());

            metricsTask1.setProgress(0.3, "first update");
            sink.taskUpdate(statusTracker1, metricsTask1.getTaskStatus());

            metricsTask1.setProgress(0.7, "second update");
            sink.taskUpdate(statusTracker1, metricsTask1.getTaskStatus());

            metricsTask2.setProgress(0.5, null);
            sink.taskUpdate(statusTracker2, metricsTask2.getTaskStatus());

        assertEquals(3, sink.getTotalUpdates());

            MetricsStatusSink.TaskMetrics metrics1 = sink.getMetrics(statusTracker1);
            assertNotNull(metrics1);
            assertEquals("metrics-task-1", metrics1.getTaskName());
            assertEquals(2, metrics1.getUpdateCount());
            assertEquals(0.7, metrics1.getLastProgress(), 0.001);
            assertEquals(0.5, metrics1.getAverageProgress(), 0.001);
            assertFalse(metrics1.isFinished());
            assertTrue(metrics1.getDuration() > 0);

            Thread.sleep(10);

            sink.taskFinished(statusTracker1);

            assertEquals(2, sink.getTotalTasksStarted());
            assertEquals(1, sink.getTotalTasksFinished());
            assertEquals(1, sink.getActiveTaskCount());

            assertTrue(metrics1.isFinished());
            assertTrue(metrics1.getDuration() > 0);

            sink.taskFinished(statusTracker2);
            assertEquals(0, sink.getActiveTaskCount());
        }

        String report = sink.generateReport();
        assertNotNull(report);
        assertTrue(report.contains("Task Metrics Report"));
        assertTrue(report.contains("Total tasks started: 2"));
        assertTrue(report.contains("Total tasks finished: 2"));
        assertTrue(report.contains("metrics-task-1"));
        assertTrue(report.contains("metrics-task-2"));

        scope.close();
    }

    /**
     * Tests MetricsTaskSink management operations including metrics manipulation.
     *
     * <p><strong>Purpose:</strong> Validates MetricsTaskSink operations for
     * retrieving, removing, and clearing metrics data, ensuring proper
     * data management and cleanup capabilities.
     *
     * <p><strong>Why Important:</strong> Long-running applications need metrics
     * management to prevent memory leaks and enable selective data analysis.
     * This ensures proper metrics lifecycle management.
     *
     * <p><strong>Coverage:</strong> Metrics retrieval, individual metrics removal,
     * bulk metrics clearing, collection management, data lifecycle operations.
     *
     * @see #testMetricsTaskSink() for basic metrics collection
     * @see #testMetricsTaskSinkConcurrency() for concurrent operations
     */
    @Test
    public void testMetricsTaskSinkOperations() {
        MetricsStatusSink sink = new MetricsStatusSink();

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task = new TestableTask("operations-task");

        // Create simple tracker for testing
        class OperationsTestTask implements StatusUpdate.Provider<OperationsTestTask> {
            private double progress = 0.0;

            void setProgress(double progress) {
                this.progress = progress;
            }

            @Override
            public StatusUpdate<OperationsTestTask> getTaskStatus() {
                return new StatusUpdate<>(progress, StatusUpdate.RunState.RUNNING);
            }

            @Override
            public String toString() {
                return "operations-task";
            }
        }

        OperationsTestTask operationsTask = new OperationsTestTask();
        try (StatusTracker<OperationsTestTask> statusTracker = StatusTracker.withInstrumented(operationsTask)) {
            sink.taskStarted(statusTracker);
            operationsTask.setProgress(0.8);
            sink.taskUpdate(statusTracker, operationsTask.getTaskStatus());

            assertNotNull(sink.getAllMetrics());
            assertEquals(1, sink.getAllMetrics().size());

            sink.removeMetrics(statusTracker);
            assertNull(sink.getMetrics(statusTracker));

            sink.taskStarted(statusTracker);
            sink.taskUpdate(statusTracker, operationsTask.getTaskStatus());
        }
        assertEquals(1, sink.getAllMetrics().size());

        sink.clearMetrics();
        assertEquals(0, sink.getAllMetrics().size());

        scope.close();
    }

    /**
     * Tests MetricsTaskSink average duration calculation across multiple tasks.
     *
     * <p><strong>Purpose:</strong> Validates that MetricsTaskSink correctly
     * calculates average task durations across multiple completed tasks,
     * providing accurate performance benchmarks.
     *
     * <p><strong>Why Important:</strong> Average duration metrics are essential
     * for performance analysis and capacity planning. This ensures accurate
     * calculation of aggregate timing statistics.
     *
     * <p><strong>Coverage:</strong> Multi-task average calculation, duration
     * measurement accuracy, timing validation, statistical aggregation.
     *
     * @see #testMetricsTaskSink() for individual task metrics
     * @see #testMetricsTaskSinkConcurrency() for concurrent timing scenarios
     */
    @Test
    public void testMetricsTaskSinkAverages() throws InterruptedException {
        MetricsStatusSink sink = new MetricsStatusSink();

        TrackerScope scope = new TrackerScope("test-group");
        TestableTask task1 = new TestableTask("avg-task-1");
        TestableTask task2 = new TestableTask("avg-task-2");

        // Create simple trackers for testing
        class AvgTestTask implements StatusUpdate.Provider<AvgTestTask> {
            private final String name;

            AvgTestTask(String name) {
                this.name = name;
            }

            @Override
            public StatusUpdate<AvgTestTask> getTaskStatus() {
                return new StatusUpdate<>(1.0, StatusUpdate.RunState.SUCCESS);
            }

            @Override
            public String toString() {
                return name;
            }
        }

        AvgTestTask avgTask1 = new AvgTestTask("avg-task-1");
        AvgTestTask avgTask2 = new AvgTestTask("avg-task-2");

        try (StatusTracker<AvgTestTask> statusTracker1 = StatusTracker.withInstrumented(avgTask1);
             StatusTracker<AvgTestTask> statusTracker2 = StatusTracker.withInstrumented(avgTask2)) {

            sink.taskStarted(statusTracker1);
            sink.taskStarted(statusTracker2);

            Thread.sleep(50);

            sink.taskFinished(statusTracker1);

            Thread.sleep(30);

            sink.taskFinished(statusTracker2);
        }

        double avgDuration = sink.getAverageTaskDuration();
        assertTrue(avgDuration > 0);
        assertTrue(avgDuration < 200);

        scope.close();
    }

    /**
     * Tests coordination of multiple TaskSink implementations working together.
     *
     * <p><strong>Purpose:</strong> Validates that different TaskSink implementations
     * can be used simultaneously for the same task, each producing their
     * expected output without interference or conflicts.
     *
     * <p><strong>Why Important:</strong> Applications often need multiple forms
     * of progress reporting simultaneously (e.g., console output for users,
     * metrics for monitoring, logs for debugging).
     *
     * <p><strong>Coverage:</strong> Multi-sink coordination, output independence,
     * simultaneous sink operations, integration validation across sink types.
     *
     * @see StatusTrackingIntegrationTest#testMixedSinkBehavior() for comprehensive mixed sink scenarios
     * @see #testNoopTaskSink(), #testConsoleTaskSink(), #testMetricsTaskSink() for individual sink testing
     */
    @Test
    public void testMultipleSinksConcurrently() {
        StatusSink noopSink = NoopStatusSink.getInstance();
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ConsoleStatusSink consoleSink = new ConsoleStatusSink(new PrintStream(outputStream), false, false);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        TrackerScope scope = new TrackerScope("multi-sink-group");
        TestableTask task = new TestableTask("multi-sink-task");

        // Create simple tracker for testing
        class MultiSinkTestTask implements StatusUpdate.Provider<MultiSinkTestTask> {
            @Override
            public StatusUpdate<MultiSinkTestTask> getTaskStatus() {
                return new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "multi-sink-task";
            }
        }

        MultiSinkTestTask multiTask = new MultiSinkTestTask();
        try (StatusTracker<MultiSinkTestTask> statusTracker = StatusTracker.withInstrumented(multiTask)) {
            noopSink.taskStarted(statusTracker);
            consoleSink.taskStarted(statusTracker);
            metricsSink.taskStarted(statusTracker);

            StatusUpdate<MultiSinkTestTask> status = multiTask.getTaskStatus();

            noopSink.taskUpdate(statusTracker, status);
            consoleSink.taskUpdate(statusTracker, status);
            metricsSink.taskUpdate(statusTracker, status);

            noopSink.taskFinished(statusTracker);
            consoleSink.taskFinished(statusTracker);
            metricsSink.taskFinished(statusTracker);
        }

        String consoleOutput = outputStream.toString();
        assertTrue(consoleOutput.contains("multi-sink-task"));
        assertTrue(consoleOutput.contains("50.0%"));

        assertEquals(1, metricsSink.getTotalTasksStarted());
        assertEquals(1, metricsSink.getTotalTasksFinished());
        assertEquals(1, metricsSink.getTotalUpdates());

        scope.close();
    }

    /**
     * Tests TaskSink compatibility with custom task objects and tracker implementations.
     *
     * <p><strong>Purpose:</strong> Validates that TaskSink implementations can
     * work with custom task objects and tracker implementations beyond the
     * standard TestableTask, demonstrating flexibility in task object handling.
     *
     * <p><strong>Why Important:</strong> Applications may need to track custom
     * objects or use specialized tracker implementations. This ensures TaskSink
     * implementations are flexible enough to handle varied task types.
     *
     * <p><strong>Coverage:</strong> Custom task object handling, non-TestableTask
     * compatibility, flexible type support, custom tracker integration.
     *
     * @see TrackerTest#testTrackerWithCustomObject() for custom object tracking patterns
     * @see #testMultipleSinksConcurrently() for sink integration scenarios
     */
    @Test
    public void testTaskSinkWithNonTestableTask() {
        // Since TestableTask is now used, create a regular instance
        TestableTask customTask = new TestableTask("CustomTask");

        // TaskUpdate removed - TaskStatus now handles all status information

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ConsoleStatusSink consoleSink = new ConsoleStatusSink(new PrintStream(outputStream), false, false);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        // Create simple tracker for testing
        class CustomTestTask implements StatusUpdate.Provider<CustomTestTask> {
            @Override
            public StatusUpdate<CustomTestTask> getTaskStatus() {
                return new StatusUpdate<>(0.3, StatusUpdate.RunState.RUNNING);
            }
            @Override
            public String toString() {
                return "CustomTask";
            }
        }

        CustomTestTask customTestTask = new CustomTestTask();
        try (StatusTracker<CustomTestTask> statusTracker = StatusTracker.withInstrumented(customTestTask)) {
            consoleSink.taskStarted(statusTracker);
            StatusUpdate<CustomTestTask> status = customTestTask.getTaskStatus();
            consoleSink.taskUpdate(statusTracker, status);
            consoleSink.taskFinished(statusTracker);

            metricsSink.taskStarted(statusTracker);
            metricsSink.taskUpdate(statusTracker, status);
            metricsSink.taskFinished(statusTracker);

            String output = outputStream.toString();
            assertTrue(output.contains("CustomTask"));

            MetricsStatusSink.TaskMetrics metrics = metricsSink.getMetrics(statusTracker);
            assertNotNull(metrics);
            assertEquals("CustomTask", metrics.getTaskName());
        }
    }

    /**
     * Tests ConsoleTaskSink behavior under edge case conditions and boundary values.
     *
     * <p><strong>Purpose:</strong> Validates that ConsoleTaskSink handles edge
     * cases gracefully including 0% and 100% progress, null messages, and
     * various progress values without formatting errors or exceptions.
     *
     * <p><strong>Why Important:</strong> Edge cases often reveal formatting
     * bugs or error conditions. This ensures ConsoleTaskSink is robust
     * and handles all possible input scenarios gracefully.
     *
     * <p><strong>Coverage:</strong> Boundary progress values (0%, 100%),
     * null message handling, edge case formatting, error resilience.
     *
     * @see #testConsoleTaskSink() for normal console output behavior
     * @see #testConsoleTaskSinkWithTimestamp() for timestamp edge cases
     */
    @Test
    public void testConsoleTaskSinkEdgeCases() {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream testOutput = new PrintStream(outputStream);

        ConsoleStatusSink sink = new ConsoleStatusSink(testOutput, false, true);

        TrackerScope scope = new TrackerScope("edge-case-group");
        TestableTask task = new TestableTask("edge-task");

        try {
            // Create a mock Tracker for the sink API
            class MockTracker implements StatusUpdate.Provider<MockTracker> {
                private final String name;
                private double progress;
                private String message;

                MockTracker(String name) {
                    this.name = name;
                }

                void setProgress(double progress, String message) {
                    this.progress = progress;
                    this.message = message;
                }

                @Override
                public StatusUpdate<MockTracker> getTaskStatus() {
                    return new StatusUpdate<>(progress,
                        progress >= 1.0 ? StatusUpdate.RunState.SUCCESS :
                        progress > 0 ? StatusUpdate.RunState.RUNNING : StatusUpdate.RunState.PENDING);
                }

                @Override
                public String toString() {
                    return name;
                }
            }

            MockTracker mockTask = new MockTracker("edge-task");
            StatusTracker<MockTracker> statusTracker = StatusTracker.withInstrumented(mockTask);

            try {
                // Test with 0% progress
                mockTask.setProgress(0.0, "Starting");
                StatusUpdate<MockTracker> status = mockTask.getTaskStatus();
                sink.taskUpdate(statusTracker, status);
                String zeroOutput = outputStream.toString();
                assertTrue(zeroOutput.contains("0.0%"));

                outputStream.reset();

                // Test with 100% progress
                mockTask.setProgress(1.0, "Complete");
                status = mockTask.getTaskStatus();
                sink.taskUpdate(statusTracker, status);
                String fullOutput = outputStream.toString();
                assertTrue(fullOutput.contains("100.0%"));

                outputStream.reset();

                // Test with null message
                mockTask.setProgress(0.5, null);
                status = mockTask.getTaskStatus();
                sink.taskUpdate(statusTracker, status);
                String nullMsgOutput = outputStream.toString();
                assertTrue(nullMsgOutput.contains("50.0%"));
                // Should handle null message gracefully
            } finally {
                statusTracker.close();
            }
        } finally {
            scope.close();
        }
    }

    /**
     * Tests MetricsTaskSink thread safety and concurrent access patterns.
     *
     * <p><strong>Purpose:</strong> Validates that MetricsTaskSink can handle
     * concurrent operations from multiple threads safely, including concurrent
     * task starts, updates, and completions without data corruption.
     *
     * <p><strong>Why Important:</strong> Multi-threaded applications may have
     * multiple threads reporting task progress simultaneously. MetricsTaskSink
     * must be thread-safe to provide accurate statistics.
     *
     * <p><strong>Coverage:</strong> Concurrent task operations, thread safety,
     * data integrity under concurrency, accurate counting with parallel access.
     *
     * @see #testMetricsTaskSink() for basic metrics functionality
     * @see #testMultipleSinksConcurrently() for multi-sink scenarios
     * @see TaskMonitorTest#testConcurrentStartStop() for monitor-level concurrency
     */
    @Test
    public void testMetricsTaskSinkConcurrency() throws InterruptedException {
        MetricsStatusSink sink = new MetricsStatusSink();

        class ConcurrentTask implements StatusUpdate.Provider<ConcurrentTask> {
            private final String name;
            private double progress;

            ConcurrentTask(String name) {
                this.name = name;
            }

            void setProgress(double progress) {
                this.progress = progress;
            }

            @Override
            public StatusUpdate<ConcurrentTask> getTaskStatus() {
                return new StatusUpdate<>(progress,
                    progress >= 1.0 ? StatusUpdate.RunState.SUCCESS : StatusUpdate.RunState.RUNNING);
            }

            @Override
            public String toString() {
                return name;
            }
        }

        List<StatusTracker<ConcurrentTask>> statusTrackers = new ArrayList<>();

        try {
            // Create trackers
            for (int i = 0; i < 10; i++) {
                ConcurrentTask task = new ConcurrentTask("concurrent-task-" + i);
                StatusTracker<ConcurrentTask> statusTracker = StatusTracker.withInstrumented(task);
                statusTrackers.add(statusTracker);
            }

            // Start all tasks concurrently
            for (StatusTracker<ConcurrentTask> statusTracker : statusTrackers) {
                sink.taskStarted(statusTracker);
            }

            assertEquals(10, sink.getTotalTasksStarted());
            assertEquals(10, sink.getActiveTaskCount());

            // Update all tasks concurrently
            for (int i = 0; i < statusTrackers.size(); i++) {
                StatusTracker<ConcurrentTask> statusTracker = statusTrackers.get(i);
                ConcurrentTask task = statusTracker.getTracked();
                task.setProgress((i + 1) / 10.0);
                StatusUpdate<ConcurrentTask> status = task.getTaskStatus();
                sink.taskUpdate(statusTracker, status);
            }

            assertEquals(10, sink.getTotalUpdates());

            // Finish all tasks
            for (StatusTracker<ConcurrentTask> statusTracker : statusTrackers) {
                sink.taskFinished(statusTracker);
            }

            assertEquals(10, sink.getTotalTasksFinished());
            assertEquals(0, sink.getActiveTaskCount());
        } finally {
            for (StatusTracker<ConcurrentTask> statusTracker : statusTrackers) {
                statusTracker.close();
            }
        }
    }

    /**
     * Tests TaskSink error handling and recovery patterns with unreliable sink implementations.
     *
     * <p><strong>Purpose:</strong> Validates error handling patterns for TaskSink
     * implementations that may experience intermittent failures, demonstrating
     * how applications can handle and recover from sink errors.
     *
     * <p><strong>Why Important:</strong> Real-world sink implementations (e.g.,
     * network-based sinks, external systems) may experience transient failures.
     * This demonstrates proper error handling patterns.
     *
     * <p><strong>Coverage:</strong> Intermittent failure simulation, error recovery
     * patterns, continued operation after failures, failure isolation.
     *
     * @see TaskMonitorTest#testExceptionHandling() for monitor-level error handling
     * @see TaskMonitorTest#testStatusChangeCallbackException() for callback error scenarios
     */
    @Test
    public void testSinkErrorRecovery() {
        class UnreliableSink implements StatusSink {
            private int callCount = 0;

            @Override
            public void taskStarted(StatusTracker<?> task) {
                callCount++;
            }

            @Override
            public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
                callCount++;
                if (callCount % 2 == 0) {
                    throw new RuntimeException("Simulated intermittent failure");
                }
                // Succeeds on odd calls
            }

            @Override
            public void taskFinished(StatusTracker<?> task) {
                callCount++;
            }

            public int getCallCount() {
                return callCount;
            }
        }

        class ErrorTask implements StatusUpdate.Provider<ErrorTask> {
            private double progress;

            ErrorTask() {}

            void setProgress(double progress) {
                this.progress = progress;
            }

            @Override
            public StatusUpdate<ErrorTask> getTaskStatus() {
                return new StatusUpdate<>(progress, StatusUpdate.RunState.RUNNING);
            }

            @Override
            public String toString() {
                return "error-recovery-task";
            }
        }

        UnreliableSink unreliableSink = new UnreliableSink();
        ErrorTask task = new ErrorTask();

        try (StatusTracker<ErrorTask> statusTracker = StatusTracker.withInstrumented(task)) {
            // Start should work (first call)
            unreliableSink.taskStarted(statusTracker);
            assertEquals(1, unreliableSink.getCallCount());

            // First update should fail (second call)
            try {
                task.setProgress(0.3);
                StatusUpdate<ErrorTask> status = task.getTaskStatus();
                unreliableSink.taskUpdate(statusTracker, status);
                fail("Expected RuntimeException");
            } catch (RuntimeException e) {
                assertEquals("Simulated intermittent failure", e.getMessage());
            }
            assertEquals(2, unreliableSink.getCallCount());

            // Second update should succeed (third call)
            task.setProgress(0.6);
            StatusUpdate<ErrorTask> status = task.getTaskStatus();
            unreliableSink.taskUpdate(statusTracker, status);
            assertEquals(3, unreliableSink.getCallCount());
        }
    }
}