package io.github.jbellis.jvector.status;

import com.carrotsearch.randomizedtesting.RandomizedTest;

import io.github.jbellis.jvector.status.sinks.ConsoleStatusSink;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.sinks.NoopStatusSink;
import io.github.jbellis.jvector.status.sinks.StatusSinkTest;
import org.junit.Test;
import org.junit.jupiter.api.Tag;


import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Function;

import static org.junit.Assert.*;

/**
 * End-to-end integration tests for the complete status tracking framework.
 *
 * <p>This test suite validates the entire status tracking system working together,
 * including realistic usage patterns, complex scenarios, and production-like conditions:
 * <ul>
 *   <li><strong>Complete Workflows:</strong> Full task lifecycle from creation to completion</li>
 *   <li><strong>Multi-Component Integration:</strong> Trackers, monitors, sinks, and scopes working together</li>
 *   <li><strong>Real-World Scenarios:</strong> Concurrent tasks, failure handling, resource management</li>
 *   <li><strong>Performance Validation:</strong> System behavior under stress and high load</li>
 * </ul>
 *
 * <h2>Integration Test Scenarios:</h2>
 * <ul>
 *   <li>End-to-end task execution with multiple sinks</li>
 *   <li>Concurrent task execution and coordination</li>
 *   <li>Failure propagation and error recovery</li>
 *   <li>Resource cleanup and lifecycle management</li>
 *   <li>Mixed tracking patterns (instrumented vs functor-based)</li>
 *   <li>Complex sink configurations and interactions</li>
 * </ul>
 *
 * <h2>Relationship to Unit Tests:</h2>
 * <p>While unit test classes ({@link StatusTrackerTest}, {@link TaskMonitorTest}, etc.) test
 * individual components in isolation, this integration test suite validates:
 * <ul>
 *   <li>Component interactions and data flow</li>
 *   <li>System-wide behavior and performance</li>
 *   <li>Real-world usage patterns and edge cases</li>
 *   <li>Cross-component error handling and recovery</li>
 * </ul>
 *
 * @see StatusTrackerTest for core Tracker functionality
 * @see TaskMonitorTest for monitoring behavior
 * @see StatusSinkTest for sink implementations
 * @see StatusTrackerScopeTest for scope management
 * @since 4.0.0
 */
@Tag("Integration")
public class StatusTrackingIntegrationTest extends RandomizedTest {

    static class SimulatedWorkTask implements StatusUpdate.Provider<SimulatedWorkTask> {
        private final String name;
        private final long totalWork;
        private volatile long completedWork = 0;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile boolean failed = false;

        public SimulatedWorkTask(String name, long totalWork) {
            this.name = name;
            this.totalWork = totalWork;
        }

        @Override
        public StatusUpdate<SimulatedWorkTask> getTaskStatus() {
            double progress = totalWork > 0 ? (double) completedWork / totalWork : 0.0;
            return new StatusUpdate<>(progress, state, this);
        }

        public void doWork(long amount) {
            if (state == StatusUpdate.RunState.PENDING) {
                state = StatusUpdate.RunState.RUNNING;
            }
            completedWork = Math.min(totalWork, completedWork + amount);
            if (completedWork >= totalWork && !failed) {
                state = StatusUpdate.RunState.SUCCESS;
            }
        }

        public void fail() {
            failed = true;
            state = StatusUpdate.RunState.FAILED;
        }

        public String getName() {
            return name;
        }

        @Override
        public String toString() {
            return name;
        }

        public boolean isComplete() {
            return state == StatusUpdate.RunState.SUCCESS || state == StatusUpdate.RunState.FAILED;
        }
    }

    /**
     * Tests complete end-to-end workflow with instrumented tasks and multiple sinks.
     *
     * <p><strong>Purpose:</strong> Validates the entire status tracking system working
     * together in a realistic scenario, including Tracker, TaskMonitor, ConsoleTaskSink,
     * MetricsTaskSink, and SimulatedWorkTask with actual task execution and progress updates.
     *
     * <p><strong>Why Important:</strong> This is the primary integration test that
     * demonstrates the complete system functioning as designed in a production-like
     * scenario with real timing, threading, and progress updates.
     *
     * <p><strong>Coverage:</strong> Full system integration, instrumented task patterns,
     * multiple sink coordination, console output validation, metrics collection,
     * concurrent execution with ExecutorService.
     *
     * @see #testMultipleTasksWithDifferentSinks() for multi-task scenarios
     * @see #testFunctorBasedTracking() for alternative tracking patterns
     * @see StatusTrackerTest#testWithInstrumented() for basic instrumented task testing
     */
    @Test
    public void testCompleteWorkflowWithInstrumentedTasks() throws InterruptedException, ExecutionException, TimeoutException {
        ByteArrayOutputStream consoleOutput = new ByteArrayOutputStream();
        ConsoleStatusSink consoleSink = new ConsoleStatusSink(new PrintStream(consoleOutput), false, true);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        SimulatedWorkTask workTask = new SimulatedWorkTask("data-processing", 1000);

        try (StatusTracker<SimulatedWorkTask> statusTracker = StatusTracker.withInstrumented(workTask)) {
            consoleSink.taskStarted(statusTracker);
            metricsSink.taskStarted(statusTracker);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            try {
                Future<?> workFuture = executor.submit(() -> {
                try {
                    for (int i = 0; i < 10; i++) {
                        workTask.doWork(100);

                        StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();
                        consoleSink.taskUpdate(statusTracker, status);
                        metricsSink.taskUpdate(statusTracker, status);

                        Thread.sleep(20);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });

                workFuture.get(5, TimeUnit.SECONDS);

                Thread.sleep(200);

                StatusUpdate<SimulatedWorkTask> finalStatus = statusTracker.getStatus();
                assertEquals(1.0, finalStatus.progress, 0.001);
                assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);

                // Send final status update to ensure 100% is captured
                consoleSink.taskUpdate(statusTracker, finalStatus);
                metricsSink.taskUpdate(statusTracker, finalStatus);

                consoleSink.taskFinished(statusTracker);
                metricsSink.taskFinished(statusTracker);

                String output = consoleOutput.toString();
                assertTrue(output.contains("▶ Started: data-processing"));
                assertTrue(output.contains("✓ Finished: data-processing"));
                assertTrue(output.contains("100.0%"));

                assertEquals(1, metricsSink.getTotalTasksStarted());
                assertEquals(1, metricsSink.getTotalTasksFinished());
                assertTrue(metricsSink.getTotalUpdates() > 0);
            } finally {
                executor.shutdown();
                if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            }
        }
    }

    /**
     * Tests concurrent execution of multiple tasks with metrics collection.
     *
     * <p><strong>Purpose:</strong> Validates that the status tracking system can
     * handle multiple concurrent tasks with different execution patterns,
     * each tracked independently with accurate metrics collection.
     *
     * <p><strong>Why Important:</strong> Real applications often run multiple
     * tasks concurrently. This test ensures the system scales properly and
     * maintains accurate tracking across concurrent task execution.
     *
     * <p><strong>Coverage:</strong> Multi-task concurrency, independent task tracking,
     * metrics aggregation across tasks, concurrent progress updates, thread pool execution.
     *
     * @see #testCompleteWorkflowWithInstrumentedTasks() for single-task integration
     * @see #testPassiveMonitoringIntegration() for passive monitoring patterns
     * @see TaskMonitorTest#testConcurrentStartStop() for monitor-level concurrency
     */
    @Test
    public void testMultipleTasksWithDifferentSinks() throws InterruptedException, ExecutionException, TimeoutException {
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        List<SimulatedWorkTask> tasks = new ArrayList<>();
        List<StatusTracker<SimulatedWorkTask>> statusTrackers = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            SimulatedWorkTask task = new SimulatedWorkTask("task-" + i, 500 + i * 100);
            tasks.add(task);
            statusTrackers.add(StatusTracker.withInstrumented(task));
        }

        ExecutorService executor = Executors.newFixedThreadPool(3);
        try {
            List<Future<?>> futures = new ArrayList<>();

            for (int i = 0; i < tasks.size(); i++) {
            final int taskIndex = i;
            metricsSink.taskStarted(statusTrackers.get(i));

            Future<?> future = executor.submit(() -> {
                SimulatedWorkTask task = tasks.get(taskIndex);
                StatusTracker<SimulatedWorkTask> statusTracker = statusTrackers.get(taskIndex);

                try {
                    for (int step = 0; step < 10; step++) {
                        task.doWork(task.totalWork / 10);

                        StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();
                        metricsSink.taskUpdate(statusTracker, status);

                        Thread.sleep(10 + taskIndex * 5);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            futures.add(future);
        }

        for (Future<?> future : futures) {
            future.get(10, TimeUnit.SECONDS);
        }

        Thread.sleep(300);

        for (int i = 0; i < tasks.size(); i++) {
            StatusUpdate<SimulatedWorkTask> status = statusTrackers.get(i).getStatus();
            assertEquals(1.0, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, status.runstate);

            metricsSink.taskFinished(statusTrackers.get(i));
        }

        assertEquals(5, metricsSink.getTotalTasksStarted());
        assertEquals(5, metricsSink.getTotalTasksFinished());
        assertEquals(0, metricsSink.getActiveTaskCount());

        String report = metricsSink.generateReport();
        assertTrue(report.contains("Total tasks started: 5"));
        assertTrue(report.contains("Total tasks finished: 5"));

            for (StatusTracker<SimulatedWorkTask> statusTracker : statusTrackers) {
                statusTracker.close();
            }
        } finally {
            executor.shutdown();
            if (!executor.awaitTermination(2, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        }
    }

    /**
     * Tests comprehensive failure handling across the entire status tracking system.
     *
     * <p><strong>Purpose:</strong> Validates that the system correctly handles task
     * failures, propagating failure states through all components while maintaining
     * system stability and accurate failure reporting.
     *
     * <p><strong>Why Important:</strong> Task failures are common in real applications.
     * The system must handle failures gracefully, report them accurately, and
     * maintain stability without compromising other tasks.
     *
     * <p><strong>Coverage:</strong> Task failure propagation, failure state detection,
     * sink notification of failures, metrics tracking of failed tasks,
     * console output for failures.
     *
     * @see #testCompleteWorkflowWithInstrumentedTasks() for success scenarios
     * @see StatusTrackerTest#testTrackerStatusAfterTaskFailure() for basic failure handling
     * @see TaskMonitorTest#testAutoStopOnFailure() for monitor failure behavior
     */
    @Test
    public void testFailureHandling() throws InterruptedException {
        ByteArrayOutputStream consoleOutput = new ByteArrayOutputStream();
        ConsoleStatusSink consoleSink = new ConsoleStatusSink(new PrintStream(consoleOutput), false, false);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        SimulatedWorkTask failingTask = new SimulatedWorkTask("failing-task", 1000);

        try (StatusTracker<SimulatedWorkTask> statusTracker = StatusTracker.withInstrumented(failingTask)) {
            consoleSink.taskStarted(statusTracker);
            metricsSink.taskStarted(statusTracker);

            failingTask.doWork(300);
            StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();
            consoleSink.taskUpdate(statusTracker, status);
            metricsSink.taskUpdate(statusTracker, status);

            Thread.sleep(100);

            failingTask.fail();

            Thread.sleep(200);

            StatusUpdate<SimulatedWorkTask> finalStatus = statusTracker.getStatus();
            assertEquals(StatusUpdate.RunState.FAILED, finalStatus.runstate);

            consoleSink.taskFinished(statusTracker);
            metricsSink.taskFinished(statusTracker);

            String output = consoleOutput.toString();
            assertTrue(output.contains("failing-task"));

            MetricsStatusSink.TaskMetrics metrics = metricsSink.getMetrics(statusTracker);
            assertNotNull(metrics);
            assertTrue(metrics.isFinished());
            assertTrue(metrics.getDuration() > 0);
        }
    }

    /**
     * Tests integration of functor-based tracking patterns with metrics collection.
     *
     * <p><strong>Purpose:</strong> Validates that the status tracking system works
     * correctly with functor-based tracking (custom status functions) as an
     * alternative to instrumented tasks, demonstrating pattern flexibility.
     *
     * <p><strong>Why Important:</strong> Applications may need to track objects
     * that cannot be modified to implement TaskStatus.Provider. Functor-based
     * tracking provides this flexibility while maintaining full system integration.
     *
     * <p><strong>Coverage:</strong> Functor-based tracking integration, custom status
     * functions, alternative tracking patterns, metrics collection with functors.
     *
     * @see #testCompleteWorkflowWithInstrumentedTasks() for instrumented patterns
     * @see StatusTrackerTest#testWithFunctors() for basic functor usage
     * @see StatusTrackerTest#testCustomStatusFunction() for advanced functor scenarios
     */
    @Test
    public void testFunctorBasedTracking() throws InterruptedException {
        class CustomTask {
            private final String name;
            private volatile double progress = 0.0;
            private volatile boolean completed = false;

            CustomTask(String name) {
                this.name = name;
            }

            void updateProgress(double progress) {
                this.progress = progress;
                if (progress >= 1.0) {
                    completed = true;
                }
            }

            String getName() {
                return name;
            }

            double getProgress() {
                return progress;
            }

            boolean isCompleted() {
                return completed;
            }
        }

        Function<CustomTask, StatusUpdate<CustomTask>> statusFunction = task -> {
            StatusUpdate.RunState state = task.isCompleted() ? StatusUpdate.RunState.SUCCESS :
                                       task.getProgress() > 0 ? StatusUpdate.RunState.RUNNING :
                                       StatusUpdate.RunState.PENDING;
            return new StatusUpdate<>(task.getProgress(), state);
        };

        CustomTask customTask = new CustomTask("custom-work");
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try (StatusTracker<CustomTask> statusTracker = StatusTracker.withFunctors(customTask, statusFunction)) {
            metricsSink.taskStarted(statusTracker);

            for (int i = 1; i <= 10; i++) {
                customTask.updateProgress(i / 10.0);

                Thread.sleep(50);

                StatusUpdate<CustomTask> status = statusTracker.getStatus();
                metricsSink.taskUpdate(statusTracker, status);
            }

            Thread.sleep(200);

            StatusUpdate<CustomTask> finalStatus = statusTracker.getStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);

            metricsSink.taskFinished(statusTracker);

            assertEquals(1, metricsSink.getTotalTasksStarted());
            assertEquals(1, metricsSink.getTotalTasksFinished());
        }
    }

    /**
     * Tests passive monitoring integration where tasks update independently of monitoring.
     *
     * <p><strong>Purpose:</strong> Validates passive monitoring patterns where
     * background tasks update their own progress while monitors passively observe
     * and report changes, demonstrating asynchronous monitoring capabilities.
     *
     * <p><strong>Why Important:</strong> Many real-world scenarios involve background
     * tasks that update independently. The system must support passive monitoring
     * without interfering with task execution.
     *
     * <p><strong>Coverage:</strong> Passive monitoring patterns, asynchronous task
     * execution, independent progress updates, CompletableFuture integration.
     *
     * @see #testCompleteWorkflowWithInstrumentedTasks() for active monitoring scenarios
     * @see StatusTrackerTest#testPassiveMonitoringWithInstrumented() for basic passive patterns
     * @see TaskMonitorTest#testBasicMonitoring() for monitoring fundamentals
     */
    @Test
    public void testPassiveMonitoringIntegration() throws InterruptedException {
        SimulatedWorkTask backgroundTask = new SimulatedWorkTask("background-task", 500);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try (StatusTracker<SimulatedWorkTask> statusTracker = StatusTracker.withInstrumented(backgroundTask)) {
            metricsSink.taskStarted(statusTracker);

            CompletableFuture.runAsync(() -> {
                try {
                    for (int i = 0; i < 5; i++) {
                        backgroundTask.doWork(100);
                        Thread.sleep(100);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });

            Thread.sleep(50);

            for (int i = 0; i < 8; i++) {
                StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();
                metricsSink.taskUpdate(statusTracker, status);
                Thread.sleep(80);
            }

            Thread.sleep(200);

            StatusUpdate<SimulatedWorkTask> finalStatus = statusTracker.getStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);

            metricsSink.taskFinished(statusTracker);

            assertTrue(metricsSink.getTotalUpdates() > 0);
            MetricsStatusSink.TaskMetrics metrics = metricsSink.getMetrics(statusTracker);
            assertNotNull(metrics);
            assertTrue(metrics.isFinished());
        }
    }

    /**
     * Tests comprehensive resource management across multiple task lifecycles.
     *
     * <p><strong>Purpose:</strong> Validates that the system properly manages
     * resources during multiple task creation, execution, and cleanup cycles,
     * ensuring no resource leaks or memory issues.
     *
     * <p><strong>Why Important:</strong> Long-running applications create and
     * destroy many tasks over time. The system must manage resources efficiently
     * to prevent memory leaks and maintain performance.
     *
     * <p><strong>Coverage:</strong> Resource lifecycle management, multiple task
     * creation/cleanup cycles, memory efficiency, proper resource disposal.
     *
     * @see #testMultipleTasksWithDifferentSinks() for concurrent resource management
     * @see StatusTrackerTest#testTrackerLifecycle() for basic resource management
     * @see TaskTrackerTest#testTaskTrackerMemoryCleanup() for memory management testing
     */
    @Test
    public void testResourceManagementIntegration() throws InterruptedException {
        List<StatusTracker<SimulatedWorkTask>> statusTrackers = new ArrayList<>();
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try {
            for (int i = 0; i < 3; i++) {
                SimulatedWorkTask task = new SimulatedWorkTask("resource-task-" + i, 200);
                StatusTracker<SimulatedWorkTask> statusTracker = StatusTracker.withInstrumented(task);
                statusTrackers.add(statusTracker);

                metricsSink.taskStarted(statusTracker);

                task.doWork(200);
                Thread.sleep(50);

                StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();
                metricsSink.taskUpdate(statusTracker, status);

                metricsSink.taskFinished(statusTracker);
            }

            Thread.sleep(200);

            for (StatusTracker<SimulatedWorkTask> statusTracker : statusTrackers) {
                StatusUpdate<SimulatedWorkTask> finalStatus = statusTracker.getStatus();
                assertEquals(1.0, finalStatus.progress, 0.001);
                assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);
            }

            assertEquals(3, metricsSink.getTotalTasksStarted());
            assertEquals(3, metricsSink.getTotalTasksFinished());

        } finally {
            for (StatusTracker<SimulatedWorkTask> statusTracker : statusTrackers) {
                statusTracker.close();
            }
        }
    }

    /**
     * Tests mixed sink behavior with different sink types working together.
     *
     * <p><strong>Purpose:</strong> Validates that different types of sinks
     * (NoopTaskSink, ConsoleTaskSink, MetricsTaskSink) can work together
     * simultaneously for the same task, each producing expected output independently.
     *
     * <p><strong>Why Important:</strong> Production applications typically use
     * multiple sink types simultaneously (console for users, metrics for monitoring,
     * noop for performance-critical paths). This ensures proper coordination.
     *
     * <p><strong>Coverage:</strong> Multi-sink coordination, sink independence,
     * output validation across different sink types, synchronized sink operations.
     *
     * @see StatusSinkTest#testMultipleSinksConcurrently() for detailed sink coordination testing
     * @see #testCompleteWorkflowWithInstrumentedTasks() for console/metrics integration
     * @see StatusSinkTest for individual sink testing
     */
    @Test
    public void testMixedSinkBehavior() throws InterruptedException {
        SimulatedWorkTask task = new SimulatedWorkTask("mixed-sink-task", 300);

        ByteArrayOutputStream consoleOutput = new ByteArrayOutputStream();
        ConsoleStatusSink consoleSink = new ConsoleStatusSink(new PrintStream(consoleOutput), true, true);
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try (StatusTracker<SimulatedWorkTask> statusTracker = StatusTracker.withInstrumented(task)) {
            NoopStatusSink.getInstance().taskStarted(statusTracker);
            consoleSink.taskStarted(statusTracker);
            metricsSink.taskStarted(statusTracker);

            for (int i = 1; i <= 3; i++) {
                task.doWork(100);
                Thread.sleep(100);

                StatusUpdate<SimulatedWorkTask> status = statusTracker.getStatus();

                NoopStatusSink.getInstance().taskUpdate(statusTracker, status);
                consoleSink.taskUpdate(statusTracker, status);
                metricsSink.taskUpdate(statusTracker, status);
            }

            Thread.sleep(200);

            NoopStatusSink.getInstance().taskFinished(statusTracker);
            consoleSink.taskFinished(statusTracker);
            metricsSink.taskFinished(statusTracker);

            String output = consoleOutput.toString();
            assertTrue(output.contains("mixed-sink-task"));
            assertTrue(output.contains("100.0%"));

            assertEquals(1, metricsSink.getTotalTasksStarted());
            assertEquals(1, metricsSink.getTotalTasksFinished());
            assertEquals(3, metricsSink.getTotalUpdates());

            StatusUpdate<SimulatedWorkTask> finalStatus = statusTracker.getStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);
        }
    }
}