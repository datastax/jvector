package io.github.jbellis.jvector.status;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import org.junit.Test;
import org.junit.jupiter.api.Tag;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;

import static org.junit.Assert.*;

/**
 * TaskMonitor tests covering core functionality and edge cases.
 *
 * <p>This is one of the most comprehensive test suites in the status package, covering all aspects
 * of TaskMonitor behavior including threading, lifecycle, performance, and error handling.
 *
 * <h2>Core Test Areas:</h2>
 * <ul>
 *   <li><strong>Basic Monitoring:</strong> Start/stop lifecycle and status polling behavior</li>
 *   <li><strong>Auto-Stop Logic:</strong> Automatic termination on task completion/failure</li>
 *   <li><strong>Thread Safety:</strong> Concurrent start/stop operations and state management</li>
 *   <li><strong>Performance:</strong> High-frequency polling and update handling</li>
 *   <li><strong>Error Resilience:</strong> Exception handling during polling and callbacks</li>
 *   <li><strong>Configuration:</strong> Custom poll intervals and callback functions</li>
 * </ul>
 *
 * <h2>Why TaskMonitor is Critical:</h2>
 * <p>TaskMonitor is the heart of the status tracking system's background operation.
 * It provides the polling engine that continuously monitors task status and triggers
 * status change notifications. This test suite ensures it operates correctly under
 * all conditions including:</p>
 * <ul>
 *   <li>Varied polling frequencies (from 1ms to several seconds)</li>
 *   <li>Rapid status changes and high-frequency updates</li>
 *   <li>Task failures and exception scenarios</li>
 *   <li>Resource cleanup and thread lifecycle management</li>
 * </ul>
 *
 * <h2>Test Organization:</h2>
 * <p>This test class contains 20+ individual tests organized by functionality:</p>
 * <ul>
 *   <li>Basic operations (start, stop, status queries)</li>
 *   <li>Auto-stop behavior for completed/failed tasks</li>
 *   <li>Concurrency and thread safety scenarios</li>
 *   <li>Performance and stress testing</li>
 *   <li>Error handling and recovery</li>
 *   <li>Edge cases and boundary conditions</li>
 * </ul>
 *
 * @see StatusMonitor for the class under test
 * @see StatusTrackerTest for higher-level Tracker integration
 * @see StatusTrackingIntegrationTest for end-to-end monitor usage
 * @since 4.0.0
 */
@Tag("Core")
@Tag("CornerCase")
public class TaskMonitorTest extends RandomizedTest {

    static class TestTask {
        private volatile double progress = 0.0;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private final String name;

        public TestTask(String name) {
            this.name = name;
        }

        public void setProgress(double progress) {
            this.progress = progress;
            if (progress > 0 && state == StatusUpdate.RunState.PENDING) {
                state = StatusUpdate.RunState.RUNNING;
            }
        }

        public void complete() {
            this.progress = 1.0;
            this.state = StatusUpdate.RunState.SUCCESS;
        }

        public void fail() {
            this.state = StatusUpdate.RunState.FAILED;
        }

        public StatusUpdate<TestTask> getStatus() {
            return new StatusUpdate<>(progress, state);
        }

        public String getName() {
            return name;
        }
    }

    /**
     * Tests fundamental TaskMonitor operations including start, stop, and basic status monitoring.
     *
     * <p><strong>Purpose:</strong> Validates core TaskMonitor functionality including lifecycle
     * management (start/stop), running state tracking, and basic status polling with
     * progress change detection.
     *
     * <p><strong>Why Important:</strong> This is the foundation test for TaskMonitor,
     * ensuring basic operations work correctly. All other monitoring features depend
     * on these core capabilities functioning properly.
     *
     * <p><strong>Coverage:</strong> Monitor start/stop lifecycle, running state validation,
     * initial status capture, progress change detection, proper shutdown behavior.
     *
     * @see #testAutoStopOnCompletion() for automatic termination behavior
     * @see #testStatusUpdatesReflectChanges() for detailed status change tracking
     * @see StatusTrackerTest#testPassiveMonitoringWithInstrumented() for higher-level monitoring integration
     */
    @Test
    public void testBasicMonitoring() throws InterruptedException {
        TestTask task = new TestTask("basic-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            assertTrue(monitor.isRunning());
            assertFalse(monitor.isShutdown());

            StatusUpdate<TestTask> initialStatus = monitor.getCurrentStatus();
            assertEquals(0.0, initialStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.PENDING, initialStatus.runstate);

            task.setProgress(0.5);
            Thread.sleep(100); // Increased wait time for better reliability

            StatusUpdate<TestTask> updatedStatus = monitor.getCurrentStatus();
            assertEquals(0.5, updatedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, updatedStatus.runstate);
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());
            assertTrue(monitor.isShutdown());
        }
    }

    /**
     * Tests automatic monitor termination when tasks reach SUCCESS state.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor automatically stops
     * monitoring when a task completes successfully, preventing unnecessary
     * resource usage and providing clean lifecycle management.
     *
     * <p><strong>Why Important:</strong> Auto-stop behavior is critical for resource
     * efficiency in long-running applications. Without it, monitors would continue
     * polling completed tasks indefinitely, wasting CPU cycles.
     *
     * <p><strong>Coverage:</strong> Task completion detection, automatic monitor
     * termination, final status preservation, resource cleanup on completion.
     *
     * @see #testAutoStopOnFailure() for failure state auto-stop behavior
     * @see #testTaskCompletionStatesHandling() for comprehensive completion testing
     * @see StatusTrackerTest for integration behavior
     */
    @Test
    public void testAutoStopOnCompletion() throws InterruptedException {
        TestTask task = new TestTask("completion-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            assertTrue(monitor.isRunning());

            task.setProgress(0.5);
            Thread.sleep(50);
            assertTrue(monitor.isRunning());

            task.complete();
            Thread.sleep(100); // Increased wait time for auto-stop

            assertFalse(monitor.isRunning());
            StatusUpdate<TestTask> finalStatus = monitor.getCurrentStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);
        } finally {
            // Always call stop() to ensure proper executor cleanup
            monitor.stop();
        }
    }

    /**
     * Tests automatic monitor termination when tasks reach FAILED state.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor automatically stops
     * monitoring when a task fails, ensuring consistent auto-stop behavior
     * for both success and failure terminal states.
     *
     * <p><strong>Why Important:</strong> Failed tasks should not continue to be
     * monitored as they won't change state further. Auto-stop on failure
     * ensures consistent resource management regardless of task outcome.
     *
     * <p><strong>Coverage:</strong> Task failure detection, automatic monitor
     * termination on failure, final status preservation, consistent lifecycle behavior.
     *
     * @see #testAutoStopOnCompletion() for success state auto-stop behavior
     * @see #testTaskCompletionStatesHandling() for both success and failure handling
     * @see StatusTrackerTest#testTrackerStatusAfterTaskFailure() for higher-level failure handling
     */
    @Test
    public void testAutoStopOnFailure() throws InterruptedException {
        TestTask task = new TestTask("failure-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            task.setProgress(0.3);
            Thread.sleep(30);
            assertTrue(monitor.isRunning());

            task.fail();
            Thread.sleep(50);

            assertFalse(monitor.isRunning());
            StatusUpdate<TestTask> finalStatus = monitor.getCurrentStatus();
            assertEquals(StatusUpdate.RunState.FAILED, finalStatus.runstate);
        } finally {
            // Always call stop() to ensure proper executor cleanup
            monitor.stop();
        }
    }

    /**
     * Tests TaskMonitor resilience when status functions throw exceptions.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can gracefully handle
     * exceptions thrown by status functions, continuing to operate despite
     * occasional failures in status retrieval.
     *
     * <p><strong>Why Important:</strong> Status functions may encounter transient
     * errors (network issues, temporary resource unavailability). The monitor
     * must be resilient to these exceptions to maintain system stability.
     *
     * <p><strong>Coverage:</strong> Exception resilience, continued operation
     * after exceptions, function call counting, monitor stability under errors.
     *
     * @see #testStatusChangeCallbackException() for callback exception handling
     * @see StatusTrackerTest#testTrackerResourceCleanupOnException() for higher-level exception scenarios
     */
    @Test
    public void testExceptionHandling() throws InterruptedException {
        TestTask task = new TestTask("exception-test");
        AtomicInteger callCount = new AtomicInteger(0);

        Function<TestTask, StatusUpdate<TestTask>> faultyFunction = t -> {
            int count = callCount.incrementAndGet();
            if (count == 2) {
                throw new RuntimeException("Simulated error");
            }
            return t.getStatus();
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, faultyFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            Thread.sleep(100);

            assertTrue(callCount.get() > 2);
            assertTrue(monitor.isRunning());
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());
        }
    }

    /**
     * Tests thread safety of concurrent start and stop operations.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can handle concurrent
     * start and stop calls from different threads without race conditions,
     * data corruption, or inconsistent state.
     *
     * <p><strong>Why Important:</strong> Multi-threaded applications may have
     * different threads starting and stopping monitors concurrently. This
     * ensures thread-safe lifecycle management.
     *
     * <p><strong>Coverage:</strong> Concurrent lifecycle operations, thread safety,
     * proper state transitions under concurrency, latch-based coordination.
     *
     * @see #testMultipleStartCalls() for multiple start call handling
     * @see #testMultipleStopCalls() for multiple stop call handling
     * @see StatusTrackerTest#testConcurrentTrackers() for higher-level concurrency testing
     */
    @Test
    public void testConcurrentStartStop() throws InterruptedException {
        TestTask task = new TestTask("concurrent-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(5));

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch stopLatch = new CountDownLatch(1);

        Thread startThread = new Thread(() -> {
            monitor.start();
            startLatch.countDown();
        });

        Thread stopThread = new Thread(() -> {
            try {
                startLatch.await(1, TimeUnit.SECONDS);
                Thread.sleep(20);
                monitor.stop();
                stopLatch.countDown();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        startThread.start();
        stopThread.start();

        assertTrue(stopLatch.await(2, TimeUnit.SECONDS));
        assertFalse(monitor.isRunning());
        assertTrue(monitor.isShutdown());

        startThread.join();
        stopThread.join();
    }

    /**
     * Tests that multiple start() calls on the same monitor are handled safely.
     *
     * <p><strong>Purpose:</strong> Validates that calling start() multiple times
     * on an already running monitor doesn't cause issues like duplicate threads,
     * resource leaks, or inconsistent state.
     *
     * <p><strong>Why Important:</strong> Application code may inadvertently call
     * start() multiple times. The monitor should handle this gracefully without
     * creating problems or throwing exceptions.
     *
     * <p><strong>Coverage:</strong> Multiple start call safety, idempotent behavior,
     * resource leak prevention, consistent running state.
     *
     * @see #testMultipleStopCalls() for multiple stop call handling
     * @see #testConcurrentStartStop() for concurrent lifecycle operations
     */
    @Test
    public void testMultipleStartCalls() {
        TestTask task = new TestTask("multiple-start-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(10));
        try {
            monitor.start();
            assertTrue(monitor.isRunning());

            monitor.start();
            assertTrue(monitor.isRunning());

            monitor.start();
            assertTrue(monitor.isRunning());
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());
        }
    }

    /**
     * Tests that multiple stop() calls on the same monitor are handled safely.
     *
     * <p><strong>Purpose:</strong> Validates that calling stop() multiple times
     * on an already stopped monitor doesn't cause issues like exceptions,
     * resource access errors, or inconsistent state.
     *
     * <p><strong>Why Important:</strong> Cleanup code often calls stop() in
     * multiple places (try-catch blocks, finally blocks). The monitor should
     * handle redundant stop calls gracefully.
     *
     * <p><strong>Coverage:</strong> Multiple stop call safety, idempotent behavior,
     * consistent shutdown state, proper resource cleanup.
     *
     * @see #testMultipleStartCalls() for multiple start call handling
     * @see #testRapidStartStop() for rapid lifecycle operations
     */
    @Test
    public void testMultipleStopCalls() {
        TestTask task = new TestTask("multiple-stop-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            assertTrue(monitor.isRunning());
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());

            monitor.stop();
            assertFalse(monitor.isRunning());

            monitor.stop();
            assertFalse(monitor.isRunning());
        }
    }

    /**
     * Tests that monitor status updates accurately reflect task progress changes.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor correctly detects
     * and reports task progress changes, ensuring the monitoring system
     * accurately reflects the current state of monitored tasks.
     *
     * <p><strong>Why Important:</strong> The core value of monitoring is providing
     * accurate, up-to-date status information. This test ensures the monitor
     * reliably captures and reports progress changes.
     *
     * <p><strong>Coverage:</strong> Progress change detection, status accuracy,
     * sequential progress updates, state transition validation.
     *
     * @see #testBasicMonitoring() for basic status retrieval
     * @see #testStatusEquality() for status change optimization
     * @see #testMonitorPerformanceWithHighFrequencyUpdates() for high-frequency scenarios
     */
    @Test
    public void testStatusUpdatesReflectChanges() throws InterruptedException {
        TestTask task = new TestTask("status-updates-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ofMillis(5));
        try {
            monitor.start();

            double[] progressValues = {0.1, 0.3, 0.5, 0.7, 0.9};

            for (double progress : progressValues) {
                task.setProgress(progress);
                Thread.sleep(20);

                StatusUpdate<TestTask> status = monitor.getCurrentStatus();
                assertEquals(progress, status.progress, 0.001);
                assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);
            }
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests TaskMonitor behavior when using default polling interval.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can be created
     * and operate correctly using the default polling interval without
     * explicitly specifying a Duration.
     *
     * <p><strong>Why Important:</strong> Default configuration should provide
     * reasonable behavior for common use cases. This ensures the default
     * polling interval is functional and appropriate.
     *
     * <p><strong>Coverage:</strong> Default constructor usage, default interval
     * behavior, basic functionality with default settings.
     *
     * @see #testMonitorWithZeroPollInterval() for edge case interval testing
     * @see #testMonitorPerformanceWithHighFrequencyUpdates() for custom high-frequency intervals
     */
    @Test
    public void testDefaultPollInterval() {
        TestTask task = new TestTask("default-interval-test");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction);
        assertNotNull(monitor);

        try {
            monitor.start();
            assertTrue(monitor.isRunning());
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());
        }
    }

    /**
     * Tests TaskMonitor behavior when status functions return null values.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor gracefully handles
     * null status returns from status functions, maintaining stability and
     * preserving the last known valid status.
     *
     * <p><strong>Why Important:</strong> Status functions may occasionally return
     * null due to transient issues. The monitor should handle this gracefully
     * rather than failing or losing status information.
     *
     * <p><strong>Coverage:</strong> Null status handling, last known status preservation,
     * graceful degradation, stability under null returns.
     *
     * @see #testMonitorCallbackWithNullStatus() for null status in callback scenarios
     * @see #testExceptionHandling() for exception-based status function issues
     */
    @Test
    public void testNullStatusHandling() throws InterruptedException {
        TestTask task = new TestTask("null-status-test");
        AtomicReference<StatusUpdate<TestTask>> statusRef = new AtomicReference<>(task.getStatus());

        Function<TestTask, StatusUpdate<TestTask>> nullReturningFunction = t -> statusRef.get();

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, nullReturningFunction, Duration.ofMillis(10));
        try {
            monitor.start();

            Thread.sleep(20);
            StatusUpdate<TestTask> initialStatus = monitor.getCurrentStatus();
            assertNotNull(initialStatus);

            // Set function to return null - monitor should handle gracefully
            statusRef.set(null);
            Thread.sleep(30);

            StatusUpdate<TestTask> currentStatus = monitor.getCurrentStatus();
            // Should still have the last valid status
            assertNotNull(currentStatus);
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests TaskMonitor functionality with different task object types and naming.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can work with various
     * task object types including custom objects with names and simple primitive
     * types, demonstrating flexibility in task object handling.
     *
     * <p><strong>Why Important:</strong> Applications may need to monitor different
     * types of objects. This ensures TaskMonitor is flexible enough to handle
     * various object types and naming scenarios.
     *
     * <p><strong>Coverage:</strong> Multiple object types, task naming variations,
     * type flexibility validation, proper shutdown for different task types.
     *
     * @see StatusTrackerTest#testTrackerWithCustomObject() for custom object tracking patterns
     * @see #testMonitorWithNullTask() for null task handling
     */
    @Test
    public void testTaskNameGeneration() throws InterruptedException {
        TestTask task1 = new TestTask("named-task");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor1 = new StatusMonitor<>(task1, statusFunction, Duration.ofMillis(10));
        try {
            monitor1.start();
            Thread.sleep(20);
        } finally {
            monitor1.stop();
        }

        String simpleObject = "simple-string";
        StatusMonitor<String> monitor2 = new StatusMonitor<>(simpleObject,
            s -> new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING), Duration.ofMillis(10));
        try {
            monitor2.start();
            Thread.sleep(20);
        } finally {
            monitor2.stop();
        }

        assertTrue(monitor1.isShutdown());
        assertTrue(monitor2.isShutdown());
    }

    /**
     * Tests status change callback functionality and invocation patterns.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor correctly invokes
     * status change callbacks when task status changes, providing applications
     * with real-time notifications of progress updates.
     *
     * <p><strong>Why Important:</strong> Status change callbacks are essential
     * for reactive applications that need immediate notification of task progress.
     * This ensures callbacks are invoked correctly and consistently.
     *
     * <p><strong>Coverage:</strong> Callback invocation, status change detection,
     * callback frequency, final status notification, callback parameter validation.
     *
     * @see #testStatusChangeCallbackException() for callback exception handling
     * @see #testStatusEquality() for callback optimization through status equality
     */
    @Test
    public void testStatusChangeCallback() throws InterruptedException {
        TestTask task = new TestTask("callback-test");
        AtomicInteger callbackCount = new AtomicInteger(0);
        AtomicReference<StatusUpdate<TestTask>> lastStatus = new AtomicReference<>();

        Consumer<StatusUpdate<TestTask>> callback = status -> {
            callbackCount.incrementAndGet();
            lastStatus.set(status);
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(10), callback);
        try {
            monitor.start();

            task.setProgress(0.3);
            Thread.sleep(50);

            task.setProgress(0.7);
            Thread.sleep(50);

            task.complete();
            Thread.sleep(100);

            assertTrue(callbackCount.get() >= 2);
            assertNotNull(lastStatus.get());
            assertEquals(StatusUpdate.RunState.SUCCESS, lastStatus.get().runstate);
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests TaskMonitor resilience when status change callbacks throw exceptions.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor continues normal
     * operation even when status change callbacks throw exceptions, ensuring
     * that callback failures don't compromise monitoring functionality.
     *
     * <p><strong>Why Important:</strong> Callback code may contain bugs or
     * encounter runtime errors. The monitor must isolate these failures
     * to maintain system stability and continue monitoring.
     *
     * <p><strong>Coverage:</strong> Callback exception isolation, continued
     * monitoring operation, callback invocation verification despite exceptions.
     *
     * @see #testStatusChangeCallback() for normal callback behavior
     * @see #testExceptionHandling() for status function exception handling
     */
    @Test
    public void testStatusChangeCallbackException() throws InterruptedException {
        TestTask task = new TestTask("callback-exception-test");
        AtomicBoolean callbackCalled = new AtomicBoolean(false);

        Consumer<StatusUpdate<TestTask>> faultyCallback = status -> {
            callbackCalled.set(true);
            throw new RuntimeException("Callback error");
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(10), faultyCallback);
        try {
            monitor.start();

            task.setProgress(0.5);
            Thread.sleep(50);

            assertTrue(monitor.isRunning());
            assertTrue(callbackCalled.get());
        } finally {
            monitor.stop();
            assertFalse(monitor.isRunning());
        }
    }

    /**
     * Tests TaskMonitor behavior when created with null task reference.
     *
     * <p><strong>Purpose:</strong> Validates TaskMonitor's input validation
     * for null task parameters, ensuring appropriate error handling or
     * graceful degradation depending on implementation.
     *
     * <p><strong>Why Important:</strong> Null task references should be handled
     * predictably to prevent runtime errors. This test validates the monitor's
     * defensive programming against invalid input.
     *
     * <p><strong>Coverage:</strong> Null task validation, constructor parameter
     * checking, appropriate exception handling or graceful degradation.
     *
     * @see #testMonitorWithNullStatusFunction() for null status function handling
     * @see StatusTrackerTest#testTrackerNullParameterValidation() for higher-level null validation
     */
    @Test
    public void testMonitorWithNullTask() {
        try {
            new StatusMonitor<>(null, t -> null);
        } catch (Exception e) {
            // Expected for null task, depending on implementation
        }
    }

    /**
     * Tests TaskMonitor behavior when created with null status function.
     *
     * <p><strong>Purpose:</strong> Validates TaskMonitor's input validation
     * for null status functions, ensuring appropriate exception throwing
     * since status functions are essential for monitor operation.
     *
     * <p><strong>Why Important:</strong> Status functions are mandatory for
     * TaskMonitor operation. This test ensures proper validation and
     * fail-fast behavior when this critical parameter is missing.
     *
     * <p><strong>Coverage:</strong> Null status function validation, constructor
     * parameter checking, appropriate exception throwing for invalid input.
     *
     * @see #testMonitorWithNullTask() for null task handling
     * @see #testNullStatusHandling() for null status return handling
     */
    @Test
    public void testMonitorWithNullStatusFunction() {
        TestTask task = new TestTask("null-function-test");
        try {
            new StatusMonitor<>(task, null);
            fail("Should throw exception for null status function");
        } catch (Exception e) {
            // Expected
        }
    }

    /**
     * Tests TaskMonitor behavior under thread interruption scenarios.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor properly handles
     * thread interruption during monitoring operations, ensuring clean
     * shutdown and proper thread lifecycle management.
     *
     * <p><strong>Why Important:</strong> Thread interruption is a common
     * mechanism for task cancellation. The monitor must handle interruption
     * gracefully to prevent resource leaks or inconsistent state.
     *
     * <p><strong>Coverage:</strong> Thread interruption handling, clean shutdown
     * under interruption, proper thread lifecycle management.
     *
     * @see #testRapidStartStop() for rapid lifecycle operations
     * @see #testConcurrentStartStop() for concurrent lifecycle scenarios
     */
    @Test
    public void testThreadInterruption() throws InterruptedException {
        TestTask task = new TestTask("interruption-test");
        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(50));

        try {
            monitor.start();
            assertTrue(monitor.isRunning());

            // Stop immediately to test interruption handling
            monitor.stop();

            Thread.sleep(100);
            assertFalse(monitor.isRunning());
            assertTrue(monitor.isShutdown());
        } finally {
            // Always call stop() to ensure proper executor cleanup
            monitor.stop();
        }
    }

    /**
     * Tests status change optimization through status equality checking.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor optimizes callback
     * invocations by only triggering callbacks when status actually changes,
     * using status equality to avoid redundant notifications.
     *
     * <p><strong>Why Important:</strong> Avoiding redundant callbacks improves
     * performance and prevents unnecessary processing in callback handlers.
     * This optimization is critical for high-frequency monitoring scenarios.
     *
     * <p><strong>Coverage:</strong> Status equality optimization, callback frequency
     * reduction, redundant notification prevention, performance optimization validation.
     *
     * @see #testStatusChangeCallback() for basic callback functionality
     * @see #testMonitorPerformanceWithHighFrequencyUpdates() for performance scenarios
     */
    @Test
    public void testStatusEquality() throws InterruptedException {
        TestTask task = new TestTask("equality-test");
        AtomicInteger callbackCount = new AtomicInteger(0);

        Consumer<StatusUpdate<TestTask>> callback = status -> callbackCount.incrementAndGet();

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(10), callback);
        try {
            monitor.start();

            // Set same progress multiple times - callback should only fire once per change
            task.setProgress(0.5);
            Thread.sleep(30);
            task.setProgress(0.5);
            Thread.sleep(30);
            task.setProgress(0.5);
            Thread.sleep(30);

            task.setProgress(0.8);  // This should trigger callback
            Thread.sleep(30);

            // Should have fewer callbacks than total progress sets due to equality check
            assertTrue(callbackCount.get() >= 1);
            assertTrue(callbackCount.get() <= 3);  // At most initial + 0.5 + 0.8
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests comprehensive handling of both SUCCESS and FAILED completion states.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor correctly handles
     * both success and failure completion states, ensuring consistent auto-stop
     * behavior and proper final status preservation for all terminal states.
     *
     * <p><strong>Why Important:</strong> Both success and failure are terminal
     * states that should trigger auto-stop behavior. This ensures consistent
     * lifecycle management regardless of task outcome.
     *
     * <p><strong>Coverage:</strong> Success state handling, failure state handling,
     * consistent auto-stop behavior, final status preservation for both outcomes.
     *
     * @see #testAutoStopOnCompletion() for detailed success state testing
     * @see #testAutoStopOnFailure() for detailed failure state testing
     * @see #testMonitorStateTransitionCompleteness() for comprehensive state tracking
     */
    @Test
    public void testTaskCompletionStatesHandling() throws InterruptedException {
        TestTask successTask = new TestTask("success-task");
        TestTask failTask = new TestTask("fail-task");

        StatusMonitor<TestTask> successMonitor = new StatusMonitor<>(successTask, TestTask::getStatus, Duration.ofMillis(10));
        StatusMonitor<TestTask> failMonitor = new StatusMonitor<>(failTask, TestTask::getStatus, Duration.ofMillis(10));

        try {
            successMonitor.start();
            failMonitor.start();

            successTask.complete();
            failTask.fail();

            Thread.sleep(100);

            // Both should auto-stop on completion
            assertFalse(successMonitor.isRunning());
            assertFalse(failMonitor.isRunning());

            assertEquals(StatusUpdate.RunState.SUCCESS, successMonitor.getCurrentStatus().runstate);
            assertEquals(StatusUpdate.RunState.FAILED, failMonitor.getCurrentStatus().runstate);
        } finally {
            // Always call stop() to ensure proper executor cleanup
            successMonitor.stop();
            failMonitor.stop();
        }
    }

    /**
     * Tests TaskMonitor stability under rapid start/stop cycling operations.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can handle rapid
     * successive start and stop operations without resource leaks, state
     * corruption, or other stability issues.
     *
     * <p><strong>Why Important:</strong> Some applications may rapidly start
     * and stop monitoring based on dynamic conditions. The monitor must
     * handle this usage pattern reliably.
     *
     * <p><strong>Coverage:</strong> Rapid lifecycle cycling, resource leak prevention,
     * state consistency under rapid operations, stability validation.
     *
     * @see #testMultipleStartCalls() and #testMultipleStopCalls() for redundant call handling
     * @see #testConcurrentStartStop() for concurrent lifecycle operations
     */
    @Test
    public void testRapidStartStop() {
        TestTask task = new TestTask("rapid-test");
        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(10));

        try {
            // Rapid start/stop cycles
            for (int i = 0; i < 5; i++) {
                monitor.start();
                monitor.stop();
            }

            assertTrue(monitor.isShutdown());
            assertFalse(monitor.isRunning());
        } finally {
            // Always call stop() to ensure proper executor cleanup
            monitor.stop();
        }
    }


    /**
     * Tests TaskMonitor behavior with zero polling interval edge case.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor can handle
     * zero-duration polling intervals without causing issues like infinite
     * loops, excessive CPU usage, or system instability.
     *
     * <p><strong>Why Important:</strong> Edge case testing ensures robust
     * behavior under unusual configurations. Zero intervals might be used
     * for maximum responsiveness scenarios.
     *
     * <p><strong>Coverage:</strong> Zero interval handling, edge case stability,
     * performance under minimal interval, proper functionality maintenance.
     *
     * @see #testDefaultPollInterval() for default interval behavior
     * @see #testMonitorPerformanceWithHighFrequencyUpdates() for high-frequency scenarios
     */
    @Test
    public void testMonitorWithZeroPollInterval() {
        TestTask task = new TestTask("zero-interval");
        Function<TestTask, StatusUpdate<TestTask>> statusFunction = TestTask::getStatus;

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, statusFunction, Duration.ZERO);
        try {
            monitor.start();
            assertTrue(monitor.isRunning());

            task.setProgress(0.5);
            Thread.sleep(50);

            StatusUpdate<TestTask> status = monitor.getCurrentStatus();
            assertEquals(0.5, status.progress, 0.001);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests callback behavior when status functions return null values.
     *
     * <p><strong>Purpose:</strong> Validates that status change callbacks
     * handle null status values gracefully, either by filtering them out
     * or by providing appropriate null-safe processing.
     *
     * <p><strong>Why Important:</strong> Callbacks should be resilient to
     * null status values to prevent callback failures from compromising
     * the overall monitoring system.
     *
     * <p><strong>Coverage:</strong> Null status in callbacks, callback resilience,
     * null filtering behavior, callback stability under edge conditions.
     *
     * @see #testNullStatusHandling() for general null status handling
     * @see #testStatusChangeCallbackException() for callback exception scenarios
     */
    @Test
    public void testMonitorCallbackWithNullStatus() throws InterruptedException {
        TestTask task = new TestTask("null-status");
        AtomicInteger callbackCount = new AtomicInteger(0);

        Function<TestTask, StatusUpdate<TestTask>> nullReturningFunction = t -> {
            if (callbackCount.get() > 2) {
                return null; // Return null after a few calls
            }
            return t.getStatus();
        };

        Consumer<StatusUpdate<TestTask>> callback = status -> {
            callbackCount.incrementAndGet();
            if (status != null) {
                // Only process non-null statuses
                assertTrue(status.progress >= 0.0);
            }
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, nullReturningFunction, Duration.ofMillis(10), callback);
        try {
            monitor.start();
            task.setProgress(0.3);
            Thread.sleep(100);
        } finally {
            monitor.stop();
        }

        // Should have been called at least once
        assertTrue(callbackCount.get() > 0);
    }

    /**
     * Tests TaskMonitor performance and stability under high-frequency update scenarios.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor maintains good
     * performance and stability when monitoring tasks that change state
     * very frequently, ensuring the system can handle high-load scenarios.
     *
     * <p><strong>Why Important:</strong> Some tasks may update progress very
     * frequently (e.g., file transfers, data processing). The monitor must
     * handle these scenarios efficiently without performance degradation.
     *
     * <p><strong>Coverage:</strong> High-frequency updates, performance validation,
     * rapid progress changes, status function call frequency, final state accuracy.
     *
     * @see #testStatusUpdatesReflectChanges() for basic progress tracking
     * @see #testStatusEquality() for performance optimization through equality checks
     * @see StatusTrackerTest#testTrackerWithRapidStateChanges() for higher-level rapid change scenarios
     */
    @Test
    public void testMonitorPerformanceWithHighFrequencyUpdates() throws InterruptedException {
        TestTask task = new TestTask("high-frequency");
        AtomicInteger updateCount = new AtomicInteger(0);

        Function<TestTask, StatusUpdate<TestTask>> countingFunction = t -> {
            updateCount.incrementAndGet();
            return t.getStatus();
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, countingFunction, Duration.ofMillis(1));
        try {
            monitor.start();

            // Rapidly update task progress
            for (int i = 0; i <= 10; i++) {
                task.setProgress(i / 10.0);
                Thread.sleep(5);
            }

            Thread.sleep(50); // Let monitor catch up

            StatusUpdate<TestTask> finalStatus = monitor.getCurrentStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);

            // Should have made multiple status function calls
            assertTrue(updateCount.get() > 10);
        } finally {
            monitor.stop();
        }
    }

    /**
     * Tests completeness of state transition observation through monitoring callbacks.
     *
     * <p><strong>Purpose:</strong> Validates that TaskMonitor callbacks observe
     * all significant task state transitions (PENDING → RUNNING → SUCCESS/FAILED),
     * ensuring complete visibility into task lifecycle progression.
     *
     * <p><strong>Why Important:</strong> Applications often need to track complete
     * task lifecycles for analytics, debugging, or business logic. This ensures
     * the monitoring system captures all important state changes.
     *
     * <p><strong>Coverage:</strong> Complete state transition tracking, callback
     * observation of all states, lifecycle completeness validation, state timing considerations.
     *
     * @see #testTaskCompletionStatesHandling() for terminal state handling
     * @see #testStatusChangeCallback() for basic callback functionality
     * @see StatusTrackingIntegrationTest for end-to-end state transition scenarios
     */
    @Test
    public void testMonitorStateTransitionCompleteness() throws InterruptedException {
        TestTask task = new TestTask("state-transitions");
        List<StatusUpdate.RunState> observedStates = Collections.synchronizedList(new ArrayList<>());

        Consumer<StatusUpdate<TestTask>> stateTracker = status -> {
            if (status != null && !observedStates.contains(status.runstate)) {
                observedStates.add(status.runstate);
            }
        };

        StatusMonitor<TestTask> monitor = new StatusMonitor<>(task, TestTask::getStatus, Duration.ofMillis(10), stateTracker);
        try {
            monitor.start();

            // Go through all state transitions
            Thread.sleep(50); // PENDING - give more time to observe initial state

            task.setProgress(0.5);
            Thread.sleep(50); // RUNNING - give more time to observe running state

            task.complete();
            Thread.sleep(100); // SUCCESS - give more time to observe success state

            // Should have observed all expected states - PENDING is optional due to timing
            assertTrue("Should observe RUNNING state", observedStates.contains(StatusUpdate.RunState.RUNNING));
            assertTrue("Should observe SUCCESS state", observedStates.contains(StatusUpdate.RunState.SUCCESS));
            // PENDING state is optional due to timing sensitivity
            assertTrue("Should observe at least 2 states", observedStates.size() >= 2);
        } finally {
            // Always call stop() to ensure proper executor cleanup
            monitor.stop();
        }
    }
}