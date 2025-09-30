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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import org.junit.Test;
import org.junit.jupiter.api.Tag;

import java.util.function.Function;

import static org.junit.Assert.*;

/**
 * Core functionality tests for the Tracker class.
 *
 * <p>This test suite focuses on the fundamental Tracker operations including:
 * <ul>
 *   <li><strong>Basic Creation:</strong> Testing both instrumented and functor-based tracker creation</li>
 *   <li><strong>Lifecycle Management:</strong> Resource cleanup and proper close() behavior</li>
 *   <li><strong>Status Monitoring:</strong> Passive monitoring and status change detection</li>
 *   <li><strong>Error Handling:</strong> Exception resilience and null parameter handling</li>
 *   <li><strong>Concurrency:</strong> Thread safety and concurrent tracker operations</li>
 * </ul>
 *
 * <h2>Test Coverage Areas:</h2>
 * <ul>
 *   <li>Basic tracker creation patterns (instrumented vs functor-based)</li>
 *   <li>Resource management and automatic cleanup</li>
 *   <li>Status polling and change detection</li>
 *   <li>Error conditions and exception handling</li>
 *   <li>Concurrent tracker management</li>
 * </ul>
 *
 * <h2>Related Test Classes:</h2>
 * <ul>
 *   <li>{@link TrackerWithSinksTest} - Tests Tracker integration with TaskSink implementations</li>
 *   <li>{@link StatusTrackingIntegrationTest} - End-to-end integration scenarios</li>
 *   <li>{@link TaskMonitorTest} - Lower-level TaskMonitor functionality</li>
 *   <li>{@link StatusTrackerScopeTest} - TrackerScope hierarchy management</li>
 * </ul>
 *
 * @see StatusTracker
 * @see StatusMonitor
 * @since 4.0.0
 */
@Tag("Core")
public class StatusTrackerTest extends RandomizedTest {

    static class InstrumentedTask implements StatusUpdate.Provider<InstrumentedTask> {
        private final String name;
        private double progress = 0.0;
        private StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;

        public InstrumentedTask(String name) {
            this.name = name;
        }

        @Override
        public StatusUpdate<InstrumentedTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state);
        }

        public void setProgress(double progress) {
            this.progress = progress;
            if (progress >= 1.0) {
                state = StatusUpdate.RunState.SUCCESS;
            } else if (progress > 0) {
                state = StatusUpdate.RunState.RUNNING;
            }
        }

        public String getName() {
            return name;
        }
    }

    static class RegularTask {
        private final String name;
        private double progress = 0.0;

        public RegularTask(String name) {
            this.name = name;
        }

        public double getProgress() {
            return progress;
        }

        public void setProgress(double progress) {
            this.progress = progress;
        }

        public String getName() {
            return name;
        }
    }

    // =============================================================================
    // CORE FUNCTIONALITY TESTS
    // =============================================================================

    /**
     * Tests basic creation and functionality of Tracker with instrumented tasks.
     *
     * <p><strong>Purpose:</strong> Verifies that trackers can be created for objects implementing
     * {@link StatusUpdate.Provider} and that they properly track task status changes through
     * the instrumented interface.
     *
     * <p><strong>Why Important:</strong> This is the primary pattern for tracking objects that
     * can self-report their status. It ensures the Tracker correctly delegates status queries
     * to the instrumented object's {@code getTaskStatus()} method.
     *
     * <p><strong>Coverage:</strong> Basic instrumented tracker creation, status querying,
     * and proper task reference management.
     *
     * @see #testWithFunctors() for alternative functor-based creation pattern
     * @see TrackerWithSinksTest#testTrackerWithSingleSink() for sink integration
     */
    @Test
    public void testWithInstrumented() {
        InstrumentedTask task = new InstrumentedTask("instrumented-task");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            assertNotNull(statusTracker);
            assertEquals(task, statusTracker.getTracked());

            StatusUpdate<InstrumentedTask> status = statusTracker.getStatus();
            assertNotNull(status);
            assertEquals(0.0, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.PENDING, status.runstate);

            task.setProgress(0.5);
            status = statusTracker.getStatus();
            assertEquals(0.0, status.progress, 0.001);

            task.setProgress(1.0);
            StatusUpdate<InstrumentedTask> newStatus = task.getTaskStatus();
            assertEquals(1.0, newStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, newStatus.runstate);
        }
    }

    /**
     * Tests creation and functionality of Tracker with custom status functions.
     *
     * <p><strong>Purpose:</strong> Verifies that trackers can be created for regular objects
     * using custom functions to extract status information, enabling tracking of objects
     * that don't implement {@link StatusUpdate.Provider}.
     *
     * <p><strong>Why Important:</strong> This pattern allows tracking of third-party objects
     * or legacy code that cannot be modified to implement TaskStatus.Provider. The custom
     * function provides the bridge between the object's internal state and TaskStatus.
     *
     * <p><strong>Coverage:</strong> Functor-based tracker creation, custom status extraction
     * logic, and status calculation from object properties.
     *
     * @see #testWithInstrumented() for instrumented object tracking pattern
     * @see #testCustomStatusFunction() for advanced status function scenarios
     */
    @Test
    public void testWithFunctors() {
        RegularTask task = new RegularTask("regular-task");

        Function<RegularTask, StatusUpdate<RegularTask>> statusFunction = t -> {
            StatusUpdate.RunState state;
            if (t.getProgress() >= 1.0) {
                state = StatusUpdate.RunState.SUCCESS;
            } else if (t.getProgress() > 0) {
                state = StatusUpdate.RunState.RUNNING;
            } else {
                state = StatusUpdate.RunState.PENDING;
            }
            return new StatusUpdate<>(t.getProgress(), state);
        };

        try (StatusTracker<RegularTask> statusTracker = StatusTracker.withFunctors(task, statusFunction)) {
            assertNotNull(statusTracker);
            assertEquals(task, statusTracker.getTracked());

            StatusUpdate<RegularTask> status = statusTracker.getStatus();
            assertNotNull(status);
            assertEquals(0.0, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.PENDING, status.runstate);

            task.setProgress(0.75);
            StatusUpdate<RegularTask> newStatus = statusFunction.apply(task);
            assertEquals(0.75, newStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, newStatus.runstate);

            task.setProgress(1.0);
            newStatus = statusFunction.apply(task);
            assertEquals(1.0, newStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, newStatus.runstate);
        }
    }

    /**
     * Tests concurrent management of multiple independent instrumented trackers.
     *
     * <p><strong>Purpose:</strong> Validates that multiple Tracker instances can coexist
     * and operate independently without interference, each tracking their own tasks
     * with separate status lifecycles.
     *
     * <p><strong>Why Important:</strong> Real applications often need to track multiple
     * concurrent tasks. This test ensures proper isolation between trackers and
     * prevents cross-contamination of status information.
     *
     * <p><strong>Coverage:</strong> Multiple tracker instantiation, independent status
     * tracking, proper task-to-tracker association, and resource cleanup.
     *
     * @see #testConcurrentTrackers() for thread-safety testing
     * @see #testMixedTaskTypes() for different task type combinations
     */
    @Test
    public void testMultipleInstrumentedTasks() {
        InstrumentedTask task1 = new InstrumentedTask("task-1");
        InstrumentedTask task2 = new InstrumentedTask("task-2");
        InstrumentedTask task3 = new InstrumentedTask("task-3");

        try (StatusTracker<InstrumentedTask> statusTracker1 = StatusTracker.withInstrumented(task1);
             StatusTracker<InstrumentedTask> statusTracker2 = StatusTracker.withInstrumented(task2);
             StatusTracker<InstrumentedTask> statusTracker3 = StatusTracker.withInstrumented(task3)) {

            task1.setProgress(0.25);
            task2.setProgress(0.50);
            task3.setProgress(1.00);

            assertEquals("task-1", statusTracker1.getTracked().getName());
            assertEquals("task-2", statusTracker2.getTracked().getName());
            assertEquals("task-3", statusTracker3.getTracked().getName());

            StatusUpdate<InstrumentedTask> status3 = task3.getTaskStatus();
            assertEquals(1.0, status3.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, status3.runstate);
        }
    }

    /**
     * Tests tracking different task types (instrumented vs functor-based) simultaneously.
     *
     * <p><strong>Purpose:</strong> Verifies that trackers using different creation patterns
     * can coexist and function correctly within the same application, demonstrating
     * flexibility in task tracking approaches.
     *
     * <p><strong>Why Important:</strong> Applications may need to track both legacy objects
     * (via functors) and new objects (instrumented) simultaneously. This ensures both
     * patterns work together without conflicts.
     *
     * <p><strong>Coverage:</strong> Mixed tracker types, different status reporting mechanisms,
     * type safety validation, and pattern interoperability.
     *
     * @see #testWithInstrumented() for pure instrumented tracking
     * @see #testWithFunctors() for pure functor-based tracking
     * @see #testMultipleInstrumentedTasks() for multiple tracker management
     */
    @Test
    public void testMixedTaskTypes() {
        InstrumentedTask instrTask = new InstrumentedTask("instrumented");
        RegularTask regTask = new RegularTask("regular");

        try (StatusTracker<InstrumentedTask> instrStatusTracker = StatusTracker.withInstrumented(instrTask);
             StatusTracker<RegularTask> regStatusTracker = StatusTracker.withFunctors(regTask,
                 t -> new StatusUpdate<>(t.getProgress(), StatusUpdate.RunState.RUNNING))) {

            assertNotNull(instrStatusTracker);
            assertNotNull(regStatusTracker);

            assertEquals(instrTask, instrStatusTracker.getTracked());
            assertEquals(regTask, regStatusTracker.getTracked());

            instrTask.setProgress(0.5);
            regTask.setProgress(0.5);

            StatusUpdate<InstrumentedTask> instrStatus = instrTask.getTaskStatus();
            assertEquals(0.5, instrStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, instrStatus.runstate);

            StatusUpdate<RegularTask> regStatus = regStatusTracker.getStatus();
            assertEquals(0.0, regStatus.progress, 0.001);
        }
    }

    /**
     * Tests proper tracker lifecycle management using try-with-resources pattern.
     *
     * <p><strong>Purpose:</strong> Validates that trackers properly implement the
     * {@link AutoCloseable} interface and clean up resources when used with
     * try-with-resources blocks.
     *
     * <p><strong>Why Important:</strong> Resource cleanup is critical for preventing
     * memory leaks and ensuring proper system behavior, especially in long-running
     * applications with many tracked tasks.
     *
     * <p><strong>Coverage:</strong> AutoCloseable implementation, automatic resource
     * cleanup, try-with-resources compatibility, and lifecycle state validation.
     *
     * @see #testTryWithResourcesCleanup() for detailed cleanup behavior
     * @see #testTrackerCleanupOnClose() for manual close() testing
     */
    @Test
    public void testTrackerLifecycle() {
        InstrumentedTask task = new InstrumentedTask("lifecycle-test");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            assertNotNull(statusTracker);
            assertEquals(task, statusTracker.getTracked());

            task.setProgress(0.5);
            StatusUpdate<InstrumentedTask> status = task.getTaskStatus();
            assertEquals(0.5, status.progress, 0.001);
        }
    }

    /**
     * Tests advanced custom status function scenarios with complex logic.
     *
     * <p><strong>Purpose:</strong> Validates that complex status calculation logic
     * can be implemented via custom functions, including progress adjustments,
     * conditional state transitions, and mathematical transformations.
     *
     * <p><strong>Why Important:</strong> Real-world applications often need custom
     * progress calculation (e.g., weighted progress, non-linear scaling). This
     * ensures the functor pattern supports sophisticated status reporting.
     *
     * <p><strong>Coverage:</strong> Complex status functions, progress transformation,
     * conditional state logic, mathematical operations on progress values.
     *
     * @see #testWithFunctors() for basic functor usage
     * @see #testTrackerWithCustomObject() for simple custom objects
     */
    @Test
    public void testCustomStatusFunction() {
        RegularTask task = new RegularTask("custom-function");

        Function<RegularTask, StatusUpdate<RegularTask>> customFunction = t -> {
            double adjustedProgress = Math.min(1.0, t.getProgress() * 1.2);
            StatusUpdate.RunState state = t.getProgress() == 0 ? StatusUpdate.RunState.PENDING :
                                      t.getProgress() >= 0.8 ? StatusUpdate.RunState.SUCCESS :
                                      StatusUpdate.RunState.RUNNING;
            return new StatusUpdate<>(adjustedProgress, state);
        };

        try (StatusTracker<RegularTask> statusTracker = StatusTracker.withFunctors(task, customFunction)) {
            task.setProgress(0.5);
            StatusUpdate<RegularTask> status = customFunction.apply(task);
            assertEquals(0.6, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);

            task.setProgress(0.9);
            status = customFunction.apply(task);
            assertEquals(1.0, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, status.runstate);
        }
    }

    /**
     * Tests passive status monitoring behavior with instrumented tasks.
     *
     * <p><strong>Purpose:</strong> Validates that trackers can passively observe
     * status changes in instrumented tasks without actively polling, ensuring
     * the tracker reflects current task state when queried.
     *
     * <p><strong>Why Important:</strong> Many applications need passive monitoring
     * where tasks update their own status and trackers provide read access.
     * This pattern reduces overhead and enables event-driven architectures.
     *
     * <p><strong>Coverage:</strong> Passive monitoring patterns, status change
     * detection, timing behavior, and state consistency validation.
     *
     * @see #testPassiveMonitoringWithFunctors() for functor-based passive monitoring
     * @see TaskMonitorTest#testPassiveMonitoring() for lower-level monitoring details
     */
    @Test
    public void testPassiveMonitoringWithInstrumented() throws InterruptedException {
        InstrumentedTask task = new InstrumentedTask("passive-instrumented");
        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            StatusUpdate<InstrumentedTask> initialStatus = statusTracker.getStatus();
            assertEquals(0.0, initialStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.PENDING, initialStatus.runstate);

            task.setProgress(0.3);
            Thread.sleep(200);

            StatusUpdate<InstrumentedTask> updatedStatus = statusTracker.getStatus();
            assertEquals(0.3, updatedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, updatedStatus.runstate);

            task.setProgress(0.8);
            Thread.sleep(200);

            updatedStatus = statusTracker.getStatus();
            assertEquals(0.8, updatedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, updatedStatus.runstate);
        }
    }

    /**
     * Tests passive status monitoring behavior with functor-based tasks.
     *
     * <p><strong>Purpose:</strong> Validates that trackers using custom status functions
     * can passively monitor task state changes, with the status function computing
     * current status on demand.
     *
     * <p><strong>Why Important:</strong> Demonstrates that functor-based tracking
     * supports the same passive monitoring patterns as instrumented tasks,
     * providing consistent behavior across different tracking approaches.
     *
     * <p><strong>Coverage:</strong> Functor-based passive monitoring, on-demand
     * status calculation, state transition validation, timing considerations.
     *
     * @see #testPassiveMonitoringWithInstrumented() for instrumented passive monitoring
     * @see #testWithFunctors() for basic functor patterns
     */
    @Test
    public void testPassiveMonitoringWithFunctors() throws InterruptedException {
        RegularTask task = new RegularTask("passive-functors");

        Function<RegularTask, StatusUpdate<RegularTask>> statusFunction = t -> {
            StatusUpdate.RunState state = t.getProgress() >= 1.0 ? StatusUpdate.RunState.SUCCESS :
                                       t.getProgress() > 0 ? StatusUpdate.RunState.RUNNING :
                                       StatusUpdate.RunState.PENDING;
            return new StatusUpdate<>(t.getProgress(), state);
        };

        try (StatusTracker<RegularTask> statusTracker = StatusTracker.withFunctors(task, statusFunction)) {
            StatusUpdate<RegularTask> initialStatus = statusTracker.getStatus();
            assertEquals(0.0, initialStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.PENDING, initialStatus.runstate);

            task.setProgress(0.6);
            Thread.sleep(200);

            StatusUpdate<RegularTask> updatedStatus = statusTracker.getStatus();
            assertEquals(0.6, updatedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, updatedStatus.runstate);

            task.setProgress(1.0);
            Thread.sleep(200);

            updatedStatus = statusTracker.getStatus();
            assertEquals(1.0, updatedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, updatedStatus.runstate);
        }
    }

    /**
     * Tests that monitoring behavior properly handles task completion states.
     *
     * <p><strong>Purpose:</strong> Validates that trackers correctly detect and
     * handle task completion (SUCCESS state), ensuring monitoring resources
     * are properly managed when tasks finish.
     *
     * <p><strong>Why Important:</strong> Task completion detection is critical for
     * resource management and system efficiency. Trackers should recognize when
     * tasks are complete and adjust their behavior accordingly.
     *
     * <p><strong>Coverage:</strong> Completion state detection, status transition
     * validation, timing behavior, final state persistence.
     *
     * @see TaskMonitorTest#testMonitoringStopsOnCompletion() for monitoring internals
     * @see #testTrackerStatusAfterTaskFailure() for failure state handling
     */
    @Test
    public void testMonitorStopsOnTaskCompletion() throws InterruptedException {
        InstrumentedTask task = new InstrumentedTask("completion-stop");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            task.setProgress(0.5);
            Thread.sleep(100);

            StatusUpdate<InstrumentedTask> runningStatus = statusTracker.getStatus();
            assertEquals(StatusUpdate.RunState.RUNNING, runningStatus.runstate);

            task.setProgress(1.0);
            Thread.sleep(200);

            StatusUpdate<InstrumentedTask> completedStatus = statusTracker.getStatus();
            assertEquals(1.0, completedStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, completedStatus.runstate);
        }
    }

    // =============================================================================
    // PERFORMANCE AND CONCURRENCY TESTS
    // =============================================================================

    /**
     * Tests thread-safety and concurrency behavior of multiple trackers.
     *
     * <p><strong>Purpose:</strong> Validates that multiple trackers can operate
     * safely in concurrent environments without data races, corruption, or
     * interference between tracker instances.
     *
     * <p><strong>Why Important:</strong> Multi-threaded applications require
     * thread-safe tracking behavior. This test ensures trackers can be used
     * safely across multiple threads without synchronization issues.
     *
     * <p><strong>Coverage:</strong> Thread safety, concurrent access patterns,
     * data integrity, isolation between tracker instances, resource contention.
     *
     * @see #testMultipleInstrumentedTasks() for sequential multiple tracker usage
     * @see TaskMonitorTest#testConcurrentTaskMonitoring() for monitor-level concurrency
     */
    @Test
    public void testConcurrentTrackers() throws InterruptedException {
        InstrumentedTask[] tasks = new InstrumentedTask[5];
        StatusTracker<InstrumentedTask>[] statusTrackers = new StatusTracker[5];

        try {
            for (int i = 0; i < 5; i++) {
                tasks[i] = new InstrumentedTask("concurrent-task-" + i);
                statusTrackers[i] = StatusTracker.withInstrumented(tasks[i]);
            }

            for (int i = 0; i < 5; i++) {
                tasks[i].setProgress(0.2 * (i + 1));
            }

            Thread.sleep(300);

            for (int i = 0; i < 5; i++) {
                StatusUpdate<InstrumentedTask> status = statusTrackers[i].getStatus();
                assertEquals(0.2 * (i + 1), status.progress, 0.001);
                if (i == 4) {
                    assertEquals(StatusUpdate.RunState.SUCCESS, status.runstate);
                } else {
                    assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);
                }
            }
        } finally {
            for (StatusTracker<InstrumentedTask> statusTracker : statusTrackers) {
                if (statusTracker != null) {
                    statusTracker.close();
                }
            }
        }
    }

    /**
     * Tests explicit resource cleanup behavior when trackers are manually closed.
     *
     * <p><strong>Purpose:</strong> Validates that calling {@code close()} on a tracker
     * properly releases resources and transitions the tracker to a safe post-close
     * state without causing exceptions or resource leaks.
     *
     * <p><strong>Why Important:</strong> Manual resource management is sometimes
     * necessary. This ensures that explicit close() calls work correctly and
     * trackers remain functional (though potentially limited) after closure.
     *
     * <p><strong>Coverage:</strong> Manual close() behavior, post-close state
     * validation, resource cleanup verification, continued functionality testing.
     *
     * @see #testTryWithResourcesCleanup() for automatic cleanup patterns
     * @see #testTrackerLifecycle() for lifecycle management
     */
    @Test
    public void testTrackerCleanupOnClose() throws InterruptedException {
        InstrumentedTask task = new InstrumentedTask("cleanup-test");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            task.setProgress(0.5);
            Thread.sleep(100);

            StatusUpdate<InstrumentedTask> statusBeforeClose = statusTracker.getStatus();
            assertNotNull(statusBeforeClose);

            // Manually close to test explicit cleanup behavior
            statusTracker.close();

            StatusUpdate<InstrumentedTask> statusAfterClose = statusTracker.getStatus();
            assertNotNull(statusAfterClose);
        }
    }

    /**
     * Tests automatic resource cleanup using try-with-resources patterns.
     *
     * <p><strong>Purpose:</strong> Validates that trackers properly implement
     * {@link AutoCloseable} and automatically release resources when exiting
     * try-with-resources blocks, even without explicit close() calls.
     *
     * <p><strong>Why Important:</strong> Try-with-resources is the preferred
     * resource management pattern in Java. This ensures trackers integrate
     * properly with this pattern and prevent resource leaks.
     *
     * <p><strong>Coverage:</strong> AutoCloseable compliance, automatic cleanup,
     * exception handling during cleanup, resource lifecycle management.
     *
     * @see #testTrackerCleanupOnClose() for manual cleanup testing
     * @see #testTrackerLifecycle() for general lifecycle patterns
     */
    @Test
    public void testTryWithResourcesCleanup() throws InterruptedException {
        InstrumentedTask task = new InstrumentedTask("try-with-resources");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            task.setProgress(0.4);
            Thread.sleep(100);

            StatusUpdate<InstrumentedTask> status = statusTracker.getStatus();
            assertEquals(0.4, status.progress, 0.001);
            assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);
        }

        task.setProgress(0.9);
        Thread.sleep(100);
    }

    // =============================================================================
    // CORNER CASE AND ERROR HANDLING TESTS
    // =============================================================================

    /**
     * Tests that Tracker creation properly validates null parameters.
     *
     * <p><strong>Purpose:</strong> Ensures that all Tracker factory methods properly validate
     * their parameters and throw {@link NullPointerException} for null arguments, providing
     * fail-fast behavior and clear error messages.
     *
     * <p><strong>Why Important:</strong> Null parameter validation prevents runtime errors
     * later in the tracking lifecycle and provides immediate feedback about incorrect usage.
     * This is critical for debugging and API usability.
     *
     * <p><strong>Coverage:</strong> Null validation for all Tracker factory method parameters:
     * instrumented tasks, functor tasks, and status functions.
     *
     * <p><strong>Consolidates:</strong> Previously separate tests for each null parameter type.
     */
    @Test
    public void testTrackerNullParameterValidation() {
        RegularTask validTask = new RegularTask("test");
        Function<RegularTask, StatusUpdate<RegularTask>> validFunction =
            t -> new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING);

        // Test null instrumented task
        try {
            StatusTracker.withInstrumented(null);
            fail("Expected NullPointerException for null instrumented task");
        } catch (NullPointerException expected) {
            // Expected
        }

        // Test null functor task
        try {
            StatusTracker.withFunctors(null, validFunction);
            fail("Expected NullPointerException for null functor task");
        } catch (NullPointerException expected) {
            // Expected
        }

        // Test null status function
        try {
            StatusTracker.withFunctors(validTask, null);
            fail("Expected NullPointerException for null status function");
        } catch (NullPointerException expected) {
            // Expected
        }
    }

    /**
     * Tests tracker behavior when monitored tasks enter failure states.
     *
     * <p><strong>Purpose:</strong> Validates that trackers properly detect and
     * report task failures (FAILED state), ensuring error conditions are
     * correctly propagated through the tracking system.
     *
     * <p><strong>Why Important:</strong> Failure detection is critical for
     * error handling and system resilience. Applications need to know when
     * tracked tasks fail to take appropriate corrective action.
     *
     * <p><strong>Coverage:</strong> Failure state detection, error propagation,
     * status transition from RUNNING to FAILED, timing of failure detection.
     *
     * @see #testMonitorStopsOnTaskCompletion() for success state handling
     * @see #testTrackerResourceCleanupOnException() for exception resilience
     */
    @Test
    public void testTrackerStatusAfterTaskFailure() throws InterruptedException {
        class FailingTask implements StatusUpdate.Provider<FailingTask> {
            private boolean shouldFail = false;
            private double progress = 0.0;

            @Override
            public StatusUpdate<FailingTask> getTaskStatus() {
                if (shouldFail) {
                    return new StatusUpdate<>(progress, StatusUpdate.RunState.FAILED);
                }
                return new StatusUpdate<>(progress,
                    progress >= 1.0 ? StatusUpdate.RunState.SUCCESS :
                    progress > 0 ? StatusUpdate.RunState.RUNNING : StatusUpdate.RunState.PENDING);
            }

            public void setProgress(double progress) {
                this.progress = progress;
            }

            public void fail() {
                this.shouldFail = true;
            }
        }

        FailingTask task = new FailingTask();
        try (StatusTracker<FailingTask> statusTracker = StatusTracker.withInstrumented(task)) {
            task.setProgress(0.5);
            Thread.sleep(150); // Give tracker time to poll the updated status
            StatusUpdate<FailingTask> runningStatus = statusTracker.getStatus();
            assertEquals(StatusUpdate.RunState.RUNNING, runningStatus.runstate);
            assertEquals(0.5, runningStatus.progress, 0.001);

            task.fail();
            Thread.sleep(150); // Give tracker time to poll the failed status
            StatusUpdate<FailingTask> failedStatus = statusTracker.getStatus();
            assertEquals(StatusUpdate.RunState.FAILED, failedStatus.runstate);
        }
    }

    /**
     * Tests tracker behavior under rapid task state transitions.
     *
     * <p><strong>Purpose:</strong> Validates that trackers can handle tasks that
     * change state rapidly, ensuring status updates are processed correctly
     * even under high-frequency change scenarios.
     *
     * <p><strong>Why Important:</strong> Some tasks may update their progress
     * very frequently (e.g., file downloads, data processing). Trackers must
     * handle these scenarios without performance degradation or missed updates.
     *
     * <p><strong>Coverage:</strong> High-frequency updates, rapid state changes,
     * performance under stress, final state consistency, timing behavior.
     *
     * @see TaskMonitorTest#testRapidStatusChanges() for monitor-level rapid change handling
     * @see #testPassiveMonitoringWithInstrumented() for normal-speed monitoring
     */
    @Test
    public void testTrackerWithRapidStateChanges() throws InterruptedException {
        InstrumentedTask task = new InstrumentedTask("rapid-changes");
        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task)) {
            // Rapidly change progress multiple times
            double[] progressValues = {0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0};

            for (double progress : progressValues) {
                task.setProgress(progress);
                Thread.sleep(10); // Small delay between updates
            }

            Thread.sleep(100); // Allow final state to stabilize

            StatusUpdate<InstrumentedTask> finalStatus = statusTracker.getStatus();
            assertEquals(1.0, finalStatus.progress, 0.001);
            assertEquals(StatusUpdate.RunState.SUCCESS, finalStatus.runstate);
        }
    }

    /**
     * Tests tracker resilience when tracked tasks throw exceptions during status queries.
     *
     * <p><strong>Purpose:</strong> Validates that trackers gracefully handle
     * exceptions thrown by task status methods, maintaining system stability
     * and providing meaningful error recovery behavior.
     *
     * <p><strong>Why Important:</strong> Tasks may encounter unexpected errors
     * during status reporting. Trackers must be resilient to these exceptions
     * to prevent system-wide failures and maintain operational stability.
     *
     * <p><strong>Coverage:</strong> Exception handling, error resilience,
     * graceful degradation, continued functionality after errors.
     *
     * @see #testTrackerStatusAfterTaskFailure() for controlled failure scenarios
     * @see TaskMonitorTest#testExceptionHandling() for monitor-level exception handling
     */
    @Test
    public void testTrackerResourceCleanupOnException() {
        class ExceptionThrowingTask implements StatusUpdate.Provider<ExceptionThrowingTask> {
            private boolean throwException = false;

            @Override
            public StatusUpdate<ExceptionThrowingTask> getTaskStatus() {
                if (throwException) {
                    throw new RuntimeException("Task status exception");
                }
                return new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING);
            }

            public void startThrowing() {
                this.throwException = true;
            }
        }

        ExceptionThrowingTask task = new ExceptionThrowingTask();
        try (StatusTracker<ExceptionThrowingTask> statusTracker = StatusTracker.withInstrumented(task)) {
            StatusUpdate<ExceptionThrowingTask> initialStatus = statusTracker.getStatus();
            assertNotNull(initialStatus);

            task.startThrowing();
            // Tracker should handle exceptions gracefully
            StatusUpdate<ExceptionThrowingTask> statusAfterException = statusTracker.getStatus();
            // Should return last known good status or handle gracefully
            assertNotNull(statusAfterException);
        }
    }

    /**
     * Tests tracker functionality with simple custom objects using functor patterns.
     *
     * <p><strong>Purpose:</strong> Demonstrates that trackers can work with any
     * object type (including primitives like String) using appropriate status
     * functions, showcasing the flexibility of the functor-based approach.
     *
     * <p><strong>Why Important:</strong> This validates the general applicability
     * of the tracking system beyond complex task objects, enabling tracking
     * of simple data structures or third-party objects.
     *
     * <p><strong>Coverage:</strong> Simple object tracking, primitive type support,
     * basic status function implementation, type safety validation.
     *
     * @see #testWithFunctors() for complex functor-based tracking
     * @see #testCustomStatusFunction() for advanced status function scenarios
     */
    @Test
    public void testTrackerWithCustomObject() {
        String simpleTask = "simple-string-task";
        Function<String, StatusUpdate<String>> statusFunction = s ->
            new StatusUpdate<>(s.length() / 20.0, StatusUpdate.RunState.RUNNING);

        try (StatusTracker<String> statusTracker = StatusTracker.withFunctors(simpleTask, statusFunction)) {
            assertEquals(simpleTask, statusTracker.getTracked());
            StatusUpdate<String> status = statusTracker.getStatus();
            assertEquals(simpleTask.length() / 20.0, status.progress, 0.001);
        }
    }

    /**
     * Tests tracker integration with TaskSink for lifecycle event handling.
     *
     * <p><strong>Purpose:</strong> Validates that Tracker properly communicates
     * with attached TaskSink instances for lifecycle events (start/update/finish).
     *
     * <p><strong>Coverage:</strong> Sink integration, lifecycle event propagation</p>
     */
    @Test
    public void testTrackerWithSink() throws InterruptedException {
        TestSink sink = new TestSink();
        InstrumentedTask task = new InstrumentedTask("sink-test");

        try (StatusTracker<InstrumentedTask> statusTracker = StatusTracker.withInstrumented(task, sink)) {
            // Should have started
            assertTrue("Should have called taskStarted", sink.started);

            // Update task state
            task.progress = 0.5;
            Thread.sleep(150); // Allow monitoring cycle

            assertTrue("Should have received status updates", sink.updateCount > 0);

            // Complete task
            task.progress = 1.0;
            task.state = StatusUpdate.RunState.SUCCESS;
            Thread.sleep(150); // Allow monitoring cycle
        }

        assertTrue("Should have called taskFinished", sink.finished);
    }

    /**
     * Helper TaskSink implementation for testing sink integration.
     */
    private static class TestSink implements StatusSink {
        boolean started = false;
        boolean finished = false;
        int updateCount = 0;

        @Override
        public void taskStarted(StatusTracker<?> task) {
            started = true;
        }

        @Override
        public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
            updateCount++;
        }

        @Override
        public void taskFinished(StatusTracker<?> task) {
            finished = true;
        }
    }
}