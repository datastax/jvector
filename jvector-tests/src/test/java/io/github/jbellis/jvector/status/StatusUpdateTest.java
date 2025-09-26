package io.github.jbellis.jvector.status;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.status.sinks.StatusSinkTest;
import org.junit.Test;
import org.junit.jupiter.api.Tag;

import static org.junit.Assert.*;

/**
 * Core functionality tests for TaskStatus and related status tracking components.
 *
 * <p>This test suite covers the core task status data structures and their behavior:
 * <ul>
 *   <li><strong>TaskStatus:</strong> Progress tracking, state management, timestamp handling</li>
 *   <li><strong>TestableTask:</strong> Task implementation for testing scenarios</li>
 *   <li><strong>TaskStatus.Provider:</strong> Interface implementation patterns</li>
 *   <li><strong>Integration:</strong> Usage patterns with tracking framework</li>
 * </ul>
 *
 * <h2>Test Coverage Areas:</h2>
 * <ul>
 *   <li>TaskStatus creation and immutability</li>
 *   <li>Progress value validation and constraints</li>
 *   <li>RunState transitions and behavior</li>
 *   <li>Timestamp accuracy and ordering</li>
 *   <li>TaskStatus.Provider interface compliance</li>
 *   <li>TestableTask functionality and edge cases</li>
 * </ul>
 *
 * <h2>Related Test Classes:</h2>
 * <ul>
 *   <li>{@link StatusTrackerTest} - Higher-level Tracker wrapper functionality</li>
 *   <li>{@link TaskMonitorTest} - Background monitoring and polling</li>
 *   <li>{@link StatusTrackingIntegrationTest} - End-to-end integration scenarios</li>
 *   <li>{@link StatusSinkTest} - TaskSink implementations</li>
 * </ul>
 *
 * @see StatusUpdate
 * @see TestableTask
 * @since 4.0.0
 */
@Tag("Core")
public class StatusUpdateTest extends RandomizedTest {

    /**
     * Tests basic TaskStatus creation and field access.
     *
     * <p><strong>Purpose:</strong> Validates that TaskStatus correctly captures and stores
     * progress, run state, and timestamp information during construction.</p>
     *
     * <p><strong>Why Important:</strong> TaskStatus is the fundamental data structure
     * for all status reporting in the framework.</p>
     *
     * <p><strong>Coverage:</strong> Basic object creation, field access, immutability</p>
     */
    @Test
    public void testTaskStatusBasicCreation() {
        long beforeTime = System.currentTimeMillis();
        StatusUpdate<String> status = new StatusUpdate<>(0.75, StatusUpdate.RunState.RUNNING);
        long afterTime = System.currentTimeMillis();

        assertEquals(0.75, status.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);
        assertTrue("Timestamp should be between before and after times",
                status.timestamp >= beforeTime && status.timestamp <= afterTime);
    }

    /**
     * Tests TaskStatus timestamp ordering and accuracy.
     *
     * <p><strong>Purpose:</strong> Validates that TaskStatus instances capture
     * accurate timestamps that reflect their creation order.</p>
     *
     * <p><strong>Why Important:</strong> Timestamp ordering is crucial for
     * monitoring systems that track progress over time.</p>
     *
     * <p><strong>Coverage:</strong> Timestamp accuracy, temporal ordering</p>
     */
    @Test
    public void testTaskStatusTimestampOrdering() throws InterruptedException {
        StatusUpdate<String> status1 = new StatusUpdate<>(0.25, StatusUpdate.RunState.RUNNING);

        // Small delay to ensure different timestamps
        Thread.sleep(1);

        StatusUpdate<String> status2 = new StatusUpdate<>(0.50, StatusUpdate.RunState.RUNNING);

        Thread.sleep(1);

        StatusUpdate<String> status3 = new StatusUpdate<>(0.75, StatusUpdate.RunState.SUCCESS);

        assertTrue("First status should have earlier timestamp",
                status1.timestamp < status2.timestamp);
        assertTrue("Second status should have earlier timestamp than third",
                status2.timestamp < status3.timestamp);
        assertTrue("All timestamps should be reasonable",
                status1.timestamp > 0 && status2.timestamp > 0 && status3.timestamp > 0);
    }

    /**
     * Tests RunState enum values and glyph representations.
     *
     * <p><strong>Purpose:</strong> Validates that all RunState enum values
     * exist and have appropriate string representations.</p>
     *
     * <p><strong>Coverage:</strong> Enum completeness, string representations</p>
     */
    @Test
    public void testRunStateEnum() {
        // Test all enum values exist
        StatusUpdate.RunState[] states = StatusUpdate.RunState.values();
        assertEquals("Should have exactly 5 run states", 5, states.length);

        // Test specific states exist
        assertNotNull(StatusUpdate.RunState.PENDING);
        assertNotNull(StatusUpdate.RunState.RUNNING);
        assertNotNull(StatusUpdate.RunState.SUCCESS);
        assertNotNull(StatusUpdate.RunState.FAILED);
        assertNotNull(StatusUpdate.RunState.CANCELLED);

        // Test states can be used in TaskStatus
        for (StatusUpdate.RunState state : states) {
            StatusUpdate<String> status = new StatusUpdate<>(0.5, state);
            assertEquals(state, status.runstate);
        }
    }

    /**
     * Tests TestableTask basic functionality and lifecycle.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask correctly implements
     * TaskStatus.Provider and provides expected task behavior.</p>
     *
     * <p><strong>Coverage:</strong> TestableTask creation, progress tracking, state management</p>
     */
    @Test
    public void testTestableTaskBasicFunctionality() {
        TestableTask task = new TestableTask("test-task");

        // Test initial state
        assertEquals("test-task", task.getName());
        assertEquals(0.0, task.getProgress(), 0.0001);
        assertEquals(StatusUpdate.RunState.PENDING, task.getState());

        // Test progress setting
        task.setProgress(0.5);
        assertEquals(0.5, task.getProgress(), 0.0001);
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());

        // Test completion
        task.complete();
        assertEquals(1.0, task.getProgress(), 0.0001);
        assertEquals(StatusUpdate.RunState.SUCCESS, task.getState());
    }

    /**
     * Tests TestableTask TaskStatus.Provider implementation.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask properly implements
     * the TaskStatus.Provider interface and returns consistent status objects.</p>
     *
     * <p><strong>Coverage:</strong> TaskStatus.Provider implementation, status consistency</p>
     */
    @Test
    public void testTestableTaskStatusProvider() throws InterruptedException {
        TestableTask task = new TestableTask("provider-test");

        // Test initial status
        StatusUpdate<TestableTask> status1 = task.getTaskStatus();
        assertEquals(0.0, status1.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.PENDING, status1.runstate);

        // Change task state and test status update
        task.setProgress(0.75);

        // Small delay to ensure different timestamps
        Thread.sleep(1);

        StatusUpdate<TestableTask> status2 = task.getTaskStatus();
        assertEquals(0.75, status2.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.RUNNING, status2.runstate);

        // Verify timestamps are different
        assertTrue("Status timestamps should be different",
                status2.timestamp > status1.timestamp);
    }

    /**
     * Tests TestableTask lifecycle state transitions.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask correctly handles
     * state transitions through its lifecycle methods.</p>
     *
     * <p><strong>Coverage:</strong> Lifecycle methods, state transitions</p>
     */
    @Test
    public void testTestableTaskLifecycle() {
        TestableTask task = new TestableTask("lifecycle-test");

        // Test start
        task.start();
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());
        assertEquals(0.0, task.getProgress(), 0.0001);

        // Test manual progress
        task.setProgress(0.3);
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());

        // Test failure
        task.fail();
        assertEquals(StatusUpdate.RunState.FAILED, task.getState());
        assertEquals(0.3, task.getProgress(), 0.0001); // Progress unchanged

        // Test reset and complete
        TestableTask task2 = new TestableTask("lifecycle-test-2");
        task2.complete();
        assertEquals(StatusUpdate.RunState.SUCCESS, task2.getState());
        assertEquals(1.0, task2.getProgress(), 0.0001);
    }

    /**
     * Tests TestableTask input validation and error handling.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask properly validates
     * inputs and throws appropriate exceptions for invalid values.</p>
     *
     * <p><strong>Coverage:</strong> Input validation, exception handling</p>
     */
    @Test
    public void testTestableTaskValidation() {
        // Test invalid task name
        try {
            new TestableTask(null);
            fail("Should throw exception for null name");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Task name cannot be null"));
        }

        try {
            new TestableTask("");
            fail("Should throw exception for empty name");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Task name cannot be null"));
        }

        // Test invalid progress values
        TestableTask task = new TestableTask("validation-test");

        try {
            task.setProgress(-0.1);
            fail("Should throw exception for negative progress");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Progress must be between"));
        }

        try {
            task.setProgress(1.1);
            fail("Should throw exception for progress > 1.0");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Progress must be between"));
        }

        try {
            task.setProgress(Double.NaN);
            fail("Should throw exception for NaN progress");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Progress must be between"));
        }

        // Test invalid state
        try {
            task.setState(null);
            fail("Should throw exception for null state");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("State cannot be null"));
        }
    }

    /**
     * Tests TestableTask equality and hash code behavior.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask correctly implements
     * equality based on task name only.</p>
     *
     * <p><strong>Coverage:</strong> Equals/hashCode contract, name-based equality</p>
     */
    @Test
    public void testTestableTaskEquality() {
        TestableTask task1 = new TestableTask("same-name");
        TestableTask task2 = new TestableTask("same-name");
        TestableTask task3 = new TestableTask("different-name");

        // Test equality
        assertEquals(task1, task2);
        assertNotEquals(task1, task3);
        assertNotEquals(task1, null);
        assertNotEquals(task1, "not-a-task");

        // Test hash code consistency
        assertEquals(task1.hashCode(), task2.hashCode());

        // Test that progress/state don't affect equality
        task1.setProgress(0.5);
        task2.setProgress(0.8);
        assertEquals("Tasks should be equal despite different progress", task1, task2);
    }

    /**
     * Tests TestableTask toString representation.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask provides meaningful
     * string representations for debugging and logging.</p>
     *
     * <p><strong>Coverage:</strong> String representation, debugging output</p>
     */
    @Test
    public void testTestableTaskToString() {
        TestableTask task = new TestableTask("toString-test");
        task.setProgress(0.42);
        task.setState(StatusUpdate.RunState.RUNNING);

        String str = task.toString();
        assertTrue("Should contain task name", str.contains("toString-test"));
        assertTrue("Should contain progress", str.contains("0.42"));
        assertTrue("Should contain state", str.contains("RUNNING"));
        assertTrue("Should be TestableTask format", str.startsWith("TestableTask{"));
    }

    /**
     * Tests StatusUpdate with tracked object reference.
     *
     * <p><strong>Purpose:</strong> Validates that StatusUpdate correctly stores
     * and provides access to the tracked object reference when provided.</p>
     *
     * <p><strong>Coverage:</strong> Tracked field initialization, null handling</p>
     */
    @Test
    public void testStatusUpdateWithTrackedObject() {
        TestableTask task = new TestableTask("tracked-test");

        // Test with tracked object (3-arg constructor)
        StatusUpdate<TestableTask> statusWithTracked = new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING, task);
        assertEquals(0.5, statusWithTracked.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.RUNNING, statusWithTracked.runstate);
        assertNotNull(statusWithTracked.tracked);
        assertEquals(task, statusWithTracked.tracked);
        assertEquals("tracked-test", statusWithTracked.tracked.getName());

        // Test without tracked object (2-arg constructor)
        StatusUpdate<TestableTask> statusWithoutTracked = new StatusUpdate<>(0.75, StatusUpdate.RunState.SUCCESS);
        assertEquals(0.75, statusWithoutTracked.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.SUCCESS, statusWithoutTracked.runstate);
        assertNull(statusWithoutTracked.tracked);

        // Test that TestableTask's getTaskStatus includes tracked reference
        task.setProgress(0.8);
        StatusUpdate<TestableTask> taskStatus = task.getTaskStatus();
        assertEquals(0.8, taskStatus.progress, 0.0001);
        assertEquals(StatusUpdate.RunState.RUNNING, taskStatus.runstate);
        assertNotNull(taskStatus.tracked);
        assertEquals(task, taskStatus.tracked);
    }

    /**
     * Tests that Provider implementations return StatusUpdate with tracked reference.
     */
    @Test
    public void testProviderTrackedReference() {
        TestableTask task = new TestableTask("provider-tracked-test");

        // Test PENDING state
        StatusUpdate<TestableTask> pendingStatus = task.getTaskStatus();
        assertSame("Provider should return self as tracked", task, pendingStatus.tracked);

        // Test RUNNING state
        task.setProgress(0.5);
        StatusUpdate<TestableTask> runningStatus = task.getTaskStatus();
        assertSame("Provider should return self as tracked", task, runningStatus.tracked);

        // Test SUCCESS state
        task.complete();
        StatusUpdate<TestableTask> successStatus = task.getTaskStatus();
        assertSame("Provider should return self as tracked", task, successStatus.tracked);
        assertEquals(1.0, successStatus.progress, 0.0001);
    }

    /**
     * Tests progress auto-state transitions in TestableTask.
     *
     * <p><strong>Purpose:</strong> Validates that TestableTask automatically
     * updates its run state based on progress value changes.</p>
     *
     * <p><strong>Coverage:</strong> Auto-state transitions, progress-based state logic</p>
     */
    @Test
    public void testTestableTaskAutoStateTransitions() {
        TestableTask task = new TestableTask("auto-state-test");

        // Initial state should be PENDING
        assertEquals(StatusUpdate.RunState.PENDING, task.getState());

        // Setting progress > 0 and < 1 should go to RUNNING
        task.setProgress(0.1);
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());

        task.setProgress(0.5);
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());

        task.setProgress(0.99);
        assertEquals(StatusUpdate.RunState.RUNNING, task.getState());

        // Setting progress to 1.0 should go to SUCCESS
        task.setProgress(1.0);
        assertEquals(StatusUpdate.RunState.SUCCESS, task.getState());

        // Test that setting to 0 from PENDING stays PENDING
        TestableTask task2 = new TestableTask("auto-state-test-2");
        task2.setProgress(0.0);
        assertEquals(StatusUpdate.RunState.PENDING, task2.getState());
    }
}