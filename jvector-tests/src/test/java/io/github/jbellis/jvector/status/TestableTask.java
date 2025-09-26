package io.github.jbellis.jvector.status;

import java.util.Objects;

/**
 * A testable task implementation that provides basic progress tracking functionality for testing.
 * This class serves as a test utility for validating the task tracking framework
 * without requiring complex domain objects or business logic.
 *
 * <p>This implementation provides:
 * <ul>
 *   <li>Named task identification</li>
 *   <li>Thread-safe progress tracking (0.0 to 1.0)</li>
 *   <li>TaskStatus.Provider implementation for use with Tracker</li>
 *   <li>Simple lifecycle management</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Basic Progress Tracking in Tests</h3>
 * <pre>{@code
 * TestableTask task = new TestableTask("data-processing");
 * try (Tracker<TestableTask> tracker = Tracker.withInstrumented(task, sinks)) {
 *     task.setProgress(0.25);  // 25% complete
 *     doSomeWork();
 *     task.setProgress(0.75);  // 75% complete
 *     doMoreWork();
 *     task.setProgress(1.0);   // Complete
 * }
 * }</pre>
 *
 * <h3>Integration with TrackerScope</h3>
 * <pre>{@code
 * TrackerScope scope = new TrackerScope("batch-processing");
 * TestableTask task = new TestableTask("file-processing");
 * try (Tracker<TestableTask> tracker = scope.track(task)) {
 *     // Task progress is automatically monitored and reported
 *     for (int i = 0; i < 100; i++) {
 *         processItem(i);
 *         task.setProgress(i / 100.0);
 *     }
 * }
 * }</pre>
 *
 * @since 4.0.0
 */
public final class TestableTask implements StatusUpdate.Provider<TestableTask> {
    private final String name;
    private volatile double progress = 0.0;
    private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;

    /**
     * Creates a new testable task with the specified name.
     *
     * @param name the name of the task
     * @throws IllegalArgumentException if name is null or empty
     */
    public TestableTask(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Task name cannot be null or empty");
        }
        this.name = name.trim();
    }

    /**
     * Gets the name of this task.
     *
     * @return the task name
     */
    public String getName() {
        return name;
    }

    /**
     * Gets the current progress of this task.
     *
     * @return progress value between 0.0 (0%) and 1.0 (100%)
     */
    public double getProgress() {
        return progress;
    }

    /**
     * Sets the current progress of this task.
     *
     * @param progress the progress value between 0.0 (0%) and 1.0 (100%)
     * @throws IllegalArgumentException if progress is not between 0.0 and 1.0, or is NaN/infinite
     */
    public void setProgress(double progress) {
        if (progress < 0.0 || progress > 1.0 || !Double.isFinite(progress)) {
            throw new IllegalArgumentException("Progress must be between 0.0 and 1.0, inclusive, got: " + progress);
        }
        this.progress = progress;

        // Auto-update state based on progress
        if (progress == 0.0 && state == StatusUpdate.RunState.PENDING) {
            // Stay in PENDING state
        } else if (progress > 0.0 && progress < 1.0) {
            state = StatusUpdate.RunState.RUNNING;
        } else if (progress == 1.0) {
            state = StatusUpdate.RunState.SUCCESS;
        }
    }

    /**
     * Gets the current run state of this task.
     *
     * @return the current state
     */
    public StatusUpdate.RunState getState() {
        return state;
    }

    /**
     * Sets the current run state of this task.
     *
     * @param state the new state
     * @throws IllegalArgumentException if state is null
     */
    public void setState(StatusUpdate.RunState state) {
        if (state == null) {
            throw new IllegalArgumentException("State cannot be null");
        }
        this.state = state;
    }

    /**
     * Marks this task as started (RUNNING state).
     */
    public void start() {
        this.state = StatusUpdate.RunState.RUNNING;
    }

    /**
     * Marks this task as completed (SUCCESS state and 100% progress).
     */
    public void complete() {
        this.progress = 1.0;
        this.state = StatusUpdate.RunState.SUCCESS;
    }

    /**
     * Marks this task as failed (FAILED state).
     */
    public void fail() {
        this.state = StatusUpdate.RunState.FAILED;
    }

    @Override
    public StatusUpdate<TestableTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    @Override
    public String toString() {
        return String.format("TestableTask{name='%s', progress=%.2f, state=%s}",
                           name, progress, state);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        TestableTask that = (TestableTask) obj;
        return Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }
}