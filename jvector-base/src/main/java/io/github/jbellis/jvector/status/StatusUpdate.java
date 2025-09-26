package io.github.jbellis.jvector.status;

/**
 * Represents the current status of a tracked task at a specific point in time.
 * This immutable snapshot contains the task's progress percentage, execution state,
 * timestamp when the status was created, and optionally a reference to the tracked object.
 *
 * <p>StatusUpdate objects are created by monitoring systems to report task progress
 * and are passed to {@link StatusSink} implementations for processing. Each status
 * represents a moment-in-time view of task execution and should be treated as immutable.
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Creating Status Updates</h3>
 * <pre>{@code
 * // For tasks implementing TaskStatus.Provider
 * public class DataProcessor implements TaskStatus.Provider<DataProcessor> {
 *     private volatile double progress = 0.0;
 *     private volatile RunState state = RunState.PENDING;
 *
 *     @Override
 *     public TaskStatus<DataProcessor> getTaskStatus() {
 *         return new TaskStatus<>(progress, state);
 *     }
 *
 *     public void processData() {
 *         state = RunState.RUNNING;
 *         for (int i = 0; i < totalItems; i++) {
 *             processItem(i);
 *             progress = (double) i / totalItems;
 *         }
 *         state = RunState.SUCCESS;
 *         progress = 1.0;
 *     }
 * }
 * }</pre>
 *
 * <h3>Custom Status Functions</h3>
 * <pre>{@code
 * // For tasks that don't implement the Provider interface
 * Function<BatchJob, TaskStatus<BatchJob>> statusFunction = job -> {
 *     RunState state;
 *     if (job.isComplete()) {
 *         state = job.hasErrors() ? RunState.FAILED : RunState.SUCCESS;
 *     } else if (job.isStarted()) {
 *         state = RunState.RUNNING;
 *     } else {
 *         state = RunState.PENDING;
 *     }
 *
 *     return new TaskStatus<>(job.getCompletionRatio(), state);
 * };
 * }</pre>
 *
 * <h3>Status Processing in Sinks</h3>
 * <pre>{@code
 * public class CustomSink implements TaskSink {
 *     @Override
 *     public void taskUpdate(Tracker<?> task, TaskStatus<?> status) {
 *         switch (status.runstate) {
 *             case RUNNING:
 *                 if (status.progress > 0.5) {
 *                     logger.info("Task {} is halfway complete", getTaskName(task));
 *                 }
 *                 break;
 *             case FAILED:
 *                 logger.error("Task {} failed at {}% completion",
 *                     getTaskName(task), status.progress * 100);
 *                 break;
 *             case SUCCESS:
 *                 logger.info("Task {} completed successfully", getTaskName(task));
 *                 break;
 *         }
 *     }
 * }
 * }</pre>
 *
 * <h2>Progress Values</h2>
 * <p>Progress is represented as a double value between 0.0 and 1.0:</p>
 * <ul>
 *   <li><strong>0.0:</strong> Task not started or 0% complete</li>
 *   <li><strong>0.5:</strong> Task is 50% complete</li>
 *   <li><strong>1.0:</strong> Task is 100% complete</li>
 * </ul>
 *
 * <p>Progress values should be monotonically increasing during normal execution,
 * though this is not enforced by the framework.</p>
 *
 * <h2>RunState Values</h2>
 * <p>The {@link RunState} enum represents the execution phase:</p>
 * <ul>
 *   <li><strong>PENDING:</strong> Task is queued but not yet started</li>
 *   <li><strong>RUNNING:</strong> Task is actively executing</li>
 *   <li><strong>SUCCESS:</strong> Task completed successfully</li>
 *   <li><strong>FAILED:</strong> Task completed with errors</li>
 *   <li><strong>CANCELLED:</strong> Task was cancelled before completion</li>
 * </ul>
 *
 * <h2>Provider Interface</h2>
 * <p>Objects that can provide their own status should implement the
 * {@link Provider} interface. This enables automatic status polling
 * without requiring custom status functions:</p>
 *
 * <pre>{@code
 * public class InstrumentedTask implements TaskStatus.Provider<InstrumentedTask> {
 *     @Override
 *     public TaskStatus<InstrumentedTask> getTaskStatus() {
 *         return new TaskStatus<>(getCurrentProgress(), getCurrentState());
 *     }
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>TaskStatus objects are immutable after construction and are thread-safe.
 * However, the underlying task objects that provide status may require their
 * own synchronization for thread-safe access to progress and state fields.</p>
 *
 * @param <T> the type of task being tracked
 * @see StatusTracker
 * @see StatusSink
 * @see RunState
 * @since 4.0.0
 */
public class StatusUpdate<T> {
    public final double progress;
    public final RunState runstate;
    public final long timestamp;
    public final T tracked;

    /**
     * Creates a StatusUpdate with the current timestamp and no tracked object reference.
     *
     * @param progress the task's progress (0.0 to 1.0)
     * @param runstate the task's current execution state
     */
    public StatusUpdate(double progress, RunState runstate) {
        this(progress, runstate, null);
    }

    /**
     * Creates a StatusUpdate with the current timestamp and a reference to the tracked object.
     *
     * @param progress the task's progress (0.0 to 1.0)
     * @param runstate the task's current execution state
     * @param tracked the object being tracked (may be null)
     */
    public StatusUpdate(double progress, RunState runstate, T tracked) {
        this.progress = progress;
        this.runstate = runstate;
        this.timestamp = System.currentTimeMillis();
        this.tracked = tracked;
    }

    public enum RunState {
        PENDING("⏳"),
        RUNNING("🔄"),
        SUCCESS("✅"),
        FAILED("❌"),
        CANCELLED("🚫");

        private final String glyph;

        RunState(String glyph) {
            this.glyph = glyph;
        }
    }

    /**
     * Interface for objects that can provide their own status updates.
     * Implementations should return a StatusUpdate that includes a reference
     * to themselves as the tracked object.
     *
     * @param <T> the type of the object providing status
     */
    public interface Provider<T extends Provider<T>> {
        /**
         * Gets the current status of this task.
         * Implementations should typically include themselves as the tracked object:
         * {@code return new StatusUpdate<>(progress, state, this);}
         *
         * @return the current status with tracked object reference
         */
        StatusUpdate<T> getTaskStatus();
    }
}
