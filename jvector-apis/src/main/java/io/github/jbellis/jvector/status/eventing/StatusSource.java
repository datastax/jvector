package io.github.jbellis.jvector.status.eventing;

import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;

/**
 * Interface for objects that can provide their own status updates without requiring
 * a custom status function. Objects implementing this interface can be tracked directly
 * by passing them to {@link StatusContext#track(StatusSource)}
 * methods.
 *
 * <p>Usage Example:
 * <pre>{@code
 * public class MyTask implements StatusSource<MyTask> {
 *     private volatile double progress = 0.0;
 *     private volatile RunState state = RunState.PENDING;
 *
 *     @Override
 *     public StatusUpdate<MyTask> getTaskStatus() {
 *         return new StatusUpdate<>(progress, state, this);
 *     }
 *
 *     public void execute() {
 *         state = RunState.RUNNING;
 *         // ... do work, updating progress ...
 *         progress = 1.0;
 *         state = RunState.SUCCESS;
 *     }
 * }
 *
 * // Track the task
 * try (StatusContext context = new StatusContext("my-operation");
 *      StatusTracker<MyTask> tracker = context.track(myTask)) {
 *     myTask.execute();
 * }
 * }</pre>
 *
 * <p><strong>Thread Safety:</strong> Implementations should ensure that {@link #getTaskStatus()}
 * can be safely called concurrently from the monitoring thread while the task is executing.
 * Use appropriate synchronization or volatile fields for progress and state.
 *
 * @param <T> the type of the object providing status (must be the implementing class itself)
 * @see StatusUpdate
 * @see RunState
 * @see StatusContext#track(StatusSource)
 * @since 4.0.0
 */
public interface StatusSource<T extends StatusSource<T>> {
    /**
     * Returns the current status of this object. This method is called periodically by
     * the monitoring framework to observe the object's state.
     * <p>
     * Implementations should return a new {@link StatusUpdate} containing:
     * <ul>
     *   <li>Current progress value (0.0 to 1.0)</li>
     *   <li>Current {@link RunState}</li>
     *   <li>Reference to this object (typically {@code this})</li>
     * </ul>
     *
     * <p>Example implementation:
     * <pre>{@code
     * @Override
     * public StatusUpdate<MyTask> getTaskStatus() {
     *     return new StatusUpdate<>(currentProgress, currentState, this);
     * }
     * }</pre>
     *
     * @return the current status including progress, state, and tracked object reference
     */
    StatusUpdate<T> getTaskStatus();
}
