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

package io.github.jbellis.jvector.status.eventing;

/**
 * Represents the execution state of a tracked task. This enum defines the standard
 * lifecycle states that tasks transition through during execution.
 *
 * <p>State Transitions:
 * <ul>
 *   <li><strong>PENDING → RUNNING:</strong> Task begins execution</li>
 *   <li><strong>RUNNING → SUCCESS:</strong> Task completes successfully</li>
 *   <li><strong>RUNNING → FAILED:</strong> Task encounters an error</li>
 *   <li><strong>RUNNING → CANCELLED:</strong> Task is explicitly cancelled</li>
 *   <li><strong>PENDING → CANCELLED:</strong> Task is cancelled before starting</li>
 * </ul>
 *
 * <p>Terminal States: {@link #SUCCESS}, {@link #FAILED}, and {@link #CANCELLED} are
 * terminal states - tasks should not transition from these states to any other state.
 *
 * <p>Usage in Status Updates:
 * <pre>{@code
 * // Task starts
 * return new StatusUpdate<>(0.0, RunState.PENDING, this);
 *
 * // Task begins execution
 * return new StatusUpdate<>(0.0, RunState.RUNNING, this);
 *
 * // Task progresses
 * return new StatusUpdate<>(0.5, RunState.RUNNING, this);
 *
 * // Task completes
 * return new StatusUpdate<>(1.0, RunState.SUCCESS, this);
 * }</pre>
 *
 * @see StatusUpdate
 * @see StatusSource
 * @since 4.0.0
 */
public enum RunState {
    /**
     * Task is queued or waiting to start execution.
     * Progress is typically 0.0 in this state.
     */
    PENDING("⏳"),

    /**
     * Task is actively executing.
     * Progress typically increases from 0.0 towards 1.0.
     */
    RUNNING("🔄"),

    /**
     * Task completed successfully (terminal state).
     * Progress should be 1.0 in this state.
     */
    SUCCESS("✅"),

    /**
     * Task failed due to an error (terminal state).
     * Progress may be any value depending on when the failure occurred.
     */
    FAILED("❌"),

    /**
     * Task was explicitly cancelled (terminal state).
     * Progress may be any value depending on when cancellation occurred.
     */
    CANCELLED("🚫");

    private final String glyph;

    RunState(String glyph) {
        this.glyph = glyph;
    }

    /**
     * Returns the Unicode glyph associated with this state for display purposes.
     *
     * @return a Unicode emoji character representing this state
     */
    public String getGlyph() {
        return glyph;
    }
}
