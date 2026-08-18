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

package io.github.jbellis.jvector.util.work;

import io.github.jbellis.jvector.annotations.Experimental;

/**
 * Observation contract: observes the phases of a long-running operation. {@link #startPhase} is
 * the single entry point; progress is reported through the returned {@link PhaseScope}, so the
 * scope itself is the capability to report — progress for a phase that was never started is
 * unrepresentable, and per-phase implementation state (a progress bar, a long-task timer) lives
 * in the scope instance rather than in a map keyed by {@link WorkStage}. Concurrent phases, even
 * of the same stage, are distinguished by scope identity.
 *
 * <p>Best-effort and cheap: implementations <b>must not</b> throw from {@code startPhase},
 * {@link PhaseScope#onProgress}, or {@link PhaseScope#close} (the caller invokes these on its
 * orchestrating thread and treats them as fire-and-forget). The minimal consumer is a single
 * expression — {@code stage -> (completed, total) -> bar.update(stage, completed, total)} — since
 * {@code close()} defaults to a no-op. See {@link ProgressLimiter} for the melded progress +
 * throttle surface that most consumers accept.
 */
@Experimental
@FunctionalInterface
public interface ProgressTracker {
    /**
     * Starts one phase of the operation. The returned scope receives that phase's progress and
     * must be closed exactly once, normally with try-with-resources.
     *
     * @param stage identifies the phase's stage, in consumer-defined terms
     */
    PhaseScope startPhase(WorkStage stage);

    /**
     * One started phase: receives its progress reports and, on {@link #close}, its completion.
     * {@code close()} defaults to a no-op so a progress-only implementation stays a single lambda.
     */
    @FunctionalInterface
    interface PhaseScope extends AutoCloseable {
        /**
         * Reports progress for this phase.
         *
         * @param completed work done so far, in stage-defined units; monotonically non-decreasing
         *                  within the phase
         * @param total     total work for the phase, or {@code -1} if not yet known
         */
        void onProgress(long completed, long total);

        /** Marks the end of the phase. Called exactly once; defaults to a no-op. */
        @Override
        default void close() { }

        /** A scope that discards every update. */
        PhaseScope NOOP = (completed, total) -> { };
    }

    /** A tracker that discards every phase. */
    ProgressTracker NOOP = stage -> PhaseScope.NOOP;
}
