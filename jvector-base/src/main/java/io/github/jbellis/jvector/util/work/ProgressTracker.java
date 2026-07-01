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
 * Observation contract: receives progress updates for a stage of a long-running operation.
 *
 * <p>Best-effort and cheap: implementations <b>must not</b> throw (the caller invokes this on its
 * orchestrating thread and treats it as fire-and-forget). See {@link ProgressLimiter} for the
 * melded progress + throttle surface that most consumers accept.
 */
@Experimental
@FunctionalInterface
public interface ProgressTracker {
    /**
     * Reports progress for {@code stage}.
     *
     * @param stage     the stage reporting progress
     * @param completed work done so far in this stage, in stage-defined units; monotonically
     *                  non-decreasing within a stage
     * @param total     total work for this stage, or {@code -1} if not yet known
     */
    void onProgress(WorkStage stage, long completed, long total);

    /** A tracker that discards every update. */
    ProgressTracker NOOP = (stage, completed, total) -> { };
}
