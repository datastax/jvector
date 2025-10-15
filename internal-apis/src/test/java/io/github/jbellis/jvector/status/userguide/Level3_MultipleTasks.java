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

package io.github.jbellis.jvector.status.userguide;

import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataValidator;

/**
 * Level 3: Track Multiple Tasks
 *
 * <p>Building on Level 2: Share a StatusContext across multiple tasks.
 *
 * <h2>Key Differences from Level 2:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> Explicitly create StatusContext (instead of auto-created)</li>
 *   <li><strong>NEW:</strong> Track multiple tasks with context.track()</li>
 *   <li>Share single monitoring thread across all tasks</li>
 *   <li>Share sinks across all tasks</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Context creation:</strong> Explicit with name "batch-processing"</li>
 *   <li><strong>Task tracking:</strong> Each task gets auto-created scope via context.track()</li>
 *   <li><strong>Resource management:</strong> Context closed explicitly in finally block</li>
 *   <li><strong>Scope per task:</strong> Each tracker gets "auto-scope-{taskName}"</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> Same polling overhead as Level 2, but amortized across tasks</li>
 *   <li><strong>Memory:</strong> ~1KB per task + ~1KB for shared context</li>
 *   <li><strong>Thread count:</strong> Still just 1 daemon thread (shared across all tasks)</li>
 *   <li><strong>Efficiency gain:</strong> Better than multiple Level 1 instances (1 thread vs N threads)</li>
 *   <li><strong>Overall impact:</strong> More efficient for multiple tasks</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Multiple independent tasks to track</li>
 *   <li>Want shared console output</li>
 *   <li>Better performance than separate StatusTrackers</li>
 *   <li>Batch processing or multiple operations</li>
 * </ul>
 */
public class Level3_MultipleTasks {

    public static void main(String[] args) {
        // NEW: Shared context for multiple tasks
        StatusContext ctx3 = new StatusContext("batch-processing");
        ctx3.addSink(new ConsoleLoggerSink());

        DataLoader loader3 = new DataLoader();
        DataValidator validator3 = new DataValidator();

        try (StatusTracker<DataLoader> loaderTracker = ctx3.track(loader3);
             StatusTracker<DataValidator> validatorTracker = ctx3.track(validator3)) {

            // Execute tasks independently
            loader3.load();
            validator3.validate();
        } finally {
            ctx3.close(); // Close context when done (shuts down monitor thread)
        }
    }
}
