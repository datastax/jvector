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

import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;

/**
 * Level 1: Absolute Minimum - Track ONE Task
 *
 * <p>This example shows the simplest possible usage of status tracking.
 * Just create a StatusTracker with your task - everything else is auto-created.
 *
 * <p><strong>Note:</strong> Without adding a sink, you won't see tracking output in the console.
 * However, the task is still being monitored and you can access status programmatically
 * through the tracker object. See the example below showing how to query elapsed time
 * and final status.
 *
 * <h2>What Gets Auto-Created:</h2>
 * <ul>
 *   <li>StatusScope named "tracker-DataLoader"</li>
 *   <li>StatusContext named "tracker-DataLoader"</li>
 *   <li>StatusMonitor with one daemon thread (polls every 100ms)</li>
 * </ul>
 *
 * <h2>Key Points:</h2>
 * <ul>
 *   <li>Task executes independently - no coupling to tracker</li>
 *   <li>Status is tracked in memory (no console output without a sink)</li>
 *   <li>Can query tracker for metrics: elapsed time, progress, state</li>
 *   <li>All resources auto-close via try-with-resources</li>
 * </ul>
 *
 * @see Level2_AddConsoleOutput for adding console visibility
 */
public class Level1_AbsoluteMinimum {

    public static void main(String[] args) {
        // Track it - simplest possible
        DataLoader loader1 = new DataLoader();
        try (StatusTracker<DataLoader> tracker1 = new StatusTracker<>(loader1)) {
            loader1.load(); // Task is independent - no tight coupling to tracker

            // Access status programmatically (no console output without a sink)
            System.out.println("Tracker status: " + tracker1); // Uses toString()
            // Output example: "DataLoader [100.0%] SUCCESS (1234ms)"

            // Or access individual metrics:
            System.out.println("Elapsed time: " + tracker1.getElapsedRunningTime() + "ms");
            System.out.println("Final state: " + tracker1.getStatus().runstate);
        } // Auto-closes everything (tracker, scope, context, monitor thread)
    }
}
