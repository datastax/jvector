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
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;

/**
 * Level 2: Add Console Output
 *
 * <p>Building on Level 1: Add a sink to see progress in the console.
 *
 * <h2>Key Differences from Level 1:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> Adds ConsoleLoggerSink to display progress</li>
 *   <li>Same auto-created StatusScope and StatusContext as Level 1</li>
 *   <li>Same single monitoring thread</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Sink access:</strong> Via tracker.getContext().addSink()</li>
 *   <li><strong>Output format:</strong> Timestamps + progress bars + status</li>
 *   <li><strong>When visible:</strong> Progress updates every ~100ms (default poll interval)</li>
 *   <li><strong>Output destination:</strong> System.out</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> Minimal - same as Level 1 plus console writes every 100ms</li>
 *   <li><strong>Memory:</strong> +~500 bytes for ConsoleLoggerSink</li>
 *   <li><strong>Thread count:</strong> Still just 1 daemon thread</li>
 *   <li><strong>I/O impact:</strong> Console writes only on status changes (not every iteration)</li>
 *   <li><strong>Overall impact:</strong> Negligible - console I/O is buffered</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Want to see progress in console/logs</li>
 *   <li>Running interactively or viewing logs</li>
 *   <li>Debugging or development</li>
 *   <li>Simple command-line tools</li>
 * </ul>
 */
public class Level2_AddConsoleOutput {

    public static void main(String[] args) {
        // Same DataLoader class from Level 1
        DataLoader loader2 = new DataLoader();
        try (StatusTracker<DataLoader> tracker2 = new StatusTracker<>(loader2)) {
            // NEW: Add a sink to display progress
            tracker2.getContext().addSink(new ConsoleLoggerSink());

            loader2.load(); // Task executes independently
        }
    }
}
