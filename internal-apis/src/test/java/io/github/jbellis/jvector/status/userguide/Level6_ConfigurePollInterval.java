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

import java.time.Duration;

/**
 * Level 6: Configure Poll Interval
 *
 * <p>Building on Level 5: Adjust how frequently tasks are polled.
 *
 * <h2>Key Differences from Level 5:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> Custom poll interval (50ms instead of default 100ms)</li>
 *   <li>More responsive UI updates</li>
 *   <li>Trade CPU usage for responsiveness</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Interval configuration:</strong> Duration.ofMillis(50) in context constructor</li>
 *   <li><strong>Applies to:</strong> All trackers registered with this context</li>
 *   <li><strong>Monitor behavior:</strong> Polls all tasks every 50ms</li>
 *   <li><strong>Update frequency:</strong> Status changes visible within 50ms</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> 2x overhead vs Level 5 (polling twice as often)</li>
 *   <li><strong>Memory:</strong> Same as Level 5</li>
 *   <li><strong>Thread count:</strong> Still 1 daemon thread</li>
 *   <li><strong>Responsiveness:</strong> UI updates 2x faster</li>
 *   <li><strong>Trade-off:</strong> Faster polling = more CPU, slower polling = less CPU but laggy UI</li>
 *   <li><strong>Recommended range:</strong> 20ms (very responsive) to 1000ms (very efficient)</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Need very responsive UI updates</li>
 *   <li>Short-duration tasks where 100ms is too slow</li>
 *   <li>Interactive applications</li>
 *   <li>Conversely: Use 500-1000ms for background batch jobs</li>
 * </ul>
 */
public class Level6_ConfigurePollInterval {

    public static void main(String[] args) {
        // NEW: Custom poll interval for responsive updates
        StatusContext ctx6 = new StatusContext(
            "data-pipeline",
            Duration.ofMillis(50)); // Poll every 50ms instead of default 100ms

        ctx6.addSink(new ConsoleLoggerSink());

        DataLoader loader6 = new DataLoader();
        try (StatusTracker<DataLoader> tracker6 = ctx6.track(loader6)) {
            loader6.load(); // Task executes independently
        } finally {
            ctx6.close();
        }
    }
}
