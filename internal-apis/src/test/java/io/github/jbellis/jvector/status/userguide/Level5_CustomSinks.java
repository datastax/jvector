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
import io.github.jbellis.jvector.status.StatusScope;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;

/**
 * Level 5: Add Custom Sinks
 *
 * <p>Building on Level 4: Add metrics collection alongside console output.
 *
 * <h2>Key Differences from Level 4:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> Add MetricsStatusSink to collect performance data</li>
 *   <li><strong>NEW:</strong> Access metrics programmatically after tasks complete</li>
 *   <li>Multiple sinks receive same status updates</li>
 *   <li>Sinks operate independently</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Multiple sinks:</strong> ConsoleLoggerSink + MetricsStatusSink</li>
 *   <li><strong>Metrics collection:</strong> Tracks task count, duration, update frequency</li>
 *   <li><strong>Sink notifications:</strong> All sinks notified on every status change</li>
 *   <li><strong>Data access:</strong> Query metrics after tasks complete</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> Minimal - each sink processes updates (~microseconds per update)</li>
 *   <li><strong>Memory:</strong> +~1KB for MetricsStatusSink (stores per-task metrics)</li>
 *   <li><strong>Thread count:</strong> Still 1 daemon thread</li>
 *   <li><strong>Sink overhead:</strong> O(1) per status update per sink</li>
 *   <li><strong>Overall impact:</strong> Negligible - sinks are lightweight observers</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Need performance metrics (duration, counts)</li>
 *   <li>Want multiple output destinations (console + logs + metrics)</li>
 *   <li>Production monitoring and alerting</li>
 *   <li>SLA tracking and analysis</li>
 * </ul>
 */
public class Level5_CustomSinks {

    public static void main(String[] args) {
        // Use tasks from previous levels
        StatusContext ctx5 = new StatusContext("data-pipeline");

        // NEW: Add multiple sinks for different purposes
        ctx5.addSink(new ConsoleLoggerSink());
        MetricsStatusSink metrics5 = new MetricsStatusSink();
        ctx5.addSink(metrics5);

        DataLoader loader5 = new DataLoader();

        try (StatusScope scope5 = ctx5.createScope("Work")) {
            try (StatusTracker<DataLoader> tracker5 = scope5.trackTask(loader5)) {
                loader5.load(); // Task executes independently
            }
        }

        // NEW: Access metrics after tasks complete
        System.out.println("Total tasks: " + metrics5.getTotalTasksStarted());
        System.out.println("Avg duration: " + metrics5.getAverageTaskDuration() + "ms");

        ctx5.close();
    }
}
