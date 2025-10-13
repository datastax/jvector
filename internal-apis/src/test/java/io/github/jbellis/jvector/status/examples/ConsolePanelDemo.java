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

package io.github.jbellis.jvector.status.examples;

import io.github.jbellis.jvector.status.LoggerConfig;
import io.github.jbellis.jvector.status.SimulatedClock;
import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.sinks.ConsolePanelSink;
import io.github.jbellis.jvector.status.sinks.OutputMode;

import java.util.concurrent.TimeUnit;

/**
 * Demonstration application for {@link ConsolePanelSink} with hierarchical task display and
 * integrated log panel. This minimal orchestrator creates a context with ConsolePanelSink,
 * launches a diverse workload via {@link ExampleWorkloadExecutor}, and waits for completion.
 *
 * <p>Key Features Demonstrated:
 * <ul>
 *   <li><strong>Visual Hierarchy:</strong> Multi-level task trees with clear parent-child relationships</li>
 *   <li><strong>Real-time Updates:</strong> Live progress bars, status icons, and timing information</li>
 *   <li><strong>Interactive Controls:</strong> Keyboard navigation (↑↓ for scrolling, q to quit)</li>
 *   <li><strong>Log Integration:</strong> Automatic capture of log4j2 output in scrollable panel</li>
 *   <li><strong>Terminal Optimization:</strong> Efficient differential rendering with JLine3</li>
 * </ul>
 *
 * <h2>Display Layout</h2>
 * <p>The console panel provides two sections:</p>
 * <ul>
 *   <li><strong>Task Status Panel (top):</strong> Hierarchical view of all active and completed tasks
 *       with progress bars, duration tracking, and color-coded status indicators</li>
 *   <li><strong>Console Output Panel (bottom):</strong> Scrollable log display showing all log4j2
 *       messages with level-based color coding and timestamps</li>
 * </ul>
 *
 * <h2>Architecture</h2>
 * <p>This demo illustrates the clean separation between organizational scopes and work tasks:</p>
 * <ol>
 *   <li>{@link ConsolePanelDemo} - Creates context with organizational scope for workload</li>
 *   <li>{@link TrackerScope} - Organizational container with no progress/state</li>
 *   <li>Task types ({@link ExampleDataProcessingTask}, {@link ExampleComputeTask}, {@link ExampleValidationTask}) -
 *       Leaf nodes with actual progress and state</li>
 *   <li>{@link StatusContext} - Routes status updates from tasks to sinks</li>
 *   <li>{@link ConsolePanelSink} - Receives and displays status updates in interactive UI</li>
 * </ol>
 *
 * <h2>Running the Demo</h2>
 * <p>Execute from the command line with Maven:</p>
 * <pre>
 * mvn test-compile exec:java -pl internal-apis \
 *   -Dexec.mainClass="io.github.jbellis.jvector.status.examples.ConsolePanelDemo" \
 *   -Dexec.classpathScope=test
 * </pre>
 *
 * <h2>Interactive Controls</h2>
 * <ul>
 *   <li><strong>↑ / ↓:</strong> Scroll through console log output</li>
 *   <li><strong>[ / ]:</strong> Adjust split between task and log panels</li>
 *   <li><strong>PgUp / PgDn:</strong> Quick split adjustment</li>
 *   <li><strong>Home:</strong> Reset scroll positions and split to defaults</li>
 *   <li><strong>End:</strong> Jump to end of logs and tasks</li>
 *   <li><strong>s:</strong> Save current display to file</li>
 *   <li><strong>q:</strong> Quit and shutdown (gracefully cancels tasks)</li>
 * </ul>
 *
 * <h2>Expected Behavior</h2>
 * <p>The demo will:</p>
 * <ol>
 *   <li>Display startup messages for 3 seconds</li>
 *   <li>Launch 6 diverse task groups with staggered starts:
 *       <ul>
 *         <li>DataLoad (500 records) - I/O simulation</li>
 *         <li>VectorIndexing scope - Parallel computation with main task (200 iterations) and 3 worker subtasks</li>
 *         <li>SchemaValidation (40 checks) - Sequential validation with nested checks</li>
 *         <li>DataTransform (300 records) - Data processing</li>
 *         <li>Clustering scope - Concurrent computation with main task (150 iterations) and 2 worker subtasks</li>
 *         <li>IntegrityCheck (30 checks) - Final validation</li>
 *       </ul>
 *   </li>
 *   <li>Tasks spawn child tasks demonstrating different hierarchy patterns</li>
 *   <li>All tasks emit log messages at various levels (INFO, DEBUG, WARN, ERROR)</li>
 *   <li>Tasks complete naturally after their configured work</li>
 *   <li>Console panel remains visible for 3 seconds after completion</li>
 *   <li>Application cleans up and exits, printing log history to stdout</li>
 * </ol>
 *
 * @see ConsolePanelSink
 * @see ExampleWorkloadExecutor
 * @see ExampleDataProcessingTask
 * @see ExampleComputeTask
 * @see ExampleValidationTask
 * @see StatusContext
 * @since 4.0.0
 */
public class ConsolePanelDemo {

    /**
     * Main entry point for the console panel demonstration. Creates a status context with
     * ConsolePanelSink, launches a diverse workload with multiple task types, and waits for completion.
     *
     * @param args command line arguments (unused)
     * @throws Exception if initialization or execution fails
     */
    public static void main(String[] args) throws Exception {
        // Configure logging to capture all log events in LogBuffer
        LoggerConfig.configure(OutputMode.INTERACTIVE);

        // Create simulated clock for time control
        SimulatedClock clock = new SimulatedClock();

        // Use array to allow lambda to reference the sink before it's fully initialized
        ConsolePanelSink[] sinkHolder = new ConsolePanelSink[1];

        // Create the enhanced ConsolePanelSink with time control callbacks
        ConsolePanelSink consolePanelSink = ConsolePanelSink.builder()
                .withRefreshRateMs(100)
                .withCompletedTaskRetention(5, TimeUnit.SECONDS)
                .withColorOutput(true)
                .withMaxLogLines(100)
                .withCaptureSystemStreams(true)
                .withKeyHandler("shift-right", () -> {
                    clock.speedUp();
                    sinkHolder[0].addLogMessage("Time speed: " + clock.getSpeedDescription());
                })
                .withKeyHandler("shift-left", () -> {
                    clock.slowDown();
                    sinkHolder[0].addLogMessage("Time speed: " + clock.getSpeedDescription());
                })
                .build();
        sinkHolder[0] = consolePanelSink;

        // Print reminder to terminal before entering interactive mode
        System.out.println("\n" +
            "=".repeat(70) + "\n" +
            "  ConsolePanelSink Demo - Interactive Task Monitor\n" +
            "=".repeat(70) + "\n" +
            "\n" +
            "  This demo demonstrates:\n" +
            "    • DataProcessingTask: I/O-heavy operations (record processing)\n" +
            "    • ComputeTask: CPU-intensive work with parallel subtasks\n" +
            "    • ValidationTask: Sequential checks with nested validations\n" +
            "\n" +
            "  Demo-specific feature:\n" +
            "    • Shift+← / Shift+→: Control simulated time speed\n" +
            "\n" +
            "  Press '?' inside the monitor to see all keyboard shortcuts\n" +
            "\n" +
            "=".repeat(70) + "\n");

        try (StatusContext context = new StatusContext("console-demo")) {
            context.addSink(consolePanelSink);

            // Create organizational scope for the workload
            try (TrackerScope workloadScope = context.createScope("DemoWorkload")) {

                // Launch diverse workload - all tasks are within the scope
                ExampleWorkloadExecutor executor = ExampleWorkloadExecutor.runDemoWorkload(workloadScope, clock);

                // Wait for workload to complete, checking for early shutdown
                while (!executor.isComplete() && !consolePanelSink.isClosed()) {
                    if (!executor.awaitCompletion(1000)) {
                        if (consolePanelSink.isClosed()) {
                            executor.interruptAll();
                            executor.awaitCompletion(2000);
                            break;
                        }
                    }
                }

                // Wait for user to close the panel (via 'q' key)
                // The panel will stay open until user explicitly closes it
                while (!consolePanelSink.isClosed()) {
                Thread.sleep(100);
                }
            }

        } finally {
            // Ensure panel is closed (in case of exceptions)
            if (!consolePanelSink.isClosed()) {
                consolePanelSink.close();
            }
        }
    }

}
