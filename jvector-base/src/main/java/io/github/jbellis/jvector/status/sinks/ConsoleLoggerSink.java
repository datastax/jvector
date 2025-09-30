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

package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.*;

import java.io.PrintStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * A human-readable task sink that outputs task progress and lifecycle events to the console
 * or any PrintStream. This sink provides real-time visual feedback with customizable formatting
 * including progress bars, timestamps, and status indicators.
 *
 * <p>This sink provides:
 * <ul>
 *   <li>Visual progress bars with Unicode block characters</li>
 *   <li>Task lifecycle events (started, updates, finished)</li>
 *   <li>Configurable timestamp display with millisecond precision</li>
 *   <li>Flexible output destination (console, file, etc.)</li>
 *   <li>Automatic task name extraction from various object types</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Basic Console Output</h3>
 * <pre>{@code
 * // Default configuration: System.out, timestamps, progress bars
 * TaskSink consoleSink = new ConsoleTaskSink();
 *
 * try (Tracker<MyTask> tracker = Tracker.withInstrumented(task, consoleSink)) {
 *     task.execute();
 *     // Output:
 *     // [14:32:15.123] ▶ Started: data-processing
 *     // [14:32:15.245]   data-processing [████████████░░░░░░░░]  60.0% - RUNNING
 *     // [14:32:16.891] ✓ Finished: data-processing
 * }
 * }</pre>
 *
 * <h3>Custom PrintStream Output</h3>
 * <pre>{@code
 * // Output to a file or custom stream
 * PrintStream fileOut = new PrintStream(new FileOutputStream("progress.log"));
 * TaskSink fileSink = new ConsoleTaskSink(fileOut);
 *
 * try (Tracker<BatchJob> tracker = Tracker.withInstrumented(job, fileSink)) {
 *     job.processBatch();
 * } finally {
 *     fileOut.close();
 * }
 * }</pre>
 *
 * <h3>Minimal Output Format</h3>
 * <pre>{@code
 * // No timestamps, no progress bars - just basic text updates
 * TaskSink simpleSink = new ConsoleTaskSink(System.out, false, false);
 *
 * try (Tracker<ImportTask> tracker = Tracker.withInstrumented(task, simpleSink)) {
 *     task.importData();
 *     // Output:
 *     // ▶ Started: csv-import
 *     //   csv-import [75.0%] - RUNNING
 *     // ✓ Finished: csv-import
 * }
 * }</pre>
 *
 * <h3>Multiple Tasks with Scope</h3>
 * <pre>{@code
 * TrackerScope scope = new TrackerScope("batch-operations");
 * scope.addSink(new ConsoleTaskSink());
 *
 * // All tasks in scope will output to console
 * try (Tracker<Task1> t1 = scope.track(task1);
 *      Tracker<Task2> t2 = scope.track(task2)) {
 *
 *     CompletableFuture.allOf(
 *         CompletableFuture.runAsync(task1::execute),
 *         CompletableFuture.runAsync(task2::execute)
 *     ).join();
 * }
 * }</pre>
 *
 * <h3>Debugging and Development</h3>
 * <pre>{@code
 * // Use different streams for different priority levels
 * TaskSink debugSink = new ConsoleTaskSink(System.err, true, true);
 *
 * try (Tracker<DebugTask> tracker = Tracker.withInstrumented(debugTask, debugSink)) {
 *     // Debug output goes to stderr with full formatting
 * }
 * }</pre>
 *
 * <h2>Output Format</h2>
 * <p>The sink produces formatted output with these elements:</p>
 * <ul>
 *   <li><strong>Timestamps:</strong> [HH:mm:ss.SSS] format when enabled</li>
 *   <li><strong>Status Icons:</strong> ▶ for started, ✓ for finished</li>
 *   <li><strong>Progress Bars:</strong> Unicode block characters (█ filled, ░ empty)</li>
 *   <li><strong>Progress Text:</strong> Percentage with one decimal place</li>
 *   <li><strong>Run State:</strong> Current TaskStatus.RunState value</li>
 * </ul>
 *
 * <h2>Task Name Resolution</h2>
 * <p>The sink automatically extracts meaningful task names using this priority:</p>
 * <ol>
 *   <li>getName() method found via reflection</li>
 *   <li>Any getName() method found via reflection</li>
 *   <li>Object.toString() as fallback</li>
 * </ol>
 *
 * <h2>Thread Safety</h2>
 * <p>This sink is thread-safe and can handle concurrent updates from multiple trackers.
 * Output lines are atomic but may be interleaved if multiple tasks update simultaneously.</p>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see TrackerScope
 * @since 4.0.0
 */
public class ConsoleLoggerSink implements StatusSink {

    private final PrintStream output;
    private final boolean showTimestamp;
    private final boolean useProgressBar;
    private final DateTimeFormatter timeFormatter;

    public ConsoleLoggerSink() {
        this(System.out, true, true);
    }

    public ConsoleLoggerSink(PrintStream output) {
        this(output, true, true);
    }

    public ConsoleLoggerSink(PrintStream output, boolean showTimestamp, boolean useProgressBar) {
        this.output = output;
        this.showTimestamp = showTimestamp;
        this.useProgressBar = useProgressBar;
        this.timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
        String taskName = getTaskName(task);
        String timestamp = showTimestamp ? "[" + LocalDateTime.now().format(timeFormatter) + "] " : "";
        output.println(timestamp + "▶ Started: " + taskName);
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        String taskName = getTaskName(task);
        double progress = status.progress;
        String timestamp = showTimestamp ? "[" + LocalDateTime.now().format(timeFormatter) + "] " : "";

        if (useProgressBar) {
            String progressBar = createProgressBar(progress);
            String progressText = String.format(" %.1f%%", progress * 100);
            output.println(timestamp + "  " + taskName + " " + progressBar + progressText + " - " + status.runstate);
        } else {
            output.println(timestamp + "  " + taskName + " [" + String.format("%.1f%%", progress * 100) + "] - " + status.runstate);
        }
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        String taskName = getTaskName(task);
        String timestamp = showTimestamp ? "[" + LocalDateTime.now().format(timeFormatter) + "] " : "";
        output.println(timestamp + "✓ Finished: " + taskName);
    }

    private String createProgressBar(double progress) {
        int barLength = 20;
        int filled = (int) (barLength * progress);
        int empty = barLength - filled;

        StringBuilder bar = new StringBuilder("[");
        for (int i = 0; i < filled; i++) {
            bar.append("█");
        }
        for (int i = 0; i < empty; i++) {
            bar.append("░");
        }
        bar.append("]");

        return bar.toString();
    }

    private String getTaskName(StatusTracker<?> task) {
        Object tracked = task.getTracked();
        // SimpleTask moved to test package - use reflection for getName()
        // Try to get name via reflection
        try {
            java.lang.reflect.Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
            // Fall back to toString
        }
        return tracked.toString();
    }
}