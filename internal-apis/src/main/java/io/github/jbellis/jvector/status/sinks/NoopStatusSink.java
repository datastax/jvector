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

import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusContext;

/**
 * A no-operation task sink that discards all task events and produces no output.
 * This sink is useful for disabling task monitoring without changing tracking code,
 * testing scenarios where output interferes with assertions, and performance-sensitive
 * environments where monitoring overhead must be minimized.
 *
 * <p>This sink provides:
 * <ul>
 *   <li>Zero overhead - all methods are empty and inline</li>
 *   <li>Singleton pattern to minimize memory usage</li>
 *   <li>Thread-safe operation with no state</li>
 *   <li>Drop-in replacement for other sinks</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Conditional Monitoring</h3>
 * <pre>{@code
 * // Enable monitoring only in development
 * TaskSink sink = isDevelopment()
 *     ? new ConsoleTaskSink()
 *     : NoopTaskSink.getInstance();
 *
 * try (Tracker<DataProcessor> tracker = Tracker.withInstrumented(processor, sink)) {
 *     processor.processData(); // Monitoring enabled/disabled based on environment
 * }
 * }</pre>
 *
 * <h3>Testing Without Output</h3>
 * <pre>{@code
 * // Prevent console output during unit tests
 * @Test
 * public void testTaskExecution() {
 *     StatusContext context = new StatusContext("test-scope");
 *     context.addSink(NoopTaskSink.getInstance());
 *
 *     try (StatusTracker<TestTask> tracker = context.track(testTask)) {
 *         testTask.execute();
 *         // No console output, test runs cleanly
 *         assertEquals(TestTask.State.COMPLETED, testTask.getState());
 *     }
 * }
 * }</pre>
 *
 * <h3>Performance Benchmarking</h3>
 * <pre>{@code
 * // Benchmark task execution without monitoring overhead
 * TaskSink noopSink = NoopTaskSink.getInstance();
 * long startTime = System.nanoTime();
 *
 * try (Tracker<BenchmarkTask> tracker = Tracker.withInstrumented(task, noopSink)) {
 *     task.executeBenchmark();
 * }
 *
 * long duration = System.nanoTime() - startTime;
 * System.out.println("Pure execution time: " + duration / 1_000_000 + "ms");
 * }</pre>
 *
 * <h3>Configurable Sink Selection</h3>
 * <pre>{@code
 * public class TaskRunner {
 *     private final TaskSink sink;
 *
 *     public TaskRunner(boolean enableLogging) {
 *         this.sink = enableLogging
 *             ? new LoggerTaskSink("app.tasks")
 *             : NoopTaskSink.getInstance();
 *     }
 *
 *     public void runTask(MyTask task) {
 *         try (Tracker<MyTask> tracker = Tracker.withInstrumented(task, sink)) {
 *             task.execute();
 *         }
 *     }
 * }
 * }</pre>
 *
 * <h3>Default Sink in Context Configuration</h3>
 * <pre>{@code
 * public StatusContext createContext(String name, boolean enableMonitoring) {
 *     StatusContext context = new StatusContext(name);
 *     if (enableMonitoring) {
 *         context.addSink(new ConsoleTaskSink());
 *         context.addSink(new MetricsTaskSink());
 *     } else {
 *         context.addSink(NoopTaskSink.getInstance());
 *     }
 *     return context;
 * }
 * }</pre>
 *
 * <h2>Design Pattern</h2>
 * <p>This class implements the Null Object pattern, providing a valid sink implementation
 * that performs no operations. This eliminates the need for null checks and conditional
 * logic throughout the tracking code.</p>
 *
 * <h2>Singleton Pattern</h2>
 * <p>The class uses a singleton pattern since all instances behave identically. Use
 * {@link #getInstance()} to obtain the shared instance rather than creating new objects.</p>
 *
 * <h2>Performance</h2>
 * <p>This sink has zero overhead:</p>
 * <ul>
 *   <li>All methods are empty and will be inlined by the JIT compiler</li>
 *   <li>No memory allocations or I/O operations</li>
 *   <li>No synchronization overhead</li>
 *   <li>Minimal object memory footprint due to singleton pattern</li>
 * </ul>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see StatusContext
 * @since 4.0.0
 */
public class NoopStatusSink implements StatusSink {

    private static final NoopStatusSink INSTANCE = new NoopStatusSink();

    private NoopStatusSink() {
    }

    public static NoopStatusSink getInstance() {
        return INSTANCE;
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
    }
}
