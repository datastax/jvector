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

import io.github.jbellis.jvector.status.SimulatedClock;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Demonstration task that simulates CPU-intensive compute operations with configurable
 * parallel subtask spawning. This class illustrates hierarchical task organization by
 * creating its own {@link TrackerScope} containing the main computation task and optional
 * worker subtasks.
 *
 * <p>The task performs simulated computation through trigonometric calculations while
 * emitting log messages at various levels to demonstrate logging integration. When parallel
 * subtasks are configured, each worker task executes independently with half the iteration
 * count of the parent task.
 *
 * <h2>Task Hierarchy</h2>
 * <pre>
 * ExampleComputeTask (scope owner)
 *   ├── MainTask (actual computation with progress tracking)
 *   └── Workers (optional nested scope)
 *       ├── Worker1 (ExampleComputeTask with reduced iterations)
 *       ├── Worker2
 *       └── Worker3...
 * </pre>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * try (TrackerScope scope = context.createScope("MyScope")) {
 *     SimulatedClock clock = new SimulatedClock();
 *     ExampleComputeTask task = new ExampleComputeTask(
 *         "MyComputeTask",
 *         200,  // iterations
 *         3,    // parallel workers
 *         scope,
 *         clock
 *     );
 *     new Thread(task).start();
 * }
 * }</pre>
 *
 * @see TrackerScope
 * @see StatusTracker
 * @see SimulatedClock
 * @since 4.0.0
 */
class ExampleComputeTask implements Runnable {
    private static final Logger logger = LogManager.getLogger(ExampleComputeTask.class);

    private final String name;
    private final int iterations;
    private final int parallelSubtasks;
    private final TrackerScope parentScope;
    private final SimulatedClock clock;
    private final List<Thread> childThreads = new ArrayList<>();

    /**
     * Constructs a new compute task with the specified configuration.
     *
     * @param name the display name for this task
     * @param iterations the number of computation iterations to perform
     * @param parallelSubtasks the number of parallel worker tasks to spawn (0 for no workers)
     * @param parentScope the parent tracker scope under which this task's scope will be created
     * @param clock the simulated clock for controlling task timing
     */
    ExampleComputeTask(String name, int iterations, int parallelSubtasks, TrackerScope parentScope, SimulatedClock clock) {
        this.name = name;
        this.iterations = iterations;
        this.parallelSubtasks = parallelSubtasks;
        this.parentScope = parentScope;
        this.clock = clock;
    }

    /**
     * Executes the compute task by creating a task-specific scope, spawning parallel
     * worker tasks if configured, running the main computation, and waiting for all
     * child tasks to complete. Exceptions are logged but do not propagate.
     */
    @Override
    public void run() {
        try (TrackerScope taskScope = parentScope.createChildScope(name)) {
            // Spawn parallel worker tasks if needed
            TrackerScope workersScope = null;
            if (parallelSubtasks > 0) {
                logger.info("Spawning {} parallel worker tasks for {}", parallelSubtasks, name);
                workersScope = taskScope.createChildScope("Workers");
                for (int i = 0; i < parallelSubtasks; i++) {
                    String childName = "Worker" + (i + 1);
                    logger.debug("Creating worker task: {}", childName);
                    ExampleComputeTask childTask = new ExampleComputeTask(childName, iterations / 2, 0, workersScope, clock);
                    Thread childThread = new Thread(childTask);
                    childThreads.add(childThread);
                    childThread.start();
                    logger.trace("Worker {} started with thread ID: {}", childName, childThread.getId());
                    clock.sleep(100);
                }
                logger.info("All {} worker tasks spawned successfully", parallelSubtasks);
            }

            // Execute the main task
            MainTask mainTask = new MainTask(name, iterations, taskScope, clock);
            mainTask.execute();

            // Wait for child tasks
            if (!childThreads.isEmpty()) {
                logger.debug("Waiting for {} child tasks to complete", childThreads.size());
                for (Thread child : childThreads) {
                    logger.trace("Joining thread: {}", child.getName());
                    child.join(2000);
                }
                logger.info("All child tasks completed for {}", name);
            }

            // Close workers scope if created
            if (workersScope != null) {
                workersScope.close();
                logger.debug("Workers scope closed for {}", name);
            }
        } catch (Exception e) {
            logger.error("Compute task {} failed: {}", name, e.getMessage());
        }
    }

    /**
     * Inner class representing the actual computation work with progress tracking.
     * This class implements {@link StatusSource} to provide real-time progress updates
     * to the status tracking system.
     *
     * <p>The task performs computation in iterations, simulating CPU-intensive work through
     * trigonometric calculations. Progress, state, and completion status are exposed through
     * the {@link #getTaskStatus()} method.
     *
     * @see StatusSource
     * @see StatusUpdate
     */
    private static class MainTask implements StatusSource<MainTask> {
        private final String name;
        private final int iterations;
        private final TrackerScope scope;
        private final SimulatedClock clock;
        private final Random random = new Random();

        private volatile int iterationsComplete = 0;
        private volatile RunState state = RunState.PENDING;
        private volatile boolean interrupted = false;

        /**
         * Constructs a new main task for computation execution.
         *
         * @param name the task name for display purposes
         * @param iterations the total number of iterations to perform
         * @param scope the tracker scope for registering this task
         * @param clock the simulated clock for timing control
         */
        MainTask(String name, int iterations, TrackerScope scope, SimulatedClock clock) {
            this.name = name;
            this.iterations = iterations;
            this.scope = scope;
            this.clock = clock;
        }

        /**
         * Executes the main computation task with progress tracking. Creates a
         * {@link StatusTracker} for this task, performs the computation loop,
         * and handles interruption gracefully.
         *
         * @throws InterruptedException if the task is interrupted during execution
         */
        void execute() throws InterruptedException {
            try (StatusTracker<MainTask> tracker = scope.trackTask(this, MainTask::getTaskStatus)) {
                state = RunState.RUNNING;
                logger.info("Starting compute task: {} ({} iterations)", name, iterations);
                logger.debug("Task configuration - Iterations: {}, Tracker: {}", iterations, tracker.getClass().getSimpleName());
                logger.trace("Initializing computation state for task: {}", name);

                executeComputation();

                if (interrupted) {
                    logger.warn("Compute task {} interrupted at iteration {}/{}", name, iterationsComplete, iterations);
                    logger.debug("Cleanup after interruption for task: {}", name);
                    state = RunState.CANCELLED;
                } else {
                    iterationsComplete = iterations;
                    state = RunState.SUCCESS;
                    logger.info("✓ Completed compute task: {} ({} iterations)", name, iterations);
                    logger.debug("Final state: SUCCESS, Total iterations: {}", iterations);
                    logger.trace("Task {} deallocating resources", name);
                }
            }
        }

        private void executeComputation() throws InterruptedException {
            // Perform computation
            logger.debug("Beginning main computation loop for {} iterations", iterations);
            for (int i = 0; i < iterations && !interrupted; i++) {
                iterationsComplete = i + 1;

                logger.trace("Iteration {} starting for task {}", i, name);

                // Simulate computation (busy work)
                double result = 0;
                for (int j = 0; j < 1000; j++) {
                    result += Math.sin(j * 0.01) * Math.cos(i * 0.01);
                }

                if (i % 10 == 0) {
                    clock.sleep(20); // Small delay between iterations
                    logger.trace("Checkpoint at iteration {}, result: {}", i, result);
                }

                if (i % 20 == 0 && i > 0) {
                    logger.debug("Compute progress: {}/{} iterations ({}%), result: {}",
                               iterationsComplete, iterations, (iterationsComplete * 100 / iterations), result);
                }

                if (i % 50 == 0 && i > 0) {
                    logger.info("Milestone: {} completed {}/{} iterations", name, iterationsComplete, iterations);
                }

                // Simulate various events at different log levels
                int rand = random.nextInt(100);
                if (rand > 98) {
                    logger.warn("Compute task {} encountered cache miss at iteration {}", name, i);
                } else if (rand > 95) {
                    logger.debug("Minor performance hiccup at iteration {} (still within bounds)", i);
                } else if (rand > 92) {
                    logger.trace("Memory allocation occurred at iteration {}", i);
                } else if (rand > 90) {
                    logger.info("Checkpoint saved for task {} at iteration {}", name, i);
                }

                if (i % 100 == 0 && i > 0) {
                    double progress = (i * 100.0) / iterations;
                    if (progress < 30) {
                        logger.debug("Early stage: {}% complete", progress);
                    } else if (progress < 70) {
                        logger.debug("Mid stage: {}% complete", progress);
                    } else {
                        logger.debug("Final stage: {}% complete", progress);
                    }
                }
            }
        }

        /**
         * Returns the current status of this task including progress fraction and run state.
         * Progress is calculated as the ratio of completed iterations to total iterations.
         *
         * @return a status update containing progress, state, and this task as the source
         */
        @Override
        public StatusUpdate<MainTask> getTaskStatus() {
            double progress = iterations > 0 ? (double) iterationsComplete / iterations : 0.0;
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
