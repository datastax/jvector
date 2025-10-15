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
import io.github.jbellis.jvector.status.StatusScope;

import java.util.ArrayList;
import java.util.List;

/**
 * Executor that manages a diverse workload of demo tasks showing different execution patterns.
 * Coordinates {@link ExampleDataProcessingTask}, {@link ExampleComputeTask}, and {@link ExampleValidationTask}
 * instances within a {@link StatusScope} to demonstrate the organizational hierarchy.
 */
public class ExampleWorkloadExecutor {
    private final List<Runnable> tasks = new ArrayList<>();
    private final List<Thread> threads = new ArrayList<>();
    private volatile boolean workloadComplete = false;
    private final SimulatedClock clock;

    private ExampleWorkloadExecutor(SimulatedClock clock) {
        this.clock = clock;
    }

    /**
     * Launches a complete demo workload with diverse task types within the given scope.
     * The scope provides organizational structure while tasks do the actual work.
     * ComputeTasks create their own child scopes containing main task and worker subtasks.
     * This method starts the workload and returns immediately.
     */
    public static ExampleWorkloadExecutor runDemoWorkload(StatusScope workloadScope, SimulatedClock clock) {
        ExampleWorkloadExecutor executor = new ExampleWorkloadExecutor(clock);

        // Create diverse workload showing different patterns
        // Simple tasks are direct children, compute tasks create their own scopes
        executor.addTask(new ExampleDataProcessingTask("DataLoad", 500, workloadScope, clock));
        executor.addTask(new ExampleComputeTask("VectorIndexing", 200, 3, workloadScope, clock));
        executor.addTask(new ExampleValidationTask("SchemaValidation", 40, workloadScope, clock));
        executor.addTask(new ExampleDataProcessingTask("DataTransform", 300, workloadScope, clock));
        executor.addTask(new ExampleComputeTask("Clustering", 150, 2, workloadScope, clock));
        executor.addTask(new ExampleValidationTask("IntegrityCheck", 30, workloadScope, clock));

        executor.startAll();
        return executor;
    }

    private void addTask(Runnable task) {
        tasks.add(task);
    }

    private void startAll() {
        for (Runnable task : tasks) {
            Thread thread = new Thread(task);
            threads.add(thread);
            thread.start();

            // Stagger start times
            try {
                clock.sleep(600);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Wait for all tasks in the workload to complete.
     */
    public boolean awaitCompletion() {
        try {
            for (Thread thread : threads) {
                thread.join();
            }
            workloadComplete = true;
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /**
     * Wait for all tasks with a timeout.
     */
    public boolean awaitCompletion(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        try {
            for (Thread thread : threads) {
                long remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) {
                    return false;
                }
                thread.join(remaining);
                if (thread.isAlive()) {
                    return false;
                }
            }
            workloadComplete = true;
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /**
     * Check if the workload has completed.
     */
    public boolean isComplete() {
        return workloadComplete;
    }

    /**
     * Interrupt all running tasks.
     */
    public void interruptAll() {
        for (Thread thread : threads) {
            thread.interrupt();
        }
    }
}
