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
import org.junit.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Test and demonstration of ConsolePanelSink functionality.
 * Note: This test is primarily for demonstration purposes as JLine
 * terminal interaction is difficult to test in automated tests.
 *
 * Run this test manually to see the hierarchical display in action.
 */
public class ConsolePanelSinkTest {

    /**
     * Demonstrates the ConsolePanelSink with multiple concurrent tasks
     * in a hierarchical structure.
     *
     * Run this test manually to see the visual output.
     */
    @Test
    public void demonstrateConsolePanelSink() throws Exception {
        // Skip in automated tests - this is for manual demonstration
        if (System.getProperty("consolepanel.demo") == null) {
            System.out.println("Skipping ConsolePanelSink demo. Run with -Dconsolepanel.demo=true to see visual output.");
            return;
        }

        // Create the ConsolePanelSink with custom configuration including logging
        ConsolePanelSink consolePanelSink = ConsolePanelSink.builder()
                .withRefreshRateMs(100)
                .withCompletedTaskRetention(3, TimeUnit.SECONDS)
                .withColorOutput(true)
                .withMaxLogLines(500)
                .build();

        // Console output will now be captured and displayed in the console panel
        System.out.println("[INFO] Starting ConsolePanelSink demonstration");
        System.out.println("[INFO] All console output is now captured in the panel");

        try {
            // Create a root task
            DemoTask rootTask = new DemoTask("DataProcessing", 10000);

            // Start tracking the root task
            try (StatusTracker<DemoTask> rootTracker = StatusTracker.withInstrumented(
                    rootTask,
                    java.time.Duration.ofMillis(200),
                    List.of(consolePanelSink))) {

                // Simulate the root task starting
                rootTask.start();
                System.out.println("[INFO] Root task started: DataProcessing");

                // Create and run child tasks concurrently
                CompletableFuture<Void> task1 = CompletableFuture.runAsync(() -> {
                    System.out.println("[INFO] Starting subtask: LoadData");
                    DemoTask subTask1 = new DemoTask("LoadData", 3000);
                    try (StatusTracker<DemoTask> tracker = StatusTracker.withInstrumented(
                            subTask1,
                            java.time.Duration.ofMillis(100),
                            List.of(consolePanelSink))) {
                        subTask1.execute();
                        System.out.println("[INFO] Completed subtask: LoadData");
                    }
                });

                CompletableFuture<Void> task2 = CompletableFuture.runAsync(() -> {
                    try {
                        Thread.sleep(1000); // Start slightly delayed
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }

                    System.out.println("[INFO] Starting subtask: ProcessRecords");
                    DemoTask subTask2 = new DemoTask("ProcessRecords", 5000);
                    try (StatusTracker<DemoTask> tracker = StatusTracker.withInstrumented(
                            subTask2,
                            java.time.Duration.ofMillis(100),
                            List.of(consolePanelSink))) {
                        subTask2.execute();
                        System.out.println("[WARN] ProcessRecords encountered cache miss");

                        // Nested sub-task
                        System.out.println("[DEBUG] Starting validation phase");
                        DemoTask nestedTask = new DemoTask("ValidateData", 2000);
                        try (StatusTracker<DemoTask> nestedTracker = StatusTracker.withInstrumented(
                                nestedTask,
                                java.time.Duration.ofMillis(100),
                                List.of(consolePanelSink))) {
                            nestedTask.execute();
                            System.out.println("[INFO] Validation completed successfully");
                        }
                        System.out.println("[INFO] Completed subtask: ProcessRecords");
                    }
                });

                CompletableFuture<Void> task3 = CompletableFuture.runAsync(() -> {
                    try {
                        Thread.sleep(2000); // Start more delayed
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }

                    System.out.println("[INFO] Starting subtask: SaveResults");
                    DemoTask subTask3 = new DemoTask("SaveResults", 4000);
                    try (StatusTracker<DemoTask> tracker = StatusTracker.withInstrumented(
                            subTask3,
                            java.time.Duration.ofMillis(100),
                            List.of(consolePanelSink))) {
                        subTask3.execute();
                        System.out.println("[ERROR] Failed to save to primary location, using backup");
                        System.out.println("[INFO] Results saved successfully to backup location");
                        System.out.println("[INFO] Completed subtask: SaveResults");
                    }
                });

                // Execute root task while children are running
                System.out.println("[INFO] Processing main dataset in parallel with subtasks");
                rootTask.execute();
                System.out.println("[INFO] Main dataset processing complete");

                // Wait for all tasks to complete
                CompletableFuture.allOf(task1, task2, task3).join();
                System.out.println("[INFO] All tasks completed successfully");
            }

            // Generate some final log messages
            System.out.println("[INFO] =======================================\n");
            System.out.println("[INFO] Final Summary:");
            System.out.println("[INFO] - Data loaded: 100%");
            System.out.println("[INFO] - Records processed: 50,000");
            System.out.println("[INFO] - Validation passed: true");
            System.out.println("[INFO] - Results saved: backup location");
            System.out.println("[INFO] =======================================\n");

            // Keep display visible for a moment to see completed tasks
            Thread.sleep(5000);

        } finally {
            consolePanelSink.close();
        }
    }

    /**
     * Demo task implementation that simulates work with progress updates
     */
    private static class DemoTask implements StatusUpdate.Provider<DemoTask> {
        private final String name;
        private final long durationMs;
        private volatile double progress = 0.0;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;

        DemoTask(String name, long durationMs) {
            this.name = name;
            this.durationMs = durationMs;
        }

        public String getName() {
            return name;
        }

        void start() {
            this.state = StatusUpdate.RunState.RUNNING;
        }

        void execute() {
            start();

            // Simulate progressive work with occasional log messages
            int steps = 20;
            long stepDuration = durationMs / steps;

            for (int i = 0; i <= steps; i++) {
                this.progress = (double) i / steps;

                // Generate occasional log messages during execution
                if (i == 5) {
                    System.out.println("[DEBUG] " + name + ": 25% complete");
                } else if (i == 10) {
                    System.out.println("[DEBUG] " + name + ": 50% complete");
                } else if (i == 15) {
                    System.out.println("[DEBUG] " + name + ": 75% complete");
                }

                if (i < steps) {
                    try {
                        Thread.sleep(stepDuration);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        this.state = StatusUpdate.RunState.CANCELLED;
                        return;
                    }
                }
            }

            this.state = StatusUpdate.RunState.SUCCESS;
        }

        @Override
        public StatusUpdate<DemoTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return name;
        }
    }
}