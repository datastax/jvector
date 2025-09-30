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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.status.*;
import io.github.jbellis.jvector.status.sinks.ConsolePanelSink;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Demo of the enhanced ConsolePanelSink with visual boundaries.
 * Run this to see the improved panel layout with clear borders.
 */
public class ConsolePanelDemo {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting ConsolePanelSink Demo with Enhanced Visual Boundaries");
        System.out.println("=============================================================");
        System.out.println("The panels will have:");
        System.out.println("  - Bold cyan borders around each panel");
        System.out.println("  - Yellow highlighted panel titles");
        System.out.println("  - Clear visual separation between task and console panels");
        System.out.println("  - Consistent box drawing characters for better visibility");
        System.out.println();
        System.out.println("Press Ctrl+C to exit");
        Thread.sleep(3000);

        // Create the enhanced ConsolePanelSink
        ConsolePanelSink consolePanelSink = ConsolePanelSink.builder()
                .withRefreshRateMs(100)
                .withCompletedTaskRetention(5, TimeUnit.SECONDS)
                .withColorOutput(true)
                .withMaxLogLines(100)
                .withCaptureSystemStreams(true)
                .build();

        try {
            // Create demo tasks
            DemoTask mainTask = new DemoTask("MainProcess", 30000);

            try (StatusTracker<DemoTask> mainTracker = StatusTracker.withInstrumented(
                    mainTask,
                    java.time.Duration.ofMillis(200),
                    List.of(consolePanelSink))) {

                mainTask.start();
                System.out.println("[INFO] Main process started");

                // Create some subtasks
                Thread subThread1 = new Thread(() -> runSubTask("DataLoad", 10000, consolePanelSink));
                Thread subThread2 = new Thread(() -> runSubTask("Processing", 15000, consolePanelSink));
                Thread subThread3 = new Thread(() -> runSubTask("Validation", 8000, consolePanelSink));

                subThread1.start();
                Thread.sleep(1000);
                subThread2.start();
                Thread.sleep(1000);
                subThread3.start();

                // Generate log messages while tasks run
                for (int i = 0; i < 30; i++) {
                    Thread.sleep(1000);
                    mainTask.updateProgress((double) i / 30.0);

                    if (i % 5 == 0) {
                        System.out.println("[INFO] Processing batch " + i);
                    }
                    if (i % 7 == 0) {
                        System.out.println("[DEBUG] Cache hit ratio: " + (70 + i) + "%");
                    }
                    if (i == 15) {
                        System.err.println("[WARN] Memory usage above threshold");
                    }
                    if (i == 20) {
                        System.err.println("[ERROR] Connection timeout (will retry)");
                    }
                }

                mainTask.complete();
                subThread1.join();
                subThread2.join();
                subThread3.join();

                System.out.println("[INFO] All tasks completed successfully!");
            }

            Thread.sleep(3000);

        } finally {
            consolePanelSink.close();
        }
    }

    private static void runSubTask(String name, long duration, ConsolePanelSink sink) {
        DemoTask task = new DemoTask(name, duration);
        try (StatusTracker<DemoTask> tracker = StatusTracker.withInstrumented(
                task,
                java.time.Duration.ofMillis(100),
                List.of(sink))) {

            task.start();
            System.out.println("[INFO] Starting " + name);

            int steps = (int) (duration / 500);
            for (int i = 0; i <= steps; i++) {
                task.updateProgress((double) i / steps);
                Thread.sleep(500);

                if (i == steps / 2) {
                    System.out.println("[DEBUG] " + name + " reached 50%");
                }
            }

            task.complete();
            System.out.println("[INFO] Completed " + name);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

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

        void updateProgress(double progress) {
            this.progress = progress;
        }

        void complete() {
            this.progress = 1.0;
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