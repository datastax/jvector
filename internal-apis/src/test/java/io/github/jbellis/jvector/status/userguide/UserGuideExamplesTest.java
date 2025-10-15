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
import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataProcessor;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataValidator;
import io.github.jbellis.jvector.status.userguide.fauxtasks.ParallelDataLoader;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification tests for all user guide examples.
 * Ensures that each level example works as described and demonstrates the concepts correctly.
 */
public class UserGuideExamplesTest {

    @Test
    public void testLevel1_AbsoluteMinimum() {
        // Verify Level 1 example works
        DataLoader loader1 = new DataLoader();
        try (StatusTracker<DataLoader> tracker1 = new StatusTracker<>(loader1)) {
            loader1.load();

            // Verify we can access status programmatically
            assertNotNull(tracker1.toString());
            assertTrue(tracker1.toString().contains("DataLoader"));
            assertEquals(RunState.SUCCESS, tracker1.getStatus().runstate);
            assertEquals(1.0, tracker1.getStatus().progress, 0.01);
            assertTrue(tracker1.getElapsedRunningTime() > 0);
        }
    }

    @Test
    public void testLevel2_AddConsoleOutput() {
        // Verify Level 2 example works (same as Level 1 but with sink)
        DataLoader loader2 = new DataLoader();
        try (StatusTracker<DataLoader> tracker2 = new StatusTracker<>(loader2)) {
            // Add sink
            tracker2.getContext().addSink(new ConsoleLoggerSink());

            loader2.load();

            // Verify tracking worked
            assertEquals(RunState.SUCCESS, tracker2.getStatus().runstate);
            assertEquals(1.0, tracker2.getStatus().progress, 0.01);
        }
    }

    @Test
    public void testLevel3_MultipleTasks() {
        // Verify Level 3 example works (shared context)
        StatusContext ctx3 = new StatusContext("batch-processing");
        ctx3.addSink(new ConsoleLoggerSink());

        DataLoader loader3 = new DataLoader();
        DataValidator validator3 = new DataValidator();

        try (StatusTracker<DataLoader> loaderTracker = ctx3.track(loader3);
             StatusTracker<DataValidator> validatorTracker = ctx3.track(validator3)) {

            // Execute tasks
            loader3.load();
            validator3.validate();

            // Verify both completed
            assertEquals(RunState.SUCCESS, loaderTracker.getStatus().runstate);
            assertEquals(RunState.SUCCESS, validatorTracker.getStatus().runstate);
            assertEquals(2, ctx3.getActiveTrackerCount());
        } finally {
            ctx3.close();
        }
    }

    @Test
    public void testLevel4_OrganizeWithScopes() throws InterruptedException {
        // Verify Level 4 example works (explicit scopes)
        StatusContext ctx4 = new StatusContext("data-pipeline");
        ctx4.addSink(new ConsoleLoggerSink());

        try (StatusScope ingestionScope = ctx4.createScope("Ingestion");
             StatusScope processingScope = ctx4.createScope("Processing")) {

            // Ingestion phase
            DataLoader loader4 = new DataLoader();
            DataValidator validator4 = new DataValidator();

            try (StatusTracker<DataLoader> loaderTracker4 = ingestionScope.trackTask(loader4);
                 StatusTracker<DataValidator> validatorTracker4 = ingestionScope.trackTask(validator4)) {
                loader4.load();
                validator4.validate();
            }

            // Wait for ingestion to complete
            while (!ingestionScope.isComplete()) {
                Thread.sleep(10);
            }

            assertTrue(ingestionScope.isComplete());

            // Processing phase
            DataProcessor processor4 = new DataProcessor();
            try (StatusTracker<DataProcessor> processorTracker4 = processingScope.trackTask(processor4)) {
                processor4.process();
                assertEquals(RunState.SUCCESS, processorTracker4.getStatus().runstate);
            }
        } finally {
            ctx4.close();
        }
    }

    @Test
    public void testLevel5_CustomSinks() {
        // Verify Level 5 example works (multiple sinks)
        StatusContext ctx5 = new StatusContext("data-pipeline");

        ctx5.addSink(new ConsoleLoggerSink());
        MetricsStatusSink metrics5 = new MetricsStatusSink();
        ctx5.addSink(metrics5);

        DataLoader loader5 = new DataLoader();

        try (StatusScope scope5 = ctx5.createScope("Work")) {
            try (StatusTracker<DataLoader> tracker5 = scope5.trackTask(loader5)) {
                loader5.load();
            }
        }

        // Verify metrics were collected
        assertEquals(1, metrics5.getTotalTasksStarted());
        assertEquals(1, metrics5.getTotalTasksFinished());
        assertTrue(metrics5.getAverageTaskDuration() > 0);

        ctx5.close();
    }

    @Test
    public void testLevel6_ConfigurePollInterval() {
        // Verify Level 6 example works (custom poll interval)
        StatusContext ctx6 = new StatusContext(
            "data-pipeline",
            Duration.ofMillis(50)); // Will be clamped to 100ms minimum

        ctx6.addSink(new ConsoleLoggerSink());

        // Verify minimum poll interval was enforced
        assertTrue(ctx6.getDefaultPollInterval().toMillis() >= 100,
                "Poll interval should be at least 100ms");

        DataLoader loader6 = new DataLoader();
        try (StatusTracker<DataLoader> tracker6 = ctx6.track(loader6)) {
            loader6.load();
            assertEquals(RunState.SUCCESS, tracker6.getStatus().runstate);
        } finally {
            ctx6.close();
        }
    }

    @Test
    public void testLevel7_ParallelExecution() throws Exception {
        // Verify Level 7 example works (parallel execution)
        StatusContext ctx7 = new StatusContext("parallel-work");
        ctx7.addSink(new ConsoleLoggerSink());

        ExecutorService executor7 = Executors.newFixedThreadPool(3);
        try (StatusScope scope7 = ctx7.createScope("Parallel")) {

            List<Future<?>> futures7 = new ArrayList<>();

            for (int i = 0; i < 3; i++) {
                ParallelDataLoader loader7 = new ParallelDataLoader();
                Future<?> future = executor7.submit(() -> {
                    try (StatusTracker<ParallelDataLoader> tracker7 = scope7.trackTask(loader7)) {
                        loader7.load();
                        assertEquals(RunState.SUCCESS, tracker7.getStatus().runstate);
                    }
                });
                futures7.add(future);
            }

            // Wait for all tasks
            for (Future<?> f : futures7) {
                f.get();
            }

            // Verify scope completion
            while (!scope7.isComplete()) {
                Thread.sleep(10);
            }
            assertTrue(scope7.isComplete());

        } finally {
            executor7.shutdown();
            ctx7.close();
        }
    }

    @Test
    public void testAllLevelExamplesRunSuccessfully() {
        // Meta-test: Verify all individual level tests pass
        // This ensures the user guide examples remain functional
        assertDoesNotThrow(() -> {
            testLevel1_AbsoluteMinimum();
            testLevel2_AddConsoleOutput();
            testLevel3_MultipleTasks();
            testLevel4_OrganizeWithScopes();
            testLevel5_CustomSinks();
            testLevel6_ConfigurePollInterval();
            testLevel7_ParallelExecution();
        }, "All user guide examples should run without errors");
    }
}
