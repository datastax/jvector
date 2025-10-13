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
 * Demo task simulating validation and verification operations.
 * Demonstrates sequential validation stages with nested check operations.
 */
class ExampleValidationTask implements StatusSource<ExampleValidationTask>, Runnable {
    private static final Logger logger = LogManager.getLogger(ExampleValidationTask.class);

    private final String name;
    private final int checkCount;
    private final TrackerScope parentScope;
    private final Random random = new Random();

    private volatile int checksComplete = 0;
    private volatile RunState state = RunState.PENDING;
    private volatile boolean interrupted = false;
    private final List<Thread> childThreads = new ArrayList<>();
    private TrackerScope detailsScope; // Scope for nested validations
    private final boolean allowNesting; // Only top-level tasks spawn nested validations
    private final SimulatedClock clock;

    ExampleValidationTask(String name, int checkCount, TrackerScope parentScope, SimulatedClock clock) {
        this(name, checkCount, parentScope, true, clock);
    }

    ExampleValidationTask(String name, int checkCount, TrackerScope parentScope, boolean allowNesting, SimulatedClock clock) {
        this.name = name;
        this.checkCount = checkCount;
        this.parentScope = parentScope;
        this.allowNesting = allowNesting;
        this.clock = clock;
    }

    @Override
    public void run() {
        try (StatusTracker<ExampleValidationTask> tracker = createTracker()) {
            execute(tracker);
        } catch (Exception e) {
            logger.error("Validation task {} failed: {}", name, e.getMessage());
        }
    }

    private StatusTracker<ExampleValidationTask> createTracker() {
        return parentScope.trackTask(this, ExampleValidationTask::getTaskStatus, Duration.ofMillis(100));
    }

    private void execute(StatusTracker<ExampleValidationTask> tracker) throws InterruptedException {
        state = RunState.RUNNING;
        logger.info("Starting validation: {} ({} checks)", name, checkCount);
        logger.debug("Validation scope: {}, Expected duration: ~{}ms", name, checkCount * 150);
        logger.trace("Loading validation rules for {}", name);

        int warningCount = 0;
        int errorCount = 0;
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < checkCount && !interrupted; i++) {
            checksComplete = i + 1;

            logger.trace("Executing validation check #{} of {}", i + 1, checkCount);

            // Simulate validation check
            clock.sleep(100 + random.nextInt(100));

            if (i % 3 == 0) {
                logger.debug("Validation check {}/{} passed in {}", checksComplete, checkCount, name);
            }

            if (i % 5 == 0 && i > 0) {
                double progress = (checksComplete * 100.0) / checkCount;
                logger.debug("Validation progress: {:.1f}% ({}/{})", progress, checksComplete, checkCount);
            }

            if (i % 10 == 0 && i > 0) {
                logger.info("Validation milestone: {} checks completed ({}% done)",
                           checksComplete, (checksComplete * 100) / checkCount);
            }

            // Spawn nested validation at 50% mark (only for top-level validation tasks)
            if (i == checkCount / 2 && allowNesting && detailsScope == null) {
                logger.info("Spawning detailed validation subtasks for {}", name);
                logger.debug("Creating 3 specialized validation tasks");
                detailsScope = parentScope.createChildScope(name + "-Details");
                spawnDetailedValidation(tracker, "Schema", 10);
                spawnDetailedValidation(tracker, "Integrity", 12);
                spawnDetailedValidation(tracker, "Consistency", 8);
                logger.trace("Nested validation tasks scheduled");
            }

            // Simulate various validation outcomes
            int rand = random.nextInt(100);
            if (rand > 95) {
                warningCount++;
                logger.warn("Validation warning in {} at check {}: minor inconsistency detected (total warnings: {})",
                           name, i, warningCount);
                logger.trace("Warning details: threshold exceeded by 5%");
            } else if (rand > 92) {
                logger.debug("Check {} passed with constraints", i);
            } else if (rand > 89) {
                logger.trace("Cached validation result reused for check {}", i);
            } else if (rand > 86) {
                logger.info("Complex validation rule #{} evaluated successfully", i);
            }

            if (random.nextInt(100) > 98) {
                errorCount++;
                logger.error("Validation error in {} at check {}: retrying (error count: {})", name, i, errorCount);
                logger.debug("Attempting retry with relaxed constraints");
                clock.sleep(200); // Simulate retry delay
                logger.debug("Retry succeeded for check {}", i);
            }

            if (i > 0 && i % 20 == 0) {
                long elapsed = System.currentTimeMillis() - startTime;
                double rate = checksComplete / (elapsed / 1000.0);
                logger.debug("Validation rate: {:.1f} checks/sec, Warnings: {}, Errors: {}",
                           rate, warningCount, errorCount);
            }
        }

        // Wait for nested validations and close the details scope
        if (!childThreads.isEmpty()) {
            logger.debug("Waiting for {} nested validation tasks to complete", childThreads.size());
            for (Thread child : childThreads) {
                logger.trace("Joining validation thread: {}", child.getName());
                child.join(2000);
            }
            logger.info("All nested validations completed for {}", name);

            // Close the details scope to clean up resources
            if (detailsScope != null) {
                detailsScope.close();
                logger.debug("Details scope closed for {}", name);
            }
        }

        if (interrupted) {
            logger.warn("Validation {} interrupted at check {}/{}", name, checksComplete, checkCount);
            logger.debug("Partial results: {} checks passed, {} warnings, {} errors",
                        checksComplete, warningCount, errorCount);
            state = RunState.CANCELLED;
        } else {
            checksComplete = checkCount;
            state = RunState.SUCCESS;
            long elapsed = System.currentTimeMillis() - startTime;
            logger.info("✓ Completed validation: {} ({} checks passed)", name, checkCount);
            logger.debug("Validation summary - Duration: {}ms, Warnings: {}, Errors: {}, Success rate: {:.1f}%",
                        elapsed, warningCount, errorCount,
                        ((checkCount - errorCount) * 100.0) / checkCount);
            logger.trace("Validation state persisted for {}", name);
        }
    }

    private void spawnDetailedValidation(StatusTracker<ExampleValidationTask> tracker, String checkType, int checks) {
        // Use the shared details scope created earlier, disable nesting for child tasks
        ExampleValidationTask childTask = new ExampleValidationTask(checkType, checks, detailsScope, false, clock);
        Thread childThread = new Thread(childTask);
        childThreads.add(childThread);
        childThread.start();
    }

    public void interrupt() {
        interrupted = true;
        for (Thread child : childThreads) {
            child.interrupt();
        }
    }

    public String getName() {
        return name;
    }

    @Override
    public StatusUpdate<ExampleValidationTask> getTaskStatus() {
        double progress = checkCount > 0 ? (double) checksComplete / checkCount : 0.0;
        return new StatusUpdate<>(progress, state, this);
    }

    @Override
    public String toString() {
        return name;
    }
}
