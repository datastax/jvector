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
import io.github.jbellis.jvector.status.StatusScope;
import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.Random;

/**
 * Demo task simulating data processing operations (loading, parsing, transforming).
 * Demonstrates I/O-heavy workloads with periodic status updates as a leaf task within a scope.
 */
class ExampleDataProcessingTask implements StatusSource<ExampleDataProcessingTask>, Runnable {
    private static final Logger logger = LogManager.getLogger(ExampleDataProcessingTask.class);

    private final String name;
    private final int recordCount;
    private final StatusScope parentScope;
    private final SimulatedClock clock;
    private final Random random = new Random();

    private volatile int recordsProcessed = 0;
    private volatile RunState state = RunState.PENDING;
    private volatile boolean interrupted = false;

    ExampleDataProcessingTask(String name, int recordCount, StatusScope parentScope, SimulatedClock clock) {
        this.name = name;
        this.recordCount = recordCount;
        this.parentScope = parentScope;
        this.clock = clock;
    }

    @Override
    public void run() {
        try (StatusTracker<ExampleDataProcessingTask> tracker = createTracker()) {
            execute();
        } catch (Exception e) {
            logger.error("Data processing task {} failed: {}", name, e.getMessage());
        }
    }

    private StatusTracker<ExampleDataProcessingTask> createTracker() {
        return parentScope.trackTask(this, ExampleDataProcessingTask::getTaskStatus);
    }

    private void execute() throws InterruptedException {
        state = RunState.RUNNING;
        logger.info("Starting data processing: {} ({} records)", name, recordCount);
        logger.debug("Initializing data pipeline for {}", name);
        logger.trace("Allocating buffer space for {} records", recordCount);

        long startTime = System.currentTimeMillis();
        int batchSize = 50;
        int currentBatch = 0;

        for (int i = 0; i < recordCount && !interrupted; i++) {
            recordsProcessed = i + 1;

            if (i % batchSize == 0) {
                currentBatch++;
                logger.debug("Processing batch {}/{} (records {}-{})",
                           currentBatch, (recordCount + batchSize - 1) / batchSize,
                           i, Math.min(i + batchSize - 1, recordCount - 1));
                logger.trace("Batch {} memory footprint: ~{}KB", currentBatch, random.nextInt(500) + 100);
            }

            // Simulate I/O delay
            clock.sleep(50 + random.nextInt(50));

            if (i % 10 == 0 && i > 0) {
                logger.trace("Record {} parsed and validated", i);
            }

            if (i % 50 == 0 && i > 0) {
                double progress = (recordsProcessed * 100.0) / recordCount;
                logger.debug("Progress: {}/{} records ({:.1f}%)", recordsProcessed, recordCount, progress);
            }

            if (i % 100 == 0 && i > 0) {
                logger.info("Processed {}/{} records in {}", recordsProcessed, recordCount, name);
                long elapsed = System.currentTimeMillis() - startTime;
                double rate = recordsProcessed / (elapsed / 1000.0);
                logger.debug("Processing rate: {:.1f} records/sec", rate);
            }

            if (i % 250 == 0 && random.nextInt(10) > 7) {
                logger.info("Batch checkpoint: {} records processed successfully", recordsProcessed);
                logger.trace("Checkpoint saved to offset: {}", i);
            }

            // Simulate various data processing events
            int rand = random.nextInt(1000);
            if (rand > 995) {
                logger.warn("Slow record detected in {} at index {} (processing took >200ms)", name, i);
            } else if (rand > 990) {
                logger.debug("Data validation passed for record {}", i);
            } else if (rand > 985) {
                logger.trace("Schema version mismatch handled for record {}", i);
            } else if (rand > 980) {
                logger.info("Transformation pipeline applied to record {}", i);
            } else if (rand > 975) {
                logger.debug("Record {} required additional normalization", i);
            } else if (rand > 970) {
                logger.trace("Cache hit for lookup key at record {}", i);
            }

            // Simulate batch commits
            if (i > 0 && i % 200 == 0) {
                logger.info("Committing batch of {} records to storage", 200);
                logger.debug("Transaction ID: {}", random.nextInt(100000));
                logger.trace("Fsync completed for batch ending at record {}", i);
            }
        }

        if (interrupted) {
            logger.warn("Data processing {} interrupted at record {}/{}", name, recordsProcessed, recordCount);
            logger.debug("Performing rollback for incomplete batch");
            logger.trace("Cleaning up partial state at record {}", recordsProcessed);
            state = RunState.CANCELLED;
        } else {
            recordsProcessed = recordCount;
            state = RunState.SUCCESS;
            long elapsed = System.currentTimeMillis() - startTime;
            double rate = recordCount / (elapsed / 1000.0);
            logger.info("✓ Completed data processing: {} ({} records)", name, recordCount);
            logger.debug("Final stats - Total: {} records, Duration: {}ms, Rate: {:.1f} rec/sec",
                        recordCount, elapsed, rate);
            logger.trace("Pipeline resources released for task {}", name);
        }
    }

    public void interrupt() {
        interrupted = true;
    }

    public String getName() {
        return name;
    }

    @Override
    public StatusUpdate<ExampleDataProcessingTask> getTaskStatus() {
        double progress = recordCount > 0 ? (double) recordsProcessed / recordCount : 0.0;
        return new StatusUpdate<>(progress, state, this);
    }

    @Override
    public String toString() {
        return name;
    }
}
