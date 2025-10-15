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

package io.github.jbellis.jvector.status;

import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * JMH benchmarks characterizing the per-iteration overhead of status tracking.
 *
 * <p>Uses @Setup/@TearDown to amortize context lifecycle costs and measure
 * only the actual progress tracking overhead during task execution.
 *
 * <p>Benchmarks compare:
 * <ul>
 *   <li>Baseline: No tracking</li>
 *   <li>Efficient pattern: Volatile long increment</li>
 *   <li>Inefficient pattern: Calculate fraction every iteration</li>
 *   <li>Batched: Update every N iterations</li>
 *   <li>Atomic: AtomicLong for parallel safety</li>
 *   <li>Different poll intervals: 100ms, 500ms, 1000ms</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class StatusTrackingOverheadBenchmark {

    // Baseline: no tracking
    @State(Scope.Thread)
    public static class BaselineState {
        @Param({"100000", "1000000", "10000000"})
        int iterations;

        long itemsProcessed = 0;

        public void process(Blackhole bh) {
            for (int i = 0; i < iterations; i++) {
                bh.consume(i * 2);
                itemsProcessed++;
            }
        }
    }

    // Efficient pattern with volatile long
    @State(Scope.Thread)
    public static class EfficientState {
        @Param({"100000", "1000000", "10000000"})
        int iterations;

        @Param({"100", "500", "1000"})
        int pollIntervalMs;

        StatusContext context;
        StatusTracker<Task> tracker;
        Task task;

        static class Task implements StatusSource<Task> {
            private volatile long itemsProcessed = 0;
            private volatile RunState state = RunState.PENDING;
            private final long total;

            Task(long total) {
                this.total = total;
            }

            @Override
            public StatusUpdate<Task> getTaskStatus() {
                double progress = (double) itemsProcessed / total;
                return new StatusUpdate<>(progress, state, this);
            }

            public void process(Blackhole bh, int count) {
                state = RunState.RUNNING;
                for (int i = 0; i < count; i++) {
                    bh.consume(i * 2);
                    itemsProcessed++;  // Just increment
                }
                state = RunState.SUCCESS;
            }

            public void reset() {
                itemsProcessed = 0;
                state = RunState.PENDING;
            }
        }

        @Setup(Level.Trial)
        public void setup() {
            context = new StatusContext("bench", Duration.ofMillis(pollIntervalMs));
            task = new Task(iterations);
            tracker = context.track(task);
        }

        @TearDown(Level.Trial)
        public void teardown() {
            tracker.close();
            context.close();
        }

        @Setup(Level.Invocation)
        public void resetTask() {
            task.reset();
        }
    }

    // Inefficient pattern: division every iteration
    @State(Scope.Thread)
    public static class InefficientState {
        @Param({"100000", "1000000", "10000000"})
        int iterations;

        StatusContext context;
        StatusTracker<Task> tracker;
        Task task;

        static class Task implements StatusSource<Task> {
            private volatile double progress = 0.0;
            private volatile RunState state = RunState.PENDING;
            private final long total;

            Task(long total) {
                this.total = total;
            }

            @Override
            public StatusUpdate<Task> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            public void process(Blackhole bh, int count) {
                state = RunState.RUNNING;
                for (int i = 0; i < count; i++) {
                    bh.consume(i * 2);
                    progress = (double) (i + 1) / total;  // Division every iteration
                }
                state = RunState.SUCCESS;
            }

            public void reset() {
                progress = 0.0;
                state = RunState.PENDING;
            }
        }

        @Setup(Level.Trial)
        public void setup() {
            context = new StatusContext("bench", Duration.ofMillis(100));
            task = new Task(iterations);
            tracker = context.track(task);
        }

        @TearDown(Level.Trial)
        public void teardown() {
            tracker.close();
            context.close();
        }

        @Setup(Level.Invocation)
        public void resetTask() {
            task.reset();
        }
    }

    // Batched updates
    @State(Scope.Thread)
    public static class BatchedState {
        @Param({"100000", "1000000", "10000000"})
        int iterations;

        @Param({"100", "10"})  // 1% and 10% batching
        int batchesPerTask;

        StatusContext context;
        StatusTracker<Task> tracker;
        Task task;
        long batchSize;

        static class Task implements StatusSource<Task> {
            private volatile long itemsProcessed = 0;
            private volatile RunState state = RunState.PENDING;
            private final long total;

            Task(long total) {
                this.total = total;
            }

            @Override
            public StatusUpdate<Task> getTaskStatus() {
                double progress = (double) itemsProcessed / total;
                return new StatusUpdate<>(progress, state, this);
            }

            public void process(Blackhole bh, int count, long batchSize) {
                state = RunState.RUNNING;
                long batchCount = 0;

                for (int i = 0; i < count; i++) {
                    bh.consume(i * 2);
                    batchCount++;

                    if (batchCount >= batchSize) {
                        itemsProcessed += batchCount;
                        batchCount = 0;
                    }
                }

                itemsProcessed += batchCount;
                state = RunState.SUCCESS;
            }

            public void reset() {
                itemsProcessed = 0;
                state = RunState.PENDING;
            }
        }

        @Setup(Level.Trial)
        public void setup() {
            context = new StatusContext("bench", Duration.ofMillis(100));
            task = new Task(iterations);
            tracker = context.track(task);
            batchSize = Math.max(1, iterations / batchesPerTask);
        }

        @TearDown(Level.Trial)
        public void teardown() {
            tracker.close();
            context.close();
        }

        @Setup(Level.Invocation)
        public void resetTask() {
            task.reset();
        }
    }

    // AtomicLong for parallel safety
    @State(Scope.Thread)
    public static class AtomicState {
        @Param({"100000", "1000000", "10000000"})
        int iterations;

        StatusContext context;
        StatusTracker<Task> tracker;
        Task task;

        static class Task implements StatusSource<Task> {
            private final AtomicLong itemsProcessed = new AtomicLong(0);
            private volatile RunState state = RunState.PENDING;
            private final long total;

            Task(long total) {
                this.total = total;
            }

            @Override
            public StatusUpdate<Task> getTaskStatus() {
                double progress = (double) itemsProcessed.get() / total;
                return new StatusUpdate<>(progress, state, this);
            }

            public void process(Blackhole bh, int count) {
                state = RunState.RUNNING;
                for (int i = 0; i < count; i++) {
                    bh.consume(i * 2);
                    itemsProcessed.incrementAndGet();
                }
                state = RunState.SUCCESS;
            }

            public void reset() {
                itemsProcessed.set(0);
                state = RunState.PENDING;
            }
        }

        @Setup(Level.Trial)
        public void setup() {
            context = new StatusContext("bench", Duration.ofMillis(100));
            task = new Task(iterations);
            tracker = context.track(task);
        }

        @TearDown(Level.Trial)
        public void teardown() {
            tracker.close();
            context.close();
        }

        @Setup(Level.Invocation)
        public void resetTask() {
            task.reset();
        }
    }

    // ==================== BASELINE BENCHMARKS ====================

    /**
     * Baseline: No tracking overhead - pure task execution cost.
     * Use this to calculate relative slowdown of tracked versions.
     */
    @Benchmark
    public void baseline_noTracking(BaselineState state, Blackhole bh) {
        state.process(bh);
    }

    // ==================== TRACKED BENCHMARKS ====================

    /**
     * Efficient pattern with volatile long increment.
     * Compare against baseline_noTracking to see volatile write overhead.
     */
    @Benchmark
    public void tracked_efficient(EfficientState state, Blackhole bh) {
        state.task.process(bh, state.iterations);
    }

    /**
     * Inefficient pattern: division every iteration.
     * Compare against baseline_noTracking to see division overhead.
     */
    @Benchmark
    public void tracked_inefficient(InefficientState state, Blackhole bh) {
        state.task.process(bh, state.iterations);
    }

    /**
     * Batched updates: Only update every Nth iteration.
     * Compare against baseline_noTracking to see batching overhead.
     * This should show the LOWEST overhead of all tracked variants.
     */
    @Benchmark
    public void tracked_batched(BatchedState state, Blackhole bh) {
        state.task.process(bh, state.iterations, state.batchSize);
    }

    /**
     * AtomicLong for thread-safe operations.
     * Compare against baseline_noTracking to see atomic operation overhead.
     */
    @Benchmark
    public void tracked_atomic(AtomicState state, Blackhole bh) {
        state.task.process(bh, state.iterations);
    }

    // ==================== RELATIVE SLOWDOWN COMPARISONS ====================

    /**
     * Calculate relative slowdown by comparing results:
     *
     * Relative Slowdown = (tracked_time / baseline_time) - 1
     *
     * Example from previous run @ 10M iterations:
     * - Baseline: 0.766ms
     * - Batched (1%): 2.975ms → (2.975 / 0.766) - 1 = 2.88x = 288% slowdown
     * - Batched (10%): 3.026ms → (3.026 / 0.766) - 1 = 2.95x = 295% slowdown
     * - Efficient: 44.353ms → (44.353 / 0.766) - 1 = 56.9x = 5690% slowdown
     * - Inefficient: 3.911ms → (3.911 / 0.766) - 1 = 4.10x = 410% slowdown
     * - Atomic: 19.963ms → (19.963 / 0.766) - 1 = 25.0x = 2500% slowdown
     *
     * Key Findings:
     * 1. Batching (1% or 10%): ~3-4x slowdown - RECOMMENDED
     * 2. Inefficient (division): ~4-5x slowdown - Acceptable for <1M iterations
     * 3. Atomic: ~20-25x slowdown - Only use when thread safety required
     * 4. Efficient (volatile): ~50-60x slowdown - AVOID for hot loops!
     *
     * Conclusion: Use batching for minimal overhead (3-4x vs baseline)
     */

    /**
     * Main method to run benchmarks from IDE or command line.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}

