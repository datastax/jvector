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
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.userguide.fauxtasks.ParallelDataLoader;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Level 7: Parallel Execution
 *
 * <p>Building on Level 6: Track tasks running in parallel threads.
 *
 * <h2>Key Differences from Level 6:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> ParallelDataLoader uses AtomicLong instead of volatile long</li>
 *   <li><strong>NEW:</strong> ExecutorService for parallel task execution</li>
 *   <li><strong>NEW:</strong> Multiple threads update same task's progress concurrently</li>
 *   <li>Demonstrates thread-safe progress tracking pattern</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Parallel execution:</strong> 3 tasks run concurrently via ExecutorService</li>
 *   <li><strong>Thread-safe progress:</strong> AtomicLong.incrementAndGet() for lock-free updates</li>
 *   <li><strong>Each task:</strong> Uses Java parallel streams internally</li>
 *   <li><strong>Coordination:</strong> Future.get() waits for all tasks to complete</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> AtomicLong adds ~5-10ns per increment vs volatile (negligible)</li>
 *   <li><strong>Memory:</strong> Same as Level 6 (AtomicLong same size as volatile long)</li>
 *   <li><strong>Thread count:</strong> 1 monitor thread + 3 worker threads from ExecutorService</li>
 *   <li><strong>Contention:</strong> None - AtomicLong uses CAS, lock-free</li>
 *   <li><strong>Scalability:</strong> Excellent - monitor thread handles all tasks efficiently</li>
 *   <li><strong>Overall impact:</strong> Atomic operations add <0.1% overhead to typical workloads</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Parallel/concurrent task execution</li>
 *   <li>Multiple threads update same task's progress</li>
 *   <li>High-throughput scenarios</li>
 *   <li>Need accurate progress from concurrent updates</li>
 * </ul>
 *
 * <h2>Thread Safety Pattern:</h2>
 * <ul>
 *   <li><strong>Single-threaded tasks:</strong> Use volatile long (Levels 1-6)</li>
 *   <li><strong>Multi-threaded tasks:</strong> Use AtomicLong (Level 7)</li>
 *   <li><strong>Rule:</strong> If multiple threads update progress, use atomics</li>
 * </ul>
 */
public class Level7_ParallelExecution {

    public static void main(String[] args) throws Exception {
        StatusContext ctx7 = new StatusContext("parallel-work");
        ctx7.addSink(new ConsoleLoggerSink());

        ExecutorService executor7 = Executors.newFixedThreadPool(3);
        try (StatusScope scope7 = ctx7.createScope("Parallel")) {

            // NEW: Submit tasks to run in parallel
            List<Future<?>> futures7 = new ArrayList<>();

            for (int i = 0; i < 3; i++) {
                ParallelDataLoader loader7 = new ParallelDataLoader();
                Future<?> future = executor7.submit(() -> {
                    try (StatusTracker<ParallelDataLoader> tracker7 = scope7.trackTask(loader7)) {
                        loader7.load(); // Task executes independently
                    } // Tracker auto-closes
                });
                futures7.add(future);
            }

            // Wait for all tasks
            for (Future<?> f : futures7) {
                f.get();
            }
        } finally {
            executor7.shutdown();
            ctx7.close();
        }
    }
}
