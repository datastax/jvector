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

package io.github.jbellis.jvector.status.asyncdag;

import org.junit.jupiter.api.Test;

import java.util.concurrent.*;

public class LayeredTaskTest {

    @Test
    public void testThreeLayerTaskStructure() throws Exception {
        ExecutorService executor = Executors.newCachedThreadPool();

        try {
            TaskLayer task1a = new TaskLayer("compute-1a", 500, "Result-1a");
            TaskLayer task1b = new TaskLayer("compute-1b", 700, "Result-1b");
            TaskLayer task1c = new TaskLayer("compute-1c", 300, "Result-1c");
            TaskLayer task1d = new TaskLayer("compute-1d", 600, "Result-1d");

            TaskLayer task2a = new TaskLayer("process-2a", 2, executor);
            task2a.addSubTask(task1a);
            task2a.addSubTask(task1b);

            TaskLayer task2b = new TaskLayer("process-2b", 2, executor);
            task2b.addSubTask(task1c);
            task2b.addSubTask(task1d);

            TaskLayer task3 = new TaskLayer("orchestrate-3", 3, executor);
            task3.addSubTask(task2a);
            task3.addSubTask(task2b);

            System.out.println("Starting three-layer task execution...\n");

            Future<String> result = executor.submit(task3);

            String finalResult = result.get(10, TimeUnit.SECONDS);

            System.out.println("\n=== Final Result ===");
            System.out.println(finalResult);

        } finally {
            executor.shutdown();
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        }
    }

    @Test
    public void testParallelTaskExecution() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(4);

        try {
            TaskLayer[] layer1Tasks = new TaskLayer[6];
            for (int i = 0; i < 6; i++) {
                layer1Tasks[i] = new TaskLayer(
                    "worker-" + i,
                    200 + (i * 100),
                    "Data-" + i
                );
            }

            TaskLayer task2a = new TaskLayer("aggregator-A", 2, executor);
            task2a.addSubTask(layer1Tasks[0]);
            task2a.addSubTask(layer1Tasks[1]);
            task2a.addSubTask(layer1Tasks[2]);

            TaskLayer task2b = new TaskLayer("aggregator-B", 2, executor);
            task2b.addSubTask(layer1Tasks[3]);
            task2b.addSubTask(layer1Tasks[4]);
            task2b.addSubTask(layer1Tasks[5]);

            TaskLayer masterTask = new TaskLayer("master", 3, executor);
            masterTask.addSubTask(task2a);
            masterTask.addSubTask(task2b);

            System.out.println("Starting parallel task execution with 6 workers...\n");

            long startTime = System.currentTimeMillis();
            Future<String> result = executor.submit(masterTask);
            String finalResult = result.get(15, TimeUnit.SECONDS);
            long endTime = System.currentTimeMillis();

            System.out.println("\n=== Execution Summary ===");
            System.out.println("Total execution time: " + (endTime - startTime) + "ms");
            System.out.println(finalResult);

        } finally {
            executor.shutdown();
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        }
    }
}
