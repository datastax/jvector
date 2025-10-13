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
import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;

import java.time.Duration;

/**
 * Example showing the new scope-based organization vs traditional tracker hierarchy.
 */
public class ScopeUsageExample {

    public static void main(String[] args) throws InterruptedException {
        demonstrateScopes();
    }

    /**
     * NEW APPROACH: Use scopes for organization, tasks for work.
     * Scopes have no progress/state - they're purely organizational.
     * Tasks within scopes CANNOT have children (enforced).
     */
    private static void demonstrateScopes() throws InterruptedException {
        SimulatedClock clock = new SimulatedClock();
        try (StatusContext context = new StatusContext("pipeline",
                                                       Duration.ofMillis(100),
                                                       java.util.List.of(new ConsoleLoggerSink()))) {

            // Create organizational scopes
            try (TrackerScope ingestionScope = context.createScope("Ingestion");
                 TrackerScope processingScope = context.createScope("Processing")) {

                // Track actual work as leaf tasks
                StatusTracker<ExampleDataProcessingTask> loadTracker =
                    ingestionScope.trackTask(new ExampleDataProcessingTask("LoadCSV", 100, ingestionScope, clock));

                StatusTracker<ExampleValidationTask> validateTracker =
                    ingestionScope.trackTask(new ExampleValidationTask("ValidateSchema", 20, ingestionScope, clock));

                // ComputeTask creates its own scope hierarchy internally
                Thread transformThread = new Thread(new ExampleComputeTask("Transform", 80, 2, processingScope, clock));
                transformThread.start();

                // Tasks execute...
                Thread.sleep(500);

                // ComputeTask creates its own scope hierarchy internally
                Thread indexThread = new Thread(new ExampleComputeTask("BuildIndex", 120, 3, processingScope, clock));
                indexThread.start();

                System.out.println("\nScope hierarchy:");
                System.out.println("  Ingestion (scope)");
                System.out.println("    ├─ LoadCSV (task)");
                System.out.println("    └─ ValidateSchema (task)");
                System.out.println("  Processing (scope)");
                System.out.println("    ├─ Transform (scope)");
                System.out.println("    │   ├─ Transform (main task)");
                System.out.println("    │   └─ Workers (scope)");
                System.out.println("    │       ├─ Worker1 (task)");
                System.out.println("    │       └─ Worker2 (task)");
                System.out.println("    └─ BuildIndex (scope)");
                System.out.println("        ├─ BuildIndex (main task)");
                System.out.println("        └─ Workers (scope)");
                System.out.println("            ├─ Worker1 (task)");
                System.out.println("            ├─ Worker2 (task)");
                System.out.println("            └─ Worker3 (task)");

                // Check completion
                System.out.println("\nIs ingestion scope complete? " + ingestionScope.isComplete());
                System.out.println("Is processing scope complete? " + processingScope.isComplete());

                // Close trackers
                loadTracker.close();
                validateTracker.close();

                // Wait for compute tasks to complete
                transformThread.join(5000);
                indexThread.join(5000);

                Thread.sleep(100);

                System.out.println("\nAfter closing all tasks:");
                System.out.println("Is ingestion scope complete? " + ingestionScope.isComplete());
                System.out.println("Is processing scope complete? " + processingScope.isComplete());
            }
        }
    }
}
