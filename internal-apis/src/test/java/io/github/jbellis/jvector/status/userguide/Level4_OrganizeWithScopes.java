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
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataProcessor;
import io.github.jbellis.jvector.status.userguide.fauxtasks.DataValidator;

/**
 * Level 4: Organize with Scopes
 *
 * <p>Building on Level 3: Group related tasks using explicit scopes.
 *
 * <h2>Key Differences from Level 3:</h2>
 * <ul>
 *   <li><strong>NEW:</strong> Explicitly create StatusScopes to organize tasks</li>
 *   <li><strong>NEW:</strong> Use scope.trackTask() instead of context.track()</li>
 *   <li><strong>NEW:</strong> Check scope.isComplete() to coordinate between phases</li>
 *   <li>Hierarchical organization instead of flat task list</li>
 * </ul>
 *
 * <h2>Implementation Details:</h2>
 * <ul>
 *   <li><strong>Scope creation:</strong> Two scopes - "Ingestion" and "Processing"</li>
 *   <li><strong>Task grouping:</strong> Loader + Validator in Ingestion, Processor in Processing</li>
 *   <li><strong>Phase coordination:</strong> Wait for ingestionScope.isComplete() before processing</li>
 *   <li><strong>Sequential execution:</strong> One phase completes before next begins</li>
 * </ul>
 *
 * <h2>Performance Overhead:</h2>
 * <ul>
 *   <li><strong>CPU:</strong> Same as Level 3 - single monitoring thread polls all tasks</li>
 *   <li><strong>Memory:</strong> +~200 bytes per scope (minimal overhead)</li>
 *   <li><strong>Thread count:</strong> Still 1 daemon thread</li>
 *   <li><strong>Scope checking:</strong> isComplete() is O(N) but called infrequently</li>
 *   <li><strong>Overall impact:</strong> Negligible - scopes are lightweight containers</li>
 * </ul>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li>Multi-phase workflows (ETL pipelines, etc.)</li>
 *   <li>Need to coordinate between task groups</li>
 *   <li>Want hierarchical visualization</li>
 *   <li>Logical grouping improves understanding</li>
 * </ul>
 */
public class Level4_OrganizeWithScopes {

    public static void main(String[] args) throws InterruptedException {
        // NEW: Use explicit scopes to organize tasks
        StatusContext ctx4 = new StatusContext("data-pipeline");
        ctx4.addSink(new ConsoleLoggerSink());

        try (StatusScope ingestionScope = ctx4.createScope("Ingestion");
             StatusScope processingScope = ctx4.createScope("Processing")) {

            // Group ingestion tasks together
            DataLoader loader4 = new DataLoader();
            DataValidator validator4 = new DataValidator();

            try (StatusTracker<DataLoader> loaderTracker4 = ingestionScope.trackTask(loader4);
                 StatusTracker<DataValidator> validatorTracker4 = ingestionScope.trackTask(validator4)) {
                loader4.load();       // Tasks execute independently
                validator4.validate();
            }

            // NEW: Wait for ingestion to complete before processing
            while (!ingestionScope.isComplete()) {
                Thread.sleep(10);
            }

            // Group processing tasks together
            DataProcessor processor4 = new DataProcessor();
            try (StatusTracker<DataProcessor> processorTracker4 = processingScope.trackTask(processor4)) {
                processor4.process();
            }
        } finally {
            ctx4.close();
        }
    }
}
