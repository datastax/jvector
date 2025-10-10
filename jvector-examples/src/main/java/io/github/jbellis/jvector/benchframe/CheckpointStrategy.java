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

package io.github.jbellis.jvector.benchframe;

import io.github.jbellis.jvector.example.util.CheckpointManager;

import java.util.Collections;
import java.util.List;

/**
 * Strategy interface for managing benchmark checkpointing. Implements the Strategy pattern
 * to enable resumable benchmark execution after failures or interruptions.
 * <p>
 * Checkpointing is particularly valuable for long-running benchmarks in CI/CD environments
 * where resource limits or transient failures may interrupt execution. By tracking which
 * datasets have been completed, benchmarks can resume from where they left off.
 * <p>
 * Two implementations are provided:
 * <ul>
 *   <li>{@link NoCheckpointing} - no-op implementation for simple scenarios</li>
 *   <li>{@link FileCheckpointing} - persistent file-based checkpointing using JSON</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // No checkpointing (default)
 * CheckpointStrategy strategy = CheckpointStrategy.none();
 *
 * // File-based checkpointing
 * CheckpointStrategy strategy = CheckpointStrategy.fileBasedCheckpointing("results/checkpoint");
 *
 * // Custom implementation
 * CheckpointStrategy strategy = new CheckpointStrategy() {
 *     @Override
 *     public boolean shouldSkipDataset(String datasetName) {
 *         // Check database or cache
 *         return completedDatasets.contains(datasetName);
 *     }
 *
 *     @Override
 *     public void recordCompletion(String datasetName, List<BenchResult> results) {
 *         // Update database or cache
 *         completedDatasets.add(datasetName);
 *     }
 *
 *     @Override
 *     public List<BenchResult> getPreviousResults() {
 *         // Load from database or cache
 *         return loadPreviousResults();
 *     }
 * };
 * }</pre>
 *
 * @see BenchFrame.Builder#withCheckpointStrategy(CheckpointStrategy)
 * @see BenchResult
 */
public interface CheckpointStrategy {
    /**
     * Checks if a dataset should be skipped because it has already been completed.
     * This is called before attempting to benchmark each dataset.
     *
     * @param datasetName the name of the dataset to check
     * @return true if the dataset has already been completed and should be skipped, false otherwise
     */
    boolean shouldSkipDataset(String datasetName);

    /**
     * Records the completion of a dataset with its results. This is called after successfully
     * benchmarking a dataset. Implementations should persist this information to enable resumption.
     *
     * @param datasetName the name of the completed dataset
     * @param results the benchmark results for this dataset
     */
    void recordCompletion(String datasetName, List<BenchResult> results);

    /**
     * Retrieves any previously completed results from earlier runs. These results are included
     * in the final output to provide a complete view across multiple executions.
     *
     * @return list of results from previous runs, or empty list if none exist
     */
    List<BenchResult> getPreviousResults();

    /**
     * Creates a no-op checkpoint strategy that does not track or resume progress.
     * This is the default for simple benchmark scenarios.
     *
     * @return a checkpoint strategy that performs no checkpointing
     */
    static CheckpointStrategy none() {
        return new NoCheckpointing();
    }

    /**
     * Creates a file-based checkpoint strategy that persists progress to JSON files.
     * Creates files at {@code outputPath.checkpoint.json} containing completed dataset
     * names and their results.
     *
     * @param outputPath base path for checkpoint file (e.g., "results/benchmark")
     * @return a checkpoint strategy using file-based persistence
     * @see FileCheckpointing
     */
    static CheckpointStrategy fileBasedCheckpointing(String outputPath) {
        return new FileCheckpointing(outputPath);
    }

    /**
     * No-op implementation that performs no checkpointing. All datasets are processed
     * on every run without tracking completion state.
     */
    class NoCheckpointing implements CheckpointStrategy {
        @Override
        public boolean shouldSkipDataset(String datasetName) {
            return false;
        }

        @Override
        public void recordCompletion(String datasetName, List<BenchResult> results) {
            // Do nothing
        }

        @Override
        public List<BenchResult> getPreviousResults() {
            return Collections.emptyList();
        }
    }

    /**
     * File-based implementation that uses {@link CheckpointManager} for persistent checkpointing.
     * Stores checkpoint state in a JSON file at {@code outputPath.checkpoint.json}.
     * <p>
     * The checkpoint file contains:
     * <ul>
     *   <li>List of completed dataset names</li>
     *   <li>All benchmark results from completed datasets</li>
     *   <li>Timestamp of last update</li>
     * </ul>
     * <p>
     * On initialization, loads any existing checkpoint file to resume from previous runs.
     */
    class FileCheckpointing implements CheckpointStrategy {
        private final CheckpointManager manager;

        public FileCheckpointing(String outputPath) {
            this.manager = new CheckpointManager(outputPath);
        }

        @Override
        public boolean shouldSkipDataset(String datasetName) {
            return manager.isDatasetCompleted(datasetName);
        }

        @Override
        public void recordCompletion(String datasetName, List<BenchResult> results) {
            manager.markDatasetCompleted(datasetName, results);
        }

        @Override
        public List<BenchResult> getPreviousResults() {
            return manager.getCompletedResults();
        }
    }
}
