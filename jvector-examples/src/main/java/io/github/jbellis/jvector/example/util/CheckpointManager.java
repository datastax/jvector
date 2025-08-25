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
package io.github.jbellis.jvector.example.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Manages checkpoint files for AutoBenchYAML to enable resuming benchmarks after failures.
 * Checkpoints are stored as JSON files in the same directory as the output files.
 */
public class CheckpointManager {
    private static final Logger logger = LoggerFactory.getLogger(CheckpointManager.class);
    private final String checkpointPath;
    private final ObjectMapper mapper;
    private final Set<String> completedDatasets;
    private final List<io.github.jbellis.jvector.example.BenchResult> completedResults;

    /**
     * Creates a new CheckpointManager for the given output path.
     * The checkpoint file will be stored at outputPath + ".checkpoint.json".
     *
     * @param outputPath The base path for output files
     */
    public CheckpointManager(String outputPath) {
        this.checkpointPath = outputPath + ".checkpoint.json";
        this.mapper = new ObjectMapper();
        this.completedDatasets = new HashSet<>();
        this.completedResults = new ArrayList<>();
        loadCheckpoint();
    }

    /**
     * Loads the checkpoint file if it exists.
     */
    private void loadCheckpoint() {
        File checkpointFile = new File(checkpointPath);
        if (checkpointFile.exists()) {
            try {
                CheckpointData checkpointData = mapper.readValue(checkpointFile, CheckpointData.class);
                if (checkpointData.getCompletedDatasets() != null)
                    completedDatasets.addAll(checkpointData.getCompletedDatasets());
                if (checkpointData.getCompletedResults() != null)
                    completedResults.addAll(checkpointData.getCompletedResults());
                logger.info("Loaded checkpoint file: {} with {} completed datasets and {} results", checkpointPath, completedDatasets.size(), completedResults.size());
            } catch (IOException e) {
                logger.error("Error loading checkpoint file: {}", checkpointPath, e);
            }
        } else {
            logger.info("No checkpoint file found at: {}", checkpointPath);
        }
    }

    /**
     * Checks if a dataset has already been processed.
     *
     * @param datasetName The name of the dataset
     * @return true if the dataset has been processed, false otherwise
     */
    public boolean isDatasetCompleted(String datasetName) {
        return completedDatasets.contains(datasetName);
    }

    /**
     * Marks a dataset as completed and updates the checkpoint file.
     *
     * @param datasetName The name of the dataset
     * @param resultsForDataset The results for the dataset
     */
    public void markDatasetCompleted(String datasetName, List<io.github.jbellis.jvector.example.BenchResult> resultsForDataset) {
        completedDatasets.add(datasetName);
        if (resultsForDataset != null) {
            completedResults.addAll(resultsForDataset);
        }
        saveCheckpoint();
        logger.info("Marked dataset as completed and updated checkpoint: {}", datasetName);
    }

    /**
     * Saves the current checkpoint state to the checkpoint file.
     */
    private void saveCheckpoint() {
        File checkpointFile = new File(checkpointPath);
        try {
            CheckpointData checkpointData = new CheckpointData(new ArrayList<>(completedDatasets), new ArrayList<>(completedResults));
            mapper.writerWithDefaultPrettyPrinter().writeValue(checkpointFile, checkpointData);
            logger.info("Saved checkpoint file: {}", checkpointPath);
        } catch (IOException e) {
            logger.error("Error saving checkpoint file: {}", checkpointPath, e);
        }
    }

    /**
     * Returns the list of completed datasets.
     *
     * @return The list of completed datasets
     */
    public Set<String> getCompletedDatasets() {
        return new HashSet<>(completedDatasets);
    }

    /**
     * Returns the list of completed BenchResults.
     */
    public List<io.github.jbellis.jvector.example.BenchResult> getCompletedResults() {
        return new ArrayList<>(completedResults);
    }

    /**
     * Data class for storing checkpoint information.
     */
    private static class CheckpointData {
        private List<String> completedDatasets;
        private List<io.github.jbellis.jvector.example.BenchResult> completedResults;

        public CheckpointData() {
            // Default constructor for Jackson
        }

        public CheckpointData(List<String> completedDatasets, List<io.github.jbellis.jvector.example.BenchResult> completedResults) {
            this.completedDatasets = completedDatasets;
            this.completedResults = completedResults;
        }

        public List<String> getCompletedDatasets() {
            return completedDatasets;
        }

        public void setCompletedDatasets(List<String> completedDatasets) {
            this.completedDatasets = completedDatasets;
        }

        public List<io.github.jbellis.jvector.example.BenchResult> getCompletedResults() {
            return completedResults;
        }

        public void setCompletedResults(List<io.github.jbellis.jvector.example.BenchResult> completedResults) {
            this.completedResults = completedResults;
        }
    }
}