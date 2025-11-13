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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer;
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer.SummaryStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Strategy interface for handling benchmark results after execution completes.
 * Implements the Strategy pattern to decouple result handling from benchmark execution.
 * <p>
 * This functional interface supports various output modes including:
 * <ul>
 *   <li>Console-only output (Grid handles printing)</li>
 *   <li>File-based output (CSV summary and JSON details)</li>
 *   <li>Combined output to multiple destinations</li>
 *   <li>Custom implementations for specialized scenarios</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 * <pre>{@code
 * // Console only (default)
 * ResultHandler handler = ResultHandler.consoleOnly();
 *
 * // Write to files
 * ResultHandler handler = ResultHandler.toFiles("results/benchmark");
 *
 * // Combine multiple handlers
 * ResultHandler handler = ResultHandler.combining(
 *     ResultHandler.consoleOnly(),
 *     ResultHandler.toFiles("results/benchmark")
 * );
 *
 * // Custom implementation
 * ResultHandler handler = results -> {
 *     // Send to monitoring system
 *     monitoringService.recordBenchmarks(results);
 *     // Upload to cloud storage
 *     cloudStorage.upload("benchmarks", results);
 * };
 * }</pre>
 *
 * @see BenchResult
 * @see BenchFrame.Builder#withResultHandler(ResultHandler)
 */
@FunctionalInterface
public interface ResultHandler {
    /**
     * Handles the benchmark results after execution completes. Implementations may write
     * to files, send to external systems, or perform other processing.
     *
     * @param results list of benchmark results to handle
     * @throws IOException if output or I/O operations fail
     */
    void handleResults(List<BenchResult> results) throws IOException;

    /**
     * Creates a no-op result handler that does nothing with results. Console output
     * is already handled by Grid during benchmark execution.
     * This matches the behavior of the original Bench.java and BenchYAML.java.
     *
     * @return a result handler that performs no additional output
     */
    static ResultHandler consoleOnly() {
        return results -> {
            // Grid already printed results to console, nothing to do
        };
    }

    /**
     * Creates a result handler that writes results to CSV summary and JSON detail files.
     * This matches the behavior of AutoBenchYAML.java.
     * <p>
     * Files created:
     * <ul>
     *   <li>{@code outputBasePath.csv} - CSV summary with aggregate statistics per dataset</li>
     *   <li>{@code outputBasePath.json} - JSON file with complete detailed results</li>
     * </ul>
     * <p>
     * The CSV file contains columns: dataset, QPS, QPS StdDev, Mean Latency, Recall@10,
     * Index Construction Time.
     *
     * @param outputBasePath base path for output files (without extension)
     * @return a result handler that writes to CSV and JSON files
     * @see FileOutputHandler
     */
    static ResultHandler toFiles(String outputBasePath) {
        return new FileOutputHandler(outputBasePath);
    }

    /**
     * Implementation that writes benchmark results to CSV summary and JSON details files.
     * Uses {@link BenchmarkSummarizer} to calculate aggregate statistics across multiple
     * benchmark runs.
     */
    class FileOutputHandler implements ResultHandler {
        private static final Logger logger = LoggerFactory.getLogger(FileOutputHandler.class);
        private final String outputBasePath;

        public FileOutputHandler(String outputBasePath) {
            this.outputBasePath = outputBasePath;
        }

        @Override
        public void handleResults(List<BenchResult> results) throws IOException {
            if (results.isEmpty()) {
                logger.warn("No results to write");
                return;
            }

            // Calculate summary statistics
            SummaryStats stats = BenchmarkSummarizer.summarize(results);
            logger.info("Benchmark summary: {}", stats.toString());

            // Write detailed results to JSON
            File detailsFile = new File(outputBasePath + ".json");
            ObjectMapper mapper = new ObjectMapper();
            mapper.writerWithDefaultPrettyPrinter().writeValue(detailsFile, results);
            logger.info("Detailed results written to {}", detailsFile.getAbsolutePath());

            // Write summary to CSV
            File csvFile = new File(outputBasePath + ".csv");
            writeCsvSummary(results, csvFile);
            logger.info("Summary results written to {}", csvFile.getAbsolutePath());

            // Verify files were created
            if (csvFile.exists()) {
                logger.info("CSV file size: {} bytes", csvFile.length());
            } else {
                logger.error("Failed to create CSV file at {}", csvFile.getAbsolutePath());
            }

            if (detailsFile.exists()) {
                logger.info("JSON file size: {} bytes", detailsFile.length());
            } else {
                logger.error("Failed to create JSON file at {}", detailsFile.getAbsolutePath());
            }
        }

        private void writeCsvSummary(List<BenchResult> results, File outputFile) throws IOException {
            // Get summary statistics by dataset
            Map<String, SummaryStats> statsByDataset = BenchmarkSummarizer.summarizeByDataset(results);

            try (FileWriter writer = new FileWriter(outputFile)) {
                // Write CSV header
                writer.write("dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time\n");

                // Write one row per dataset with average metrics
                for (Map.Entry<String, SummaryStats> entry : statsByDataset.entrySet()) {
                    String dataset = entry.getKey();
                    SummaryStats datasetStats = entry.getValue();

                    writer.write(dataset + ",");
                    writer.write(datasetStats.getAvgQps() + ",");
                    writer.write(datasetStats.getQpsStdDev() + ",");
                    writer.write(datasetStats.getAvgLatency() + ",");
                    writer.write(datasetStats.getAvgRecall() + ",");
                    writer.write(datasetStats.getIndexConstruction() + "\n");
                }
            }
        }
    }

    /**
     * Creates a result handler that delegates to multiple handlers in sequence.
     * If any handler throws an exception, subsequent handlers are not called.
     *
     * @param handlers the handlers to combine
     * @return a result handler that invokes all provided handlers
     */
    static ResultHandler combining(ResultHandler... handlers) {
        return results -> {
            for (ResultHandler handler : handlers) {
                handler.handleResults(results);
            }
        };
    }
}
