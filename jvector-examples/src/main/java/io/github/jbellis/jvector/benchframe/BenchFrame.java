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

import io.github.jbellis.jvector.example.*;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetSource;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Unified benchmark framework that consolidates functionality from Bench, BenchYAML, and AutoBenchYAML.
 * Provides a modular, composable architecture using the strategy pattern for different components including
 * configuration sources, result handlers, and checkpointing strategies.
 * <p>
 * This class serves as the main orchestrator for JVector graph index benchmarks, supporting multiple execution
 * modes from simple hardcoded configurations to complex CI/CD scenarios with checkpointing and automated result
 * collection.
 *
 * <h2>Environment Variables</h2>
 * <ul>
 * <li>{@code VECTORDATA_CATALOGS} - Comma-separated list of additional catalog YAML files to load
 *     (e.g., "~/.config/custom1/catalogs.yaml,~/.config/custom2/catalogs.yaml")</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 * <h3>Command-Line Interface (Recommended)</h3>
 * <pre>{@code
 * // Run with hardcoded parameters (Bench-style)
 * BenchFrame.main(new String[]{"bench", "dataset-name"});
 *
 * // Run with YAML configuration (BenchYAML-style)
 * BenchFrame.main(new String[]{"benchyaml", "dataset-name"});
 *
 * // Run in CI/CD mode with checkpointing (AutoBenchYAML-style)
 * BenchFrame.main(new String[]{"autobenchyaml", "-o", "output", "dataset-name"});
 *
 * // List available datasets
 * BenchFrame.main(new String[]{"datasets"});
 *
 * // Access nbvectors CLI
 * BenchFrame.main(new String[]{"nbvectors", "--help"});
 * }</pre>
 *
 * <h3>Programmatic Usage - Convenience Methods</h3>
 * <pre>{@code
 * // Use hardcoded defaults
 * BenchFrame.likeBench().execute(args);
 *
 * // Use YAML configuration
 * BenchFrame.likeBenchYAML().execute(args);
 *
 * // Use CI/CD mode with checkpointing
 * BenchFrame.likeAutoBenchYAML(outputPath, diagnosticLevel).execute(args);
 * }</pre>
 *
 * <h3>Advanced - Custom Configuration with Builder</h3>
 * <pre>{@code
 * // Use a single config for all datasets
 * new BenchFrame.Builder()
 *     .withDatasetNames(List.of("my-dataset", "another-dataset"))
 *     .withConfig(BenchFrameConfig.createBenchDefaults())
 *     .withDataSetSource(DataSetSource.DEFAULT)
 *     .withResultHandler(ResultHandler.consoleOnly())
 *     .build()
 *     .execute(args);
 *
 * // Or use a function for per-dataset config (like YAML)
 * new BenchFrame.Builder()
 *     .withDatasetNames(List.of("dataset1", "dataset2"))
 *     .withConfigFunction(name -> loadYamlConfig(name))
 *     .build()
 *     .execute(args);
 * }</pre>
 *
 * <h3>For Synthetic 2D Datasets (Programmatic Only)</h3>
 * <pre>{@code
 * // Create and benchmark a 2D grid programmatically
 * var grid2d = DataSetCreator.create2DGrid(4_000_000, 10_000, 100);
 * BenchFrame.likeBench().execute(grid2d);
 * }</pre>
 *
 * @see BenchFrameConfig
 * @see ResultHandler
 * @see CheckpointStrategy
 * @see BenchFrameCLI
 */
public class BenchFrame {
    private static final Logger logger = LoggerFactory.getLogger(BenchFrame.class);

    private final List<String> datasetNames;
    private final BenchFrameConfig config;
    private final Function<String, BenchFrameConfig> configFunction;
    private final DataSetSource dataSetSource;
    private final ResultHandler resultHandler;
    private final CheckpointStrategy checkpointStrategy;
    private final boolean collectResults;
    private final int diagnosticLevel;

    private BenchFrame(Builder builder) {
        this.datasetNames = builder.datasetNames;
        this.config = builder.config;
        this.configFunction = builder.configFunction;
        this.dataSetSource = builder.dataSetSource;
        this.resultHandler = builder.resultHandler;
        this.checkpointStrategy = builder.checkpointStrategy;
        this.collectResults = builder.collectResults;
        this.diagnosticLevel = builder.diagnosticLevel;
    }

    /**
     * Executes the benchmark with a pre-created dataset. This method is primarily used by the Bench2D workflow
     * for synthetic 2D datasets but can be used programmatically with any {@link DataSet} instance.
     * <p>
     * The execution includes:
     * <ul>
     *   <li>Setting diagnostic level if configured</li>
     *   <li>Loading configuration for the dataset name</li>
     *   <li>Running the benchmark grid with configured parameters</li>
     *   <li>Handling results through the configured {@link ResultHandler}</li>
     * </ul>
     *
     * @param dataset the pre-created dataset to benchmark
     * @throws IOException if benchmark execution fails or result writing encounters I/O errors
     * @throws RuntimeException if the dataset configuration cannot be loaded
     */
    public void execute(DataSet dataset) throws IOException {
        if (diagnosticLevel > 0) {
            Grid.setDiagnosticLevel(diagnosticLevel);
        }

        logger.info("Executing benchmark for pre-created dataset: {}", dataset.getName());

        try {
            BenchFrameConfig datasetConfig = getConfigForDataset(dataset.getName());
            List<BenchResult> results = executeBenchmark(dataset, datasetConfig);

            resultHandler.handleResults(results);
            logger.info("Benchmark execution complete");
        } catch (Exception e) {
            logger.error("Failed to process dataset: {}", dataset.getName(), e);
            throw new RuntimeException("Benchmark failed for dataset: " + dataset.getName(), e);
        }
    }

    /**
     * Executes the benchmark with the given command-line arguments. This is the primary entry point for
     * benchmarking one or more datasets by name pattern.
     * <p>
     * The execution flow includes:
     * <ol>
     *   <li>Setting diagnostic level if configured</li>
     *   <li>Building a regex pattern from the provided arguments</li>
     *   <li>Filtering datasets by the pattern</li>
     *   <li>Loading previous results from checkpoint if checkpoint strategy is enabled</li>
     *   <li>For each matched dataset:
     *     <ul>
     *       <li>Checking if dataset should be skipped (already completed in checkpoint)</li>
     *       <li>Loading the dataset from the configured {@link DataSetSource}</li>
     *       <li>Loading configuration (either shared config or per-dataset function)</li>
     *       <li>Executing the benchmark</li>
     *       <li>Recording completion in checkpoint if enabled</li>
     *     </ul>
     *   </li>
     *   <li>Handling all results through the configured {@link ResultHandler}</li>
     * </ol>
     *
     * @param args command-line arguments, typically dataset name patterns. Multiple patterns are OR'd together.
     *             If empty, matches all datasets. Patterns support standard Java regex syntax.
     * @throws IOException if dataset loading, benchmark execution, or result writing encounters I/O errors
     * @throws RuntimeException if a dataset cannot be loaded or configuration cannot be retrieved
     */
    public void execute(String[] args) throws IOException {
        if (diagnosticLevel > 0) {
            Grid.setDiagnosticLevel(diagnosticLevel);
        }

        Pattern pattern = buildPattern(args);
        List<String> matchedDatasets = filterDatasets(datasetNames, pattern);

        if (matchedDatasets.isEmpty()) {
            logger.warn("No datasets matched pattern: {}", pattern.pattern());
            return;
        }

        logger.info("Executing benchmarks for datasets: {}", matchedDatasets);

        List<BenchResult> allResults = new ArrayList<>(checkpointStrategy.getPreviousResults());

        for (String datasetName : matchedDatasets) {
            if (checkpointStrategy.shouldSkipDataset(datasetName)) {
                logger.info("Skipping already completed dataset: {}", datasetName);
                continue;
            }

            logger.info("Loading dataset: {}", datasetName);
            try {
                DataSet dataset = dataSetSource.apply(datasetName)
                    .orElseThrow(() -> new RuntimeException("Unknown dataset: " + datasetName));

                BenchFrameConfig datasetConfig = getConfigForDataset(datasetName);
                List<BenchResult> datasetResults = executeBenchmark(dataset, datasetConfig);

                allResults.addAll(datasetResults);
                checkpointStrategy.recordCompletion(datasetName, datasetResults);

                logger.info("Completed benchmark for dataset: {}", datasetName);
            } catch (Exception e) {
                logger.error("Failed to process dataset: {}", datasetName, e);
                throw new RuntimeException("Benchmark failed for dataset: " + datasetName, e);
            }
        }

        resultHandler.handleResults(allResults);
        logger.info("Benchmark execution complete");
    }

    /**
     * Gets the configuration for a specific dataset. Uses configFunction if provided (for per-dataset config),
     * otherwise uses the single shared config.
     *
     * @param datasetName the dataset name
     * @return configuration for the dataset
     */
    private BenchFrameConfig getConfigForDataset(String datasetName) {
        if (configFunction != null) {
            return configFunction.apply(datasetName);
        } else {
            // Use shared config, but set the dataset name
            return config.toBuilder()
                .withDatasetName(datasetName)
                .build();
        }
    }

    /**
     * Executes the benchmark for a single dataset with the provided configuration. This method delegates
     * to {@link Grid} for the actual benchmark execution.
     *
     * @param dataset the dataset to benchmark
     * @param config the configuration specifying grid parameters and benchmark settings
     * @return list of {@link BenchResult} objects if result collection is enabled, empty list otherwise
     * @throws IOException if benchmark execution encounters I/O errors
     */
    private List<BenchResult> executeBenchmark(DataSet dataset, BenchFrameConfig config) throws IOException {
        if (collectResults) {
            return Grid.runAllAndCollectResults(
                dataset,
                config.getMGrid(),
                config.getEfConstructionGrid(),
                config.getNeighborOverflowGrid(),
                config.getAddHierarchyGrid(),
                config.getFeatureSets(),
                config.getBuildCompressors(),
                config.getSearchCompressors(),
                config.getTopKOverqueryGrid(),
                config.getUsePruningGrid()
            );
        } else {
            Grid.runAll(
                dataset,
                config.getMGrid(),
                config.getEfConstructionGrid(),
                config.getNeighborOverflowGrid(),
                config.getAddHierarchyGrid(),
                config.getRefineFinalGraphGrid(),
                config.getFeatureSets(),
                config.getBuildCompressors(),
                config.getSearchCompressors(),
                config.getTopKOverqueryGrid(),
                config.getUsePruningGrid(),
                config.getBenchmarkSpec()
            );
            return List.of();
        }
    }

    /**
     * Builds a regex pattern from command-line arguments. Multiple patterns are OR'd together.
     * Arguments can contain space-separated patterns that are split and combined.
     * <p>
     * Examples:
     * <ul>
     *   <li>Empty args: matches everything (".*")</li>
     *   <li>{"dataset1"}: matches "dataset1"</li>
     *   <li>{"dataset1", "dataset2"}: matches "dataset1" OR "dataset2"</li>
     *   <li>{"dataset1 dataset2"}: matches "dataset1" OR "dataset2" (space-split)</li>
     * </ul>
     *
     * @param args command-line arguments containing dataset name patterns
     * @return compiled regex pattern for dataset filtering
     */
    private static Pattern buildPattern(String[] args) {
        var regex = args.length == 0 ? ".*"
            : Arrays.stream(args)
                .flatMap(s -> Arrays.stream(s.split("\\s")))
                .map(s -> "(?:" + s + ")")
                .collect(Collectors.joining("|"));
        return Pattern.compile(regex);
    }

    /**
     * Filters dataset names by regex pattern using partial matching (find, not full match).
     *
     * @param datasets the list of dataset names to filter
     * @param pattern the regex pattern to match against
     * @return list of dataset names where the pattern was found
     */
    private static List<String> filterDatasets(List<String> datasets, Pattern pattern) {
        return datasets.stream()
            .filter(name -> pattern.matcher(name).find())
            .collect(Collectors.toList());
    }

    /**
     * Creates a BenchFrame configured like the original Bench.java with hardcoded grid parameters.
     * This factory method provides compatibility with the legacy Bench class behavior.
     * <p>
     * Configuration includes:
     * <ul>
     *   <li>Datasets loaded from {@link DatasetCollection}</li>
     *   <li>Hardcoded default grid parameters (M=32, efConstruction=100, etc.)</li>
     *   <li>Console-only output (no file writing)</li>
     *   <li>No checkpointing</li>
     * </ul>
     *
     * @return a BenchFrame instance configured with hardcoded defaults
     * @throws UncheckedIOException if the dataset collection cannot be loaded
     */
    public static BenchFrame likeBench() {
        try {
            return new Builder()
                .withDatasetNames(DatasetCollection.load().getAll())
                .withConfig(BenchFrameConfig.createBenchDefaults())
                .withDataSetSource(DataSetSource.DEFAULT)
                .withResultHandler(ResultHandler.consoleOnly())
                .build();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load dataset collection", e);
        }
    }

    /**
     * Creates a BenchFrame configured like the original BenchYAML.java with YAML-based configuration.
     * This factory method provides compatibility with the legacy BenchYAML class behavior.
     * <p>
     * Configuration includes:
     * <ul>
     *   <li>Datasets loaded from {@link DatasetCollection}</li>
     *   <li>Parameters loaded from YAML configuration files per dataset</li>
     *   <li>Console-only output (no file writing)</li>
     *   <li>No checkpointing</li>
     * </ul>
     *
     * @return a BenchFrame instance configured to load parameters from YAML
     * @throws UncheckedIOException if the dataset collection cannot be loaded
     */
    public static BenchFrame likeBenchYAML() {
        try {
            return new Builder()
                .withDatasetNames(DatasetCollection.load().getAll())
                .withConfigFunction(datasetName -> {
                    try {
                        MultiConfig multiConfig = MultiConfig.getDefaultConfig(datasetName);
                        return BenchFrameConfig.fromMultiConfig(multiConfig);
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException("Failed to load YAML config for dataset: " + datasetName, e);
                    }
                })
                .withDataSetSource(DataSetSource.DEFAULT)
                .withResultHandler(ResultHandler.consoleOnly())
                .build();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load dataset collection", e);
        }
    }

    /**
     * Creates a BenchFrame configured like the original AutoBenchYAML.java for CI/CD scenarios.
     * This factory method provides compatibility with the legacy AutoBenchYAML class behavior
     * with additional support for checkpointing and file-based output.
     * <p>
     * Configuration includes:
     * <ul>
     *   <li>Hardcoded dataset list for CI/CD: cap-1M, cap-6M, cohere-english-v3-1M,
     *       cohere-english-v3-10M, dpr-1M, dpr-10M</li>
     *   <li>Parameters loaded from autoDefault YAML configuration</li>
     *   <li>File-based output: CSV summary and JSON details</li>
     *   <li>File-based checkpointing to support resumption after failures</li>
     *   <li>Result collection enabled</li>
     *   <li>Configurable diagnostic level</li>
     * </ul>
     *
     * @param outputPath base path for output files (.csv, .json, .checkpoint.json)
     * @param diagnosticLevel diagnostic level controlling Grid output verbosity
     *                        (0=none, 1=basic, 2=detailed, 3=verbose)
     * @return a BenchFrame instance configured for CI/CD with checkpointing
     * @see ResultHandler#toFiles(String)
     * @see CheckpointStrategy#fileBasedCheckpointing(String)
     */
    public static BenchFrame likeAutoBenchYAML(String outputPath, int diagnosticLevel) {
        // Hardcoded list for CI/CD (matches original AutoBenchYAML)
        List<String> datasets = Arrays.asList(
            "cap-1M", "cap-6M",
            "cohere-english-v3-1M", "cohere-english-v3-10M",
            "dpr-1M", "dpr-10M"
        );

        try {
            MultiConfig multiConfig = MultiConfig.getDefaultConfig("autoDefault");
            BenchFrameConfig baseConfig = BenchFrameConfig.fromMultiConfig(multiConfig)
                .toBuilder()
                .collectResults(true)
                .build();

            return new Builder()
                .withDatasetNames(datasets)
                .withConfig(baseConfig)
                .withDataSetSource(DataSetSource.DEFAULT)
                .withResultHandler(ResultHandler.toFiles(outputPath))
                .withCheckpointStrategy(CheckpointStrategy.fileBasedCheckpointing(outputPath))
                .withDiagnosticLevel(diagnosticLevel)
                .build();
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Failed to load autoDefault YAML config", e);
        }
    }

    /**
     * Builder for constructing BenchFrame instances with fluent API. Provides fine-grained control
     * over all aspects of benchmark configuration including datasets, configuration,
     * result handling, and checkpointing.
     * <p>
     * Default values:
     * <ul>
     *   <li>datasetNames: empty list</li>
     *   <li>config: null (must be set via withConfig or withConfigFunction)</li>
     *   <li>configFunction: null</li>
     *   <li>dataSetSource: {@link DataSetSource#DEFAULT}</li>
     *   <li>resultHandler: {@link ResultHandler#consoleOnly()}</li>
     *   <li>checkpointStrategy: {@link CheckpointStrategy#none()}</li>
     *   <li>collectResults: false</li>
     *   <li>diagnosticLevel: 0</li>
     * </ul>
     */
    public static class Builder {
        private List<String> datasetNames = List.of();
        private BenchFrameConfig config = null;
        private Function<String, BenchFrameConfig> configFunction = null;
        private DataSetSource dataSetSource = DataSetSource.DEFAULT;
        private ResultHandler resultHandler = ResultHandler.consoleOnly();
        private CheckpointStrategy checkpointStrategy = CheckpointStrategy.none();
        private boolean collectResults = false;
        private int diagnosticLevel = 0;

        /**
         * Sets the list of dataset names to benchmark. The provided list is copied to prevent external modification.
         *
         * @param datasetNames the list of dataset names to benchmark
         * @return this builder for method chaining
         */
        public Builder withDatasetNames(List<String> datasetNames) {
            this.datasetNames = new ArrayList<>(datasetNames);
            return this;
        }

        /**
         * Sets a single configuration to use for all datasets.
         * Mutually exclusive with {@link #withConfigFunction}.
         *
         * @param config the configuration to use for all datasets
         * @return this builder for method chaining
         */
        public Builder withConfig(BenchFrameConfig config) {
            this.config = config;
            this.configFunction = null;
            return this;
        }

        /**
         * Sets a function to generate configuration per dataset (e.g., for YAML-based config).
         * Mutually exclusive with {@link #withConfig}.
         *
         * @param configFunction function mapping dataset name to configuration
         * @return this builder for method chaining
         */
        public Builder withConfigFunction(Function<String, BenchFrameConfig> configFunction) {
            this.configFunction = configFunction;
            this.config = null;
            return this;
        }

        /**
         * Sets the DataSetSource for loading datasets by name.
         *
         * @param source the dataset source to use
         * @return this builder for method chaining
         */
        public Builder withDataSetSource(DataSetSource source) {
            this.dataSetSource = source;
            return this;
        }

        /**
         * Sets the result handler strategy for processing benchmark results.
         *
         * @param handler the result handler strategy to use
         * @return this builder for method chaining
         * @see ResultHandler#consoleOnly()
         * @see ResultHandler#toFiles(String)
         */
        public Builder withResultHandler(ResultHandler handler) {
            this.resultHandler = handler;
            return this;
        }

        /**
         * Sets the checkpoint strategy for tracking and resuming benchmark progress.
         *
         * @param strategy the checkpoint strategy to use
         * @return this builder for method chaining
         * @see CheckpointStrategy#none()
         * @see CheckpointStrategy#fileBasedCheckpointing(String)
         */
        public Builder withCheckpointStrategy(CheckpointStrategy strategy) {
            this.checkpointStrategy = strategy;
            return this;
        }

        /**
         * Enables or disables result collection. When enabled, benchmark results are collected and returned
         * from the execution. This is required for file output and checkpointing functionality.
         *
         * @param collect true to collect results, false to discard them
         * @return this builder for method chaining
         */
        public Builder collectResults(boolean collect) {
            this.collectResults = collect;
            return this;
        }

        /**
         * Sets the diagnostic level for Grid execution output.
         *
         * @param level diagnostic level: 0=none, 1=basic, 2=detailed, 3=verbose
         * @return this builder for method chaining
         */
        public Builder withDiagnosticLevel(int level) {
            this.diagnosticLevel = level;
            return this;
        }

        /**
         * Builds and returns a configured BenchFrame instance.
         *
         * @return a new BenchFrame instance with the configured settings
         */
        public BenchFrame build() {
            return new BenchFrame(this);
        }
    }

    /**
     * Main entry point for command-line execution. Delegates to {@link BenchFrameCLI} for
     * command-line parsing and subcommand handling.
     *
     * @param args command-line arguments
     * @see BenchFrameCLI
     */
    public static void main(String[] args) {
        int exitCode = new CommandLine(new BenchFrameCLI()).execute(args);
        System.exit(exitCode);
    }
}
