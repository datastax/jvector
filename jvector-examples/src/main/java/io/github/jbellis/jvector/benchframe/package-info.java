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

/**
 * Unified benchmark framework for JVector graph indexes. This package consolidates the functionality
 * from the legacy benchmark classes (Bench, BenchYAML, AutoBenchYAML) into a modular,
 * composable architecture using closures and strategy interfaces.
 *
 * <h2>Quick Start</h2>
 *
 * <h3>Command-Line Interface</h3>
 * The recommended way to run benchmarks is via the CLI:
 * <pre>
 * # Run with hardcoded parameters (original Bench.java)
 * benchframe bench dataset-name
 *
 * # Run with YAML configuration (original BenchYAML.java)
 * benchframe benchyaml dataset-name
 *
 * # Run CI/CD mode with checkpointing (original AutoBenchYAML.java)
 * benchframe autobenchyaml -o results/output dataset-name
 *
 * # List available datasets
 * benchframe datasets
 *
 * # Access full nbvectors functionality
 * benchframe nbvectors --help
 * </pre>
 *
 * <h3>Programmatic Usage</h3>
 * For library usage, use the convenience factory methods:
 * <pre>{@code
 * // Hardcoded defaults (Bench-style)
 * BenchFrame.likeBench().execute(args);
 *
 * // YAML configuration (BenchYAML-style)
 * BenchFrame.likeBenchYAML().execute(args);
 *
 * // CI/CD with checkpointing (AutoBenchYAML-style)
 * BenchFrame.likeAutoBenchYAML(outputPath, diagnosticLevel).execute(args);
 * }</pre>
 *
 * <hr/>
 * <h2>Developer Documentation</h2>
 *
 * The sections below provide detailed information for developers working on the BenchFrame itself.
 *
 * <h2>Package Overview</h2>
 * The benchframe package provides a flexible framework for benchmarking JVector's approximate
 * nearest neighbor search implementations. It supports multiple execution modes from simple
 * interactive testing to complex CI/CD scenarios with checkpointing and automated result collection.
 *
 * <h2>Core Components</h2>
 *
 * <h3>Main Orchestrator</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.benchframe.BenchFrame} - Main orchestrator that coordinates
 *       benchmark execution using pluggable strategies</li>
 *   <li>{@link io.github.jbellis.jvector.benchframe.BenchFrameCLI} - Command-line interface providing
 *       subcommands for different benchmark modes</li>
 * </ul>
 *
 * <h3>Configuration</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.benchframe.BenchFrameConfig} - Immutable configuration class
 *       encapsulating all benchmark parameters. Can be used as a single shared config or via a
 *       Function for per-dataset configuration (e.g., YAML)</li>
 * </ul>
 *
 * <h3>Result Handling</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.benchframe.BenchResult} - Result model encapsulating dataset,
 *       parameters, and metrics</li>
 *   <li>{@link io.github.jbellis.jvector.benchframe.ResultHandler} - Strategy interface for handling
 *       results (console, files, etc.)</li>
 * </ul>
 *
 * <h3>Checkpointing</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.benchframe.CheckpointStrategy} - Strategy interface for
 *       managing resumable benchmark execution</li>
 * </ul>
 *
 * <h2>Usage Patterns</h2>
 *
 * <h2>Available CLI Subcommands</h2>
 * <ul>
 *   <li><strong>bench</strong> - Run with hardcoded grid parameters (original Bench.java)</li>
 *   <li><strong>benchyaml</strong> - Run with YAML-based configuration (original BenchYAML.java)</li>
 *   <li><strong>autobenchyaml</strong> - CI/CD mode with checkpointing and file output (original AutoBenchYAML.java)</li>
 *   <li><strong>datasets</strong> - List and manage vector datasets (delegates to nbvectors)</li>
 *   <li><strong>nbvectors</strong> - Access full nbvectors CLI functionality</li>
 * </ul>
 *
 * <h3>CLI Examples</h3>
 * <pre>
 * # Run with hardcoded parameters
 * benchframe bench dataset-name
 *
 * # Run with YAML configuration
 * benchframe benchyaml dataset-name
 *
 * # Run CI/CD mode with checkpointing (--output required)
 * benchframe autobenchyaml -o results/output dataset-name
 * benchframe autobenchyaml -o results/output -d 2 cap-1M
 *
 * # List available datasets
 * benchframe datasets
 * benchframe datasets search cohere
 *
 * # Access nbvectors functionality
 * benchframe nbvectors --help
 * benchframe nbvectors catalogs list
 * </pre>
 *
 * <h3>Environment Variables</h3>
 * <ul>
 *   <li><strong>VECTORDATA_CATALOGS</strong> - Comma-separated list of additional catalog YAML files
 *       to load (e.g., "~/.config/custom/catalogs.yaml,~/work/catalogs.yaml")</li>
 * </ul>
 *
 * <h3>Programmatic Usage - Factory Methods</h3>
 * Factory methods provide pre-configured instances matching legacy behavior:
 * <pre>{@code
 * // Bench-style: hardcoded defaults
 * BenchFrame frame = BenchFrame.likeBench();
 * frame.execute(new String[]{"dataset-name"});
 *
 * // BenchYAML-style: YAML configuration
 * BenchFrame frame = BenchFrame.likeBenchYAML();
 * frame.execute(new String[]{"dataset-name"});
 *
 * // AutoBenchYAML-style: CI/CD with checkpointing
 * BenchFrame frame = BenchFrame.likeAutoBenchYAML("results/benchmark", 2);
 * frame.execute(new String[]{"dataset-name"});
 * }</pre>
 *
 * <h3>Programmatic Usage - Custom Configuration</h3>
 * The Builder API provides fine-grained control over all aspects:
 * <pre>{@code
 * // With a single shared config
 * BenchFrame frame = new BenchFrame.Builder()
 *     .withDatasetNames(List.of("dataset1", "dataset2"))
 *     .withConfig(BenchFrameConfig.createBenchDefaults())
 *     .withDataSetSource(DataSetSource.DEFAULT)
 *     .withResultHandler(ResultHandler.toFiles("results/benchmark"))
 *     .withCheckpointStrategy(CheckpointStrategy.fileBasedCheckpointing("results/checkpoint"))
 *     .withDiagnosticLevel(2)
 *     .build();
 *
 * // With per-dataset config function (like YAML)
 * BenchFrame frame = new BenchFrame.Builder()
 *     .withDatasetNames(List.of("dataset1", "dataset2"))
 *     .withConfigFunction(name -> loadCustomConfig(name))
 *     .build();
 *
 * frame.execute(new String[]{".*"});
 * }</pre>
 *
 * <h2>Extension Points</h2>
 * The framework is designed for extension through closures and strategy interfaces:
 *
 * <h3>Custom Configuration Function</h3>
 * <pre>{@code
 * Function<String, BenchFrameConfig> customConfigFn = datasetName -> {
 *     // Load from database, REST API, etc.
 *     return new BenchFrameConfig.Builder()
 *         .withDatasetName(datasetName)
 *         .withMGrid(List.of(16, 32, 64))
 *         .build();
 * };
 * }</pre>
 *
 * <h3>Custom Result Handler</h3>
 * <pre>{@code
 * ResultHandler customHandler = results -> {
 *     // Send to monitoring system
 *     monitoringSystem.record(results);
 *     // Upload to cloud storage
 *     cloudStorage.upload("benchmarks", results);
 * };
 * }</pre>
 *
 * <h3>Custom Checkpoint Strategy</h3>
 * <pre>{@code
 * CheckpointStrategy customStrategy = new CheckpointStrategy() {
 *     public boolean shouldSkipDataset(String name) {
 *         return database.isCompleted(name);
 *     }
 *     public void recordCompletion(String name, List<BenchResult> results) {
 *         database.markCompleted(name, results);
 *     }
 *     public List<BenchResult> getPreviousResults() {
 *         return database.loadPreviousResults();
 *     }
 * };
 * }</pre>
 *
 * <h2>Architecture Benefits</h2>
 * <ul>
 *   <li><strong>Modularity:</strong> Clean separation of concerns through strategy interfaces</li>
 *   <li><strong>Composability:</strong> Mix and match strategies for different scenarios</li>
 *   <li><strong>Testability:</strong> Easy to test components in isolation with mock strategies</li>
 *   <li><strong>Extensibility:</strong> Add new strategies without modifying existing code</li>
 *   <li><strong>Backward Compatibility:</strong> Factory methods preserve legacy behavior</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * The framework components are generally not thread-safe and are designed for single-threaded
 * benchmark execution. {@link io.github.jbellis.jvector.benchframe.BenchFrameConfig} instances
 * are immutable and thread-safe once constructed.
 *
 * @see io.github.jbellis.jvector.benchframe.BenchFrame
 * @see io.github.jbellis.jvector.benchframe.BenchFrameCLI
 * @see io.github.jbellis.jvector.example.Grid
 */
package io.github.jbellis.jvector.benchframe;
