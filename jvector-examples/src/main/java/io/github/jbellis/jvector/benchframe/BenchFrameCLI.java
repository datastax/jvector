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

import picocli.CommandLine;

import java.io.IOException;
import java.util.concurrent.Callable;

/**
 * Command-line interface for BenchFrame using PicoCLI. Provides subcommands for all
 * benchmark modes from the original benchmark classes (Bench, BenchYAML, AutoBenchYAML)
 * plus dataset management via integration with nbvectors CLI.
 * <p>
 * This CLI serves as the primary entry point for command-line benchmark execution and
 * delegates to {@link BenchFrame} for actual benchmark orchestration.
 *
 * <h2>Available Subcommands</h2>
 * <ul>
 *   <li>{@code bench} - Run with hardcoded grid parameters (Bench.java style)</li>
 *   <li>{@code benchyaml} - Run with YAML-based configuration (BenchYAML.java style)</li>
 *   <li>{@code autobenchyaml} - Run in CI/CD mode with checkpointing (AutoBenchYAML.java style)</li>
 *   <li>{@code datasets} - List and manage vector datasets (delegates to nbvectors)</li>
 *   <li>{@code nbvectors} - Access full nbvectors CLI functionality</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 * <pre>
 * # Show help
 * java -jar benchframe.jar --help
 *
 * # Run Bench-style on specific datasets
 * java -jar benchframe.jar bench "dataset1|dataset2"
 *
 * # Run YAML-style on all datasets
 * java -jar benchframe.jar benchyaml
 *
 * # Run CI/CD mode with output files
 * java -jar benchframe.jar autobenchyaml -o results/benchmark
 *
 * # List available datasets
 * java -jar benchframe.jar datasets
 *
 * # Access nbvectors CLI
 * java -jar benchframe.jar nbvectors --help
 * </pre>
 *
 * @see BenchFrame
 * @see BenchCommand
 * @see BenchYAMLCommand
 * @see AutoBenchYAMLCommand
 * @see DatasetsCommand
 * @see NBVectorsCommand
 */
@CommandLine.Command(
        name = "benchframe",
        mixinStandardHelpOptions = true,
        version = "1.0",
        description = "Unified benchmark framework for JVector graph indexes",
        subcommands = {
                BenchFrameCLI.BenchCommand.class,
                BenchFrameCLI.BenchYAMLCommand.class,
                BenchFrameCLI.AutoBenchYAMLCommand.class,
                BenchFrameCLI.DatasetsCommand.class,
                BenchFrameCLI.NBVectorsCommand.class
        }
)
public class BenchFrameCLI implements Callable<Integer> {

    /**
     * Called when no subcommand is specified. Displays help information.
     *
     * @return exit code 0
     */
    @Override
    public Integer call() {
        // If no subcommand, show help
        CommandLine.usage(this, System.out);
        return 0;
    }

    /**
     * Subcommand for running Bench-style benchmarks with hardcoded grid parameters.
     * Provides compatibility with the original Bench.java behavior.
     * <p>
     * Uses fixed default parameters (M=32, efConstruction=100, etc.) and loads
     * datasets from the DatasetCollection.
     */
    @CommandLine.Command(
            name = "bench",
            description = "Run benchmarks with hardcoded grid parameters (original Bench.java style)"
    )
    static class BenchCommand implements Callable<Integer> {
        @CommandLine.Parameters(
                arity = "0..*",
                description = "Dataset name patterns (regex). If not specified, matches all datasets."
        )
        private String[] datasets = new String[0];

        @Override
        public Integer call() throws IOException {
            System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
            BenchFrame.likeBench().execute(datasets);
            return 0;
        }
    }

    /**
     * Subcommand for running BenchYAML-style benchmarks with YAML-based configuration.
     * Provides compatibility with the original BenchYAML.java behavior.
     * <p>
     * Loads benchmark parameters from YAML files per dataset, allowing different
     * configurations for different datasets.
     */
    @CommandLine.Command(
            name = "benchyaml",
            description = "Run benchmarks with YAML-based configuration (original BenchYAML.java style)"
    )
    static class BenchYAMLCommand implements Callable<Integer> {
        @CommandLine.Parameters(
                arity = "0..*",
                description = "Dataset name patterns (regex) or YAML config files. If not specified, matches all datasets."
        )
        private String[] datasets = new String[0];

        @Override
        public Integer call() throws IOException {
            System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
            BenchFrame.likeBenchYAML().execute(datasets);
            return 0;
        }
    }

    /**
     * Subcommand for running AutoBench-style benchmarks in CI/CD mode with checkpointing.
     * Provides compatibility with the original AutoBenchYAML.java behavior.
     * <p>
     * Features:
     * <ul>
     *   <li>File-based checkpointing for resumption after failures</li>
     *   <li>CSV summary and JSON detail output</li>
     *   <li>Hardcoded dataset list for consistent CI/CD runs</li>
     *   <li>Configurable diagnostic output level</li>
     * </ul>
     */
    @CommandLine.Command(
            name = "autobenchyaml",
            description = "Run benchmarks for CI/CD with checkpointing and file output (original AutoBenchYAML.java style)"
    )
    static class AutoBenchYAMLCommand implements Callable<Integer> {
        @CommandLine.Parameters(
                arity = "0..*",
                description = "Dataset name patterns (regex). If not specified, matches all datasets."
        )
        private String[] datasets = new String[0];

        @CommandLine.Option(
                names = {"-o", "--output"},
                required = true,
                description = "Base path for output files (.csv, .json, .checkpoint.json)"
        )
        private String outputPath;

        @CommandLine.Option(
                names = {"-d", "--diag"},
                description = "Diagnostic level: 0=none, 1=basic, 2=detailed, 3=verbose (default: ${DEFAULT-VALUE})",
                defaultValue = "0"
        )
        private int diagnosticLevel;

        @Override
        public Integer call() throws IOException {
            System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
            BenchFrame.likeAutoBenchYAML(outputPath, diagnosticLevel).execute(datasets);
            return 0;
        }
    }

    /**
     * Subcommand that delegates to the datatools-nbvectors datasets command.
     * Provides access to dataset listing and management functionality.
     */
    @CommandLine.Command(
            name = "datasets",
            description = "List and manage vector datasets (delegates to nbvectors datasets command)"
    )
    static class DatasetsCommand implements Callable<Integer> {
        @CommandLine.Parameters(
                arity = "0..*",
                description = "Arguments to pass to the nbvectors datasets command"
        )
        private String[] args = new String[0];

        @Override
        public Integer call() throws Exception {
            // Delegate to CommandBundler with datasets subcommand
            String[] nbvectorArgs = new String[args.length + 1];
            nbvectorArgs[0] = "datasets";
            System.arraycopy(args, 0, nbvectorArgs, 1, args.length);

            io.nosqlbench.commands.CommandBundler.main(nbvectorArgs);
            return 0;
        }
    }

    /**
     * Subcommand that delegates to the datatools-nbvectors main CLI.
     * Provides access to the full nbvectors command-line functionality.
     */
    @CommandLine.Command(
            name = "nbvectors",
            description = "Access full nbvectors CLI functionality (delegates to CommandBundler)"
        )
    static class NBVectorsCommand implements Callable<Integer> {
        @CommandLine.Parameters(
                arity = "0..*",
                description = "Arguments to pass to the nbvectors CLI"
        )
        private String[] args = new String[0];

        @Override
        public Integer call() throws Exception {
            // Delegate to CommandBundler
            io.nosqlbench.commands.CommandBundler.main(args);
            return 0;
        }
    }

    /**
     * Main entry point for command-line execution.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        int exitCode = new CommandLine(new BenchFrameCLI()).execute(args);
        System.exit(exitCode);
    }

}
