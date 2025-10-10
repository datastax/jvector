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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.benchframe.BenchFrame;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

/**
 * Automated benchmark runner for GitHub Actions workflow.
 * This class is specifically designed to handle the --output argument
 * for regression testing in the run-bench.yml workflow.
 *
 * The benchmark runner supports checkpointing to allow resuming from failures.
 * It creates a checkpoint file (outputPath + ".checkpoint.json") that records
 * which datasets have been fully processed. If the benchmark is restarted,
 * it will skip datasets that have already been processed, allowing it to
 * continue from where it left off rather than starting over from the beginning.
 *
 * This class has been refactored to use BenchFrame for modularity and DRY principles.
 * All shared functionality is now in reusable modules.
 */
@Deprecated
public class AutoBenchYAML {
    private static final Logger logger = LoggerFactory.getLogger(AutoBenchYAML.class);

    public static void main(String[] args) throws IOException {
        // Parse command-line arguments
        String outputPath = extractArgument(args, "--output");
        if (outputPath == null) {
            logger.error("Error: --output argument is required for AutoBenchYAML");
            System.exit(1);
        }

        int diagnosticLevel = extractIntArgument(args, "--diag", 0);
        String[] filteredArgs = filterArguments(args, "--output", outputPath, "--diag", String.valueOf(diagnosticLevel));

        logger.info("Heap space available is {}", Runtime.getRuntime().maxMemory());
        logger.info("Filtered arguments: {}", Arrays.toString(filteredArgs));

        // Execute benchmark using convenience method
        BenchFrame.likeAutoBenchYAML(outputPath, diagnosticLevel).execute(filteredArgs);
    }

    /**
     * Extract a string argument value from command-line args
     */
    private static String extractArgument(String[] args, String flag) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(flag)) {
                return args[i + 1];
            }
        }
        return null;
    }

    /**
     * Extract an integer argument value from command-line args
     */
    private static int extractIntArgument(String[] args, String flag, int defaultValue) {
        String value = extractArgument(args, flag);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            logger.warn("Invalid integer value for {}: {}", flag, value);
            return defaultValue;
        }
    }

    /**
     * Filter out specific arguments and their values from the args array
     */
    private static String[] filterArguments(String[] args, String... toFilter) {
        return Arrays.stream(args)
            .filter(arg -> {
                for (String filter : toFilter) {
                    if (arg.equals(filter)) {
                        return false;
                    }
                }
                return true;
            })
            .toArray(String[]::new);
    }
}
