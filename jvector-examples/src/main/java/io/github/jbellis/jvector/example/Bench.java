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

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.CompressorParameters.PQParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.example.util.LoggerConfig;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.status.sinks.ConsolePanelSink;
import io.github.jbellis.jvector.status.sinks.LogBuffer;
import io.github.jbellis.jvector.status.sinks.OutputMode;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;


import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
@Command(
    name = "bench",
    mixinStandardHelpOptions = true,
    version = "JVector Benchmark 4.0.0",
    description = "Benchmark JVector indexes against various datasets",
    headerHeading = "%n",
    header = "JVector Benchmark Tool"
)
public class Bench implements Callable<Integer> {

    @Option(
        names = {"-o", "--output"},
        description = "Output mode: ${COMPLETION-CANDIDATES}",
        defaultValue = "AUTO"
    )
    private OutputMode outputMode = OutputMode.AUTO;

    @Option(
        names = {"-m", "--max-connections"},
        description = "M parameter for graph construction",
        defaultValue = "32"
    )
    private int m = 32;

    @Option(
        names = {"-e", "--ef-construction"},
        description = "EF construction parameter",
        defaultValue = "100"
    )
    private int efConstruction = 100;

    @Option(
        names = {"--neighbor-overflow"},
        description = "Neighbor overflow factor",
        defaultValue = "1.2"
    )
    private float neighborOverflow = 1.2f;

    @Option(
        names = {"--no-hierarchy"},
        description = "Disable hierarchical index construction",
        negatable = true
    )
    private boolean addHierarchy = true;

    @Option(
        names = {"--no-refine"},
        description = "Disable final graph refinement",
        negatable = true
    )
    private boolean refineFinalGraph = true;

    @Option(
        names = {"--no-pruning"},
        description = "Disable graph pruning",
        negatable = true
    )
    private boolean usePruning = true;

    @Parameters(
        description = "Dataset patterns (regex) to match",
        arity = "0..*"
    )
    private List<String> datasetPatterns = new ArrayList<>();


    static {
        // Static initializer to configure logging as early as possible
        LoggerConfig.configureForStaticInit();
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new Bench()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        // Resolve AUTO to actual mode if needed
        if (outputMode == OutputMode.AUTO) {
            outputMode = OutputMode.detect();
        }

        // Configure logging with the resolved mode
        LoggerConfig.configure(outputMode);

        System.out.println("Heap space available: " + Runtime.getRuntime().maxMemory());
        System.out.println("Output mode: " + outputMode.getName() + " - " + outputMode.getDescription());

        // Run the benchmark
        runBenchmark(outputMode);

        return 0;
    }

    private void runBenchmark(OutputMode outputMode) throws IOException {

        var mGrid = List.of(m);
        var efConstructionGrid = List.of(efConstruction);
        var topKGrid = Map.of(
                10, // topK
                List.of(1.0, 2.0, 5.0, 10.0), // oq
                100, // topK
                List.of(1.0, 2.0) // oq
        ); // rerankK = oq * topK
        var neighborOverflowGrid = List.of(neighborOverflow);
        var addHierarchyGrid = List.of(addHierarchy);
        var refineFinalGraphGrid = List.of(refineFinalGraph);
        var usePruningGrid = List.of(usePruning);
        List<Function<DataSet, CompressorParameters>> buildCompression = Arrays.asList(
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED),
                __ -> CompressorParameters.NONE
        );
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE,
                // ds -> new CompressorParameters.BQParameters(),
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED)
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.NVQ_VECTORS),
//                EnumSet.of(FeatureId.NVQ_VECTORS, FeatureId.FUSED_ADC),
                EnumSet.of(FeatureId.INLINE_VECTORS)
        );

        // Convert dataset patterns to regex
        String[] patternArray = datasetPatterns.toArray(new String[0]);
        var regex = patternArray.length == 0 ? ".*" : Arrays.stream(patternArray).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        execute(pattern, buildCompression, featureSets, searchCompression, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, topKGrid, usePruningGrid, outputMode);
    }



    private void execute(Pattern pattern, List<Function<DataSet, CompressorParameters>> buildCompression, List<EnumSet<FeatureId>> featureSets, List<Function<DataSet, CompressorParameters>> compressionGrid, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Float> neighborOverflowGrid, List<Boolean> addHierarchyGrid, List<Boolean> refineFinalGraphGrid, Map<Integer, List<Double>> topKGrid, List<Boolean> usePruningGrid, OutputMode outputMode) throws IOException {
        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());
        System.out.println("Executing the following datasets: " + datasetNames);

        TrackerScope rootScope;
        ConsolePanelSink consolePanelSink = null;

        if (outputMode == OutputMode.INTERACTIVE) {
            // Use ConsolePanelSink for interactive mode
            consolePanelSink = ConsolePanelSink.builder()
                    .withRefreshRateMs(100)
                    .withCaptureSystemStreams(true)  // Capture System streams in hierarchical mode
                    .build();

            // Register the sink with the log appender
            LogBuffer.setActiveSink(consolePanelSink);

            rootScope = new TrackerScope("Bench",
                    Duration.ofMillis(500),
                    List.of(consolePanelSink));
        } else {
            // Use simple console output
            rootScope = new TrackerScope("Bench",
                    Duration.ofMillis(500),
                    List.of(new ConsoleLoggerSink()));
        }

        // Create the root bench task
        BenchTask benchTask = new BenchTask(datasetNames.size());

        try (StatusTracker<BenchTask> rootTracker = rootScope.track(benchTask)) {
            benchTask.start();

            for (var datasetName : datasetNames) {
                // Create a child scope for each dataset
                TrackerScope datasetScope = rootScope.createChildScope(datasetName);

                DataSet ds = DataSetLoader.loadDataSet(datasetName);
                Grid.runAll(ds, datasetScope, rootTracker, mGrid, efConstructionGrid, neighborOverflowGrid,
                           addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompression,
                           compressionGrid, topKGrid, usePruningGrid);

                benchTask.datasetCompleted();
            }

            benchTask.complete();
        } finally {
            // Close the scope and sink when done
            rootScope.close();
            if (consolePanelSink != null) {
                consolePanelSink.close();
            }
        }
    }

    /**
     * Root task representing the entire bench session
     */
    static class BenchTask implements StatusUpdate.Provider<BenchTask> {
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;
        private final int totalDatasets;
        private int completedDatasets = 0;

        public BenchTask(int totalDatasets) {
            this.totalDatasets = totalDatasets;
        }

        public void start() {
            this.state = StatusUpdate.RunState.RUNNING;
        }

        public void datasetCompleted() {
            completedDatasets++;
            this.progress = (double) completedDatasets / totalDatasets;
        }

        public void complete() {
            this.progress = 1.0;
            this.state = StatusUpdate.RunState.SUCCESS;
        }

        public void fail() {
            this.state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<BenchTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return "Bench Session";
        }
    }
}
