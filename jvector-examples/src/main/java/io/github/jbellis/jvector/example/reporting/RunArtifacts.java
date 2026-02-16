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

package io.github.jbellis.jvector.example.reporting;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetLoaderMFD;
import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.MetricSelection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Run-scoped reporting/artifacts for a benchmark invocation.
 *
 * Owns:
 * - sys_info.json (RunReporting)
 * - dataset_info.csv (DatasetInfoWriter)
 * - experiments.csv (ExperimentsCsvWriter)
 * - run-level compute/display/log selections from run.yml
 */
public final class RunArtifacts implements AutoCloseable {

    private final RunContext run;
    private final DatasetInfoWriter datasetInfoWriter;
    private final ExperimentsCsvWriter experimentsWriter;
    private final SystemStatsCollector sysStatsCollector;
    private final JfrRecorder jfrRecorder;

    private final Map<String, List<String>> benchmarksToCompute;
    private final Map<String, List<String>> benchmarksToDisplay;
    private final MetricSelection metricsToDisplay;
    private final Map<String, List<String>> benchmarksToLog;
    private final MetricSelection metricsToLog;

    private RunArtifacts(RunContext run,
                         DatasetInfoWriter datasetInfoWriter,
                         ExperimentsCsvWriter experimentsWriter,
                         SystemStatsCollector sysStatsCollector,
                         JfrRecorder jfrRecorder,
                         Map<String, List<String>> benchmarksToCompute,
                         Map<String, List<String>> benchmarksToDisplay,
                         MetricSelection metricsToDisplay,
                         Map<String, List<String>> benchmarksToLog,
                         MetricSelection metricsToLog) {
        this.run = run;
        this.datasetInfoWriter = datasetInfoWriter;
        this.experimentsWriter = experimentsWriter;
        this.sysStatsCollector = sysStatsCollector;
        this.jfrRecorder = jfrRecorder;
        this.benchmarksToCompute = benchmarksToCompute;
        this.benchmarksToDisplay = benchmarksToDisplay;
        this.metricsToDisplay = metricsToDisplay;
        this.benchmarksToLog = benchmarksToLog;
        this.metricsToLog = metricsToLog;
    }

    public static RunArtifacts open(RunConfig runCfg, List<MultiConfig> datasetConfigs) throws IOException {
        Objects.requireNonNull(runCfg, "runCfg");
        Objects.requireNonNull(datasetConfigs, "datasetConfigs");

        boolean loggingEnabled =
                runCfg.logging != null &&
                        runCfg.logging.type != null &&
                        !runCfg.logging.type.isBlank();

        // Even if logging is disabled, we still want run-level compute/display selections
        Map<String, List<String>> benchmarksToCompute = runCfg.benchmarks;

        Map<String, List<String>> benchmarksToDisplay =
                (runCfg.console != null) ? runCfg.console.benchmarks : null;
        MetricSelection metricsToDisplay =
                (runCfg.console != null) ? runCfg.console.metrics : null;

        Map<String, List<String>> benchmarksToLog =
                (runCfg.logging != null) ? runCfg.logging.benchmarks : null;
        MetricSelection metricsToLog =
                (runCfg.logging != null) ? runCfg.logging.metrics : null;

        // Validate selections once (run-level)
        ReportingSelectionResolver.validateBenchmarkSelectionSubset(
                benchmarksToCompute,
                benchmarksToDisplay,
                SearchReportingCatalog.defaultComputeBenchmarks()
        );
        ReportingSelectionResolver.validateNamedMetricSelectionNames(
                metricsToDisplay,
                SearchReportingCatalog.catalog()
        );

        ReportingSelectionResolver.validateBenchmarkSelectionSubset(
                benchmarksToCompute,
                benchmarksToLog,
                SearchReportingCatalog.defaultComputeBenchmarks()
        );
        ReportingSelectionResolver.validateNamedMetricSelectionNames(
                metricsToLog,
                SearchReportingCatalog.catalog()
        );

        if (!loggingEnabled) {
            // No sys_info/dataset_info/experiments.csv when logging is disabled
            return new RunArtifacts(
                    null,  // run
                    null,  // datasetInfoWriter
                    null,  // experimentsWriter
                    null,  // sysStatsCollector
                    null,  // jfrRecorder
                    benchmarksToCompute,
                    benchmarksToDisplay,
                    metricsToDisplay,
                    benchmarksToLog,
                    metricsToLog
            );
        }

        // Logging enabled => create run artifacts
        var reporting = RunReporting.open(runCfg);
        var run = reporting.run();

        System.out.println("Logging enabled: " + run.runDir().toAbsolutePath() + ".\n" +
                "Delete this directory to remove run artifacts and free up disk space.");

        var datasetInfoWriter = DatasetInfoWriter.open(run);

        var outputKeyColumns = LoggingSchemaPlanner.unionLoggingMetricKeys(runCfg, datasetConfigs);
        var experimentsWriter = new ExperimentsCsvWriter(run, ExperimentsSchemaV1.fixedColumns(), outputKeyColumns);

        SystemStatsCollector sysStatsCollector = null;
        if (runCfg.logging.sysStats) {
            sysStatsCollector = new SystemStatsCollector();
            sysStatsCollector.start(run.runDir(), "sys_stats.jsonl");
        }

        JfrRecorder jfrRecorder = null;
        if (runCfg.logging.jfr) {
            jfrRecorder = new JfrRecorder();
            try {
                jfrRecorder.start(run.runDir(), "run.jfr");
            } catch (Exception e) {
                System.err.println("Failed to start JFR: " + e.getMessage());
                jfrRecorder = null;
            }
        }

        return new RunArtifacts(
                run,
                datasetInfoWriter,
                experimentsWriter,
                sysStatsCollector,
                jfrRecorder,
                benchmarksToCompute,
                benchmarksToDisplay,
                metricsToDisplay,
                benchmarksToLog,
                metricsToLog
        );
    }

    /** No-op artifacts instance for legacy callers (no sys_info/dataset_info/experiments output). */
    public static RunArtifacts disabled() {
        return new RunArtifacts(
                null, null, null, null, null,
                null, null, null,
                null, null
        );
    }

    /**
     * Legacy yamlSchemaVersion "0" behavior:
     * - logging disabled
     * - console selection == compute selection
     */
    public static RunArtifacts legacyNoLogging(Map<String, List<String>> legacyBenchmarksToCompute) {
        Map<String, List<String>> compute = legacyBenchmarksToCompute;
        Map<String, List<String>> display = legacyBenchmarksToCompute;
        return new RunArtifacts(
                null,  // run
                null,  // datasetInfoWriter
                null,  // experimentsWriter
                null,  // sysStatsCollector
                null,  // jfrRecorder
                compute,
                display,
                null,  // metricsToDisplay
                null,  // benchmarksToLog
                null   // metricsToLog
        );
    }

    /**
     * Append one experiments.csv row for a single (index config × query config) measurement.
     * No-op if experiments logging is not enabled.
     */
    public void logRow(String datasetName,
                       int M,
                       int efConstruction,
                       float neighborOverflow,
                       boolean addHierarchy,
                       boolean refineFinalGraph,
                       Set<FeatureId> featureSetForIndex,
                       boolean usePruning,
                       int topK,
                       double overquery,
                       int rerankK,
                       List<Metric> logOutputs,
                       boolean compacted,
                       Integer numSplits,
                       String splitDistribution) {
        if (experimentsWriter == null || run == null) {
            return;
        }

        var fixed = ExperimentsSchemaV1.fixedValues(
                run,
                datasetName,
                M,
                efConstruction,
                neighborOverflow,
                addHierarchy,
                refineFinalGraph,
                featureSetForIndex,
                usePruning,
                topK,
                overquery,
                rerankK,
                compacted,
                numSplits,
                splitDistribution
        );

        try {
            experimentsWriter.appendRow(fixed, logOutputs);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to append experiments.csv row", e);
        }
    }

    public RunContext run() { return run; }
    public DatasetInfoWriter datasetInfoWriter() { return datasetInfoWriter; }
    public ExperimentsCsvWriter experimentsWriter() { return experimentsWriter; }

    public Map<String, List<String>> benchmarksToCompute() { return benchmarksToCompute; }
    public Map<String, List<String>> benchmarksToDisplay() { return benchmarksToDisplay; }
    public MetricSelection metricsToDisplay() { return metricsToDisplay; }
    public Map<String, List<String>> benchmarksToLog() { return benchmarksToLog; }
    public MetricSelection metricsToLog() { return metricsToLog; }

    public void registerDataset(String datasetName, DataSet ds) throws IOException {
        if (datasetInfoWriter == null) {
            return; // disabled
        }

        var mfd = DataSetLoaderMFD.MultiFileDatasource.byName.get(datasetName);

        String basePath = "";
        String queryPath = "";
        String gtPath = "";
        if (mfd != null) {
            basePath = Paths.get("fvec").resolve(mfd.basePath).toAbsolutePath().toString();
            queryPath = Paths.get("fvec").resolve(mfd.queriesPath).toAbsolutePath().toString();
            gtPath = Paths.get("fvec").resolve(mfd.groundTruthPath).toAbsolutePath().toString();
        }

        datasetInfoWriter.register(DatasetInfoWriter.fromDataSet(datasetName, basePath, queryPath, gtPath, ds));
    }

    @Override
    public void close() {
        if (sysStatsCollector != null) {
            try {
                sysStatsCollector.stop(run.runDir());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (jfrRecorder != null) {
            jfrRecorder.stop();
        }
    }
}
