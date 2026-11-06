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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.BenchmarkTablePrinter;
import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.datasets.BaseVectorLoading;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetInfo;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Interactive IVF benchmark, the counterpart of jvector-examples' {@code BenchYAML}. Run it from the
 * repository root. Arguments are dataset regexes, matched against every dataset in datasets.yml
 * (no arguments matches them all), and/or IVF config files ({@code *.yml}). run-config.yml chooses
 * what is computed, shown and logged.
 */
public class BenchIvfYAML {

    public static void main(String[] args) throws IOException {
        if (args == null) {
            throw new IllegalArgumentException("argv[] is null, check your exec configuration");
        }
        List<IvfConfig> configs = configsFor(args, DatasetCollection.load().getAll());
        if (configs.isEmpty()) {
            System.out.println("No datasets matched. Exiting.");
            return;
        }
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
        System.out.println("Executing: " + configs.stream().map(c -> c.dataset).collect(Collectors.toList()));

        RunConfig runCfg;
        try {
            runCfg = RunConfig.loadDefault();
        } catch (FileNotFoundException e) {
            System.err.println("WARNING: run-config.yml not found; computing default benchmarks, no logging.");
            runCfg = null;
        }
        IvfReporting reporting = IvfReporting.open(runCfg, configs);
        List<IvfQueryBenchmark> benchmarks = reporting.benchmarks();

        for (IvfConfig config : configs) {
            runDataset(config, benchmarks, reporting);
        }
    }

    /**
     * IVF configs for the arguments: one per matching dataset (its own ivf-parameters file, else
     * default.yml), then one per {@code .yml} argument. With no arguments at all, every dataset.
     */
    static List<IvfConfig> configsFor(String[] args, List<String> allDatasets) throws FileNotFoundException {
        List<String> words = Arrays.stream(args)
                .filter(Objects::nonNull)
                .flatMap(s -> Arrays.stream(s.split("\\s")))
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
        List<String> configFiles = words.stream().filter(w -> w.endsWith(".yml")).collect(Collectors.toList());
        List<String> regexes = words.stream().filter(w -> !w.endsWith(".yml")).collect(Collectors.toList());

        List<IvfConfig> configs = new ArrayList<>();
        if (!regexes.isEmpty() || configFiles.isEmpty()) {
            Pattern pattern = Pattern.compile(regexes.isEmpty() ? ".*" : String.join("|", regexes));
            for (String name : allDatasets) {
                if (pattern.matcher(name).find()) {
                    configs.add(IvfConfig.getDefaultConfig(stripHdf5(name)));
                }
            }
        }
        for (String file : configFiles) {
            configs.add(IvfConfig.getConfig(file));
        }
        return configs;
    }

    private static String stripHdf5(String name) {
        return name.endsWith(".hdf5") ? name.substring(0, name.length() - ".hdf5".length()) : name;
    }

    private static void runDataset(IvfConfig config, List<IvfQueryBenchmark> benchmarks, IvfReporting reporting)
            throws IOException {
        DataSetInfo info = DataSets.loadDataSet(config.dataset, BaseVectorLoading.forBenchmark(false))
                .orElseThrow(() -> new RuntimeException("Could not load dataset: " + config.dataset));
        // Check before loading the vectors, so an unsupported dataset costs nothing
        Optional<String> unsupported = info.similarityFunction().flatMap(IvfGrid::unsupportedReason);
        if (unsupported.isPresent()) {
            System.out.println("Skipping dataset " + config.dataset + ": " + unsupported.get());
            return;
        }
        try (DataSet ds = info.getDataSet()) {
            reporting.registerDataset(config.dataset, ds);
            var sink = new ConsoleSink(config, reporting);
            IvfGrid.run(ds, config, benchmarks, sink);
            sink.finish();
        }
    }

    /** Prints a configuration block per construction point and a table per topK, and logs each row. */
    private static final class ConsoleSink implements IvfGrid.RowSink {
        private final IvfConfig config;
        private final IvfReporting reporting;
        private final IvfTablePrinter table = new IvfTablePrinter(System.out);
        private Integer currentTopK;

        ConsoleSink(IvfConfig config, IvfReporting reporting) {
            this.config = config;
            this.reporting = reporting;
        }

        @Override
        public void startConstructionPoint(IvfSystem system) {
            if (currentTopK != null) {
                table.printFooter();
            }
            currentTopK = null;
            var index = new LinkedHashMap<String, Object>();
            index.put("dataset", system.dataSet().getName());
            index.put("nlist", system.constructionPoint().nlist());
            index.put("storageFactor", system.constructionPoint().storageFactor());
            index.put("buildThreads", config.construction.buildThreads == 0 ? "all CPUs" : config.construction.buildThreads);
            index.put("ssdPaths", config.construction.ssdPaths);
            index.put("index", system.reopened() ? "reopened" : "built");
            var query = new LinkedHashMap<String, Object>();
            query.put("searchThreads", config.search.searchThreads);
            query.put("queryThreads", config.search.queryThreads);
            new BenchmarkTablePrinter().printConfig(index, query);
        }

        @Override
        public void accept(IvfSystem system, SearchPoint point, List<Metric> metrics) {
            if (currentTopK == null || currentTopK != point.topK()) {
                if (currentTopK != null) {
                    table.printFooter();
                }
                currentTopK = point.topK();
                System.out.println("\ntopK = " + currentTopK);
                table.resetTable();
            }
            table.printRow(point.nprobe(), point.searchThreads(), point.queryThreads(), reporting.forConsole(metrics, point.topK()));
            try {
                reporting.log(system, config, point, metrics);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        void finish() {
            if (currentTopK != null) {
                table.printFooter();
            }
        }
    }
}
