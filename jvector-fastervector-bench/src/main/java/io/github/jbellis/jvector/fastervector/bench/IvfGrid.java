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

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.ConstructionPoint;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The IVF counterpart of jvector-examples' {@code Grid}: for each construction point in an
 * {@link IvfConfig}, build or reopen the index, then run every benchmark at every search point
 * and hand each row of metrics to a {@link RowSink}.
 */
public final class IvfGrid {
    private static final Logger log = LoggerFactory.getLogger(IvfGrid.class);

    private IvfGrid() {}

    /** Receives one row per construction point and search point. */
    @FunctionalInterface
    public interface RowSink {
        /** Called once per construction point, before its rows. */
        default void startConstructionPoint(IvfSystem system) {}

        void accept(IvfSystem system, SearchPoint point, List<Metric> metrics);
    }

    /**
     * Why fastervector can't run a dataset with this similarity function, if it can't.
     * fastervector scores by inner product: COSINE datasets are normalized on load and DOT_PRODUCT
     * is native, but EUCLIDEAN ordering differs from inner product.
     */
    public static Optional<String> unsupportedReason(VectorSimilarityFunction similarity) {
        if (similarity == VectorSimilarityFunction.EUCLIDEAN) {
            return Optional.of("fastervector scores by inner product only; EUCLIDEAN is not supported");
        }
        return Optional.empty();
    }

    /**
     * Runs {@code config} against {@code ds}. Skips the dataset, with a log message, if its
     * similarity function is unsupported.
     *
     * @return false if the dataset was skipped
     */
    public static boolean run(DataSet ds, IvfConfig config, List<IvfQueryBenchmark> benchmarks, RowSink sink)
            throws IOException {
        Optional<String> reason = unsupportedReason(ds.getSimilarityFunction());
        if (reason.isPresent()) {
            log.warn("Skipping dataset {}: {}", ds.getName(), reason.get());
            return false;
        }
        List<SearchPoint> searchPoints = config.searchPoints();
        for (ConstructionPoint point : config.constructionPoints()) {
            try (IvfSystem system = IvfSystem.buildOrOpen(ds, config, point)) {
                sink.startConstructionPoint(system);
                List<Metric> constructionMetrics = system.constructionMetrics();
                for (SearchPoint searchPoint : searchPoints) {
                    List<Metric> metrics = new ArrayList<>();
                    for (IvfQueryBenchmark benchmark : benchmarks) {
                        metrics.addAll(benchmark.runBenchmark(system, searchPoint));
                    }
                    metrics.addAll(constructionMetrics);
                    sink.accept(system, searchPoint, metrics);
                }
            }
        }
        return true;
    }
}
