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
package io.github.jbellis.jvector.bench.compaction;

import io.github.jbellis.jvector.bench.compaction.Cli;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.function.IntFunction;
import java.util.ArrayList;

public class BuildFromScratch {
    private static final Logger log = LoggerFactory.getLogger(BuildFromScratch.class);

    public enum IndexPrecision { FULLPRECISION, FUSEDPQ }

    public static void main(String[] args) throws Exception {
        var cli = new Cli(args);

        var similarityFunction = VectorSimilarityFunction.COSINE;
        String baseVecPath = cli.get("baseVectorPath", "");
        boolean enableRecall = cli.getBool("enableRecall", false);
        String datasetName = cli.get("datasetName", "");
        String queriesPath = cli.get("queriesPath", "");
        String groundTruthPath = cli.get("groundTruthPath", "");
        int graphDegree = cli.getInt("graphDegree", 32);
        int beamWidth = cli.getInt("beamWidth", 100);

        String indexPrecisionStr = cli.get("indexPrecision", "FULLPRECISION");
        IndexPrecision indexPrecision = IndexPrecision.valueOf(indexPrecisionStr);

        int parallelWriteThreads = cli.getInt("parallelWriteThreads", 1);

        String vectorizationProvider = cli.get("vectorizationProvider", "");
        if (vectorizationProvider != null && !vectorizationProvider.isBlank()) {
            System.setProperty("jvector.vectorization_provider", vectorizationProvider);
        }
        String resolvedVP = VectorizationProvider.getInstance().getClass().getSimpleName();

        String outputPathStr = cli.get("outputPath", "target/scratch-graph");
        Path outputPath = Path.of(outputPathStr);
        //Files.createDirectories(outputPath.getParent());
        if (Files.exists(outputPath)) Files.delete(outputPath);
        var baseVectors = SiftLoader.readFvecs(baseVecPath);
        int dimension = baseVectors.get(0).length();

        var full = new ListRandomAccessVectorValues(baseVectors, dimension);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(full, similarityFunction);

        log.info("Building from scratch: dataset={} vectors={} dim={} sim={} deg={} bw={} precision={} pwThreads={} vp={} -> {}",
                datasetName, full.size(), dimension, similarityFunction,
                graphDegree, beamWidth, indexPrecision, parallelWriteThreads, resolvedVP, outputPath.toAbsolutePath());

        var builder = new GraphIndexBuilder(bsp, dimension, graphDegree, beamWidth, 1.2f, 1.2f, true);
        var graph = builder.build(full);

        AbstractGraphIndexWriter.Builder<?, ?> writerBuilder =
                (parallelWriteThreads > 1)
                        ? new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
                            .withParallelWorkerThreads(parallelWriteThreads)
                        : new OnDiskGraphIndexWriter.Builder(graph, outputPath);

        writerBuilder.with(new InlineVectors(dimension));

        ProductQuantization pq = null;
        PQVectors pqVectors = null;
        if (indexPrecision == IndexPrecision.FUSEDPQ) {
            boolean centerData = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
            pq = ProductQuantization.compute(full, dimension / 8, 256, centerData);
            pqVectors = (PQVectors) pq.encodeAll(full);
            writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
        }

        try (var writer = writerBuilder.build()) {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(full.getVector(ord)));

            if (indexPrecision == IndexPrecision.FUSEDPQ) {
                var view = graph.getView();
                var finalPQ = pqVectors;
                suppliers.put(FeatureId.FUSED_PQ, ord -> new FusedPQ.State(view, finalPQ, ord));
            }

            writer.write(suppliers);
        }

        log.info("Done. Index written at {}", outputPath.toAbsolutePath());

        if(enableRecall) {
            log.info("Enable recall. Load index from {}", outputPath.toAbsolutePath());
            var queryVectors = SiftLoader.readFvecs(queriesPath);
            var groundTruth = SiftLoader.readIvecs(groundTruthPath);
            var ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

            try (var rs = ReaderSupplierFactory.open(outputPath)) {
                var compactGraph = OnDiskGraphIndex.load(rs);
                List<SearchResult> compactedRetrieved = new ArrayList<>();
                for (int n = 0; n < queryVectors.size(); ++n) {
                    compactedRetrieved.add(GraphSearcher.search(queryVectors.get(n),
                            10,
                            ravv,
                            similarityFunction,
                            compactGraph,
                            Bits.ALL));
                }
                var recall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactedRetrieved, 10, 10);
                log.info("Recall [graphDegree={}, beamWidth={}, indexPrecision={}, parallelWriteThreads={}, vectorizationProvider={}]: {}",
                        graphDegree, beamWidth, indexPrecision, parallelWriteThreads, resolvedVP, recall);
            }
        }
    }
}
