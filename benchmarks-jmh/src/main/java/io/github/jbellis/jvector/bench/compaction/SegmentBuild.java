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
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.DataSetPartitioner;
import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.function.IntFunction;

public class SegmentBuild {
    private static final Logger log = LoggerFactory.getLogger(SegmentBuild.class);

    public enum IndexPrecision { FULLPRECISION, FUSEDPQ }

    public static void main(String[] args) throws Exception {
        var cli = new Cli(args);

        String dataset = cli.get("dataset", "glove-100-angular");
        int numSources = cli.getInt("numSources", 4);
        int graphDegree = cli.getInt("graphDegree", 32);
        int beamWidth = cli.getInt("beamWidth", 100);
        double datasetPortion = cli.getDouble("datasetPortion", 1.0);

        String splitDistStr = cli.get("splitDistribution", "UNIFORM");
        TestDataPartition.Distribution splitDistribution = TestDataPartition.Distribution.valueOf(splitDistStr);

        String indexPrecisionStr = cli.get("indexPrecision", "FULLPRECISION");
        IndexPrecision indexPrecision = IndexPrecision.valueOf(indexPrecisionStr);

        int parallelWriteThreads = cli.getInt("parallelWriteThreads", 1);

        String vectorizationProvider = cli.get("vectorizationProvider", "");
        if (vectorizationProvider != null && !vectorizationProvider.isBlank()) {
            System.setProperty("jvector.vectorization_provider", vectorizationProvider);
        }
        String resolvedVP = VectorizationProvider.getInstance().getClass().getSimpleName();

        String outDirStr = cli.get("outDir", "target/segments");
        Path outDir = Path.of(outDirStr);
        Files.createDirectories(outDir);

        // optional cleanup
        boolean clean = cli.getBool("clean", false);
        if (clean) {
            for (int i = 0; i < Math.max(1, numSources); i++) {
                Path p = outDir.resolve("per-source-graph-" + i);
                if (Files.exists(p)) Files.delete(p);
            }
        }

        if (numSources < 0) throw new IllegalArgumentException("numSources must be non-negative");
        int numParts = (numSources == 0) ? 1 : numSources;

        DataSet ds = DataSets.loadDataSet(dataset)
                .orElseThrow(() -> new RuntimeException("Dataset not found: " + dataset));

        List<VectorFloat<?>> baseVectors = ds.getBaseVectors();
        if (datasetPortion != 1.0) {
            int total = ds.getBaseRavv().size();
            int portioned = (int) (total * datasetPortion);
            if (portioned < numParts) {
                throw new IllegalArgumentException("datasetPortion yields fewer vectors than numSources");
            }
            baseVectors = baseVectors.subList(0, portioned);
        }

        var partitioned = DataSetPartitioner.partition(baseVectors, numParts, splitDistribution);

        log.info("Building {} segments for dataset={} dim={} sim={} deg={} bw={} precision={} pwThreads={} vp={}",
                numParts, dataset, ds.getDimension(), ds.getSimilarityFunction(),
                graphDegree, beamWidth, indexPrecision, parallelWriteThreads, resolvedVP);

        for (int i = 0; i < numParts; i++) {
            List<VectorFloat<?>> vectorsPerSource = partitioned.vectors.get(i);
            Path outputPath = outDir.resolve("per-source-graph-" + i);

            log.info("Segment {}/{}: vectors={} -> {}", i + 1, numParts, vectorsPerSource.size(), outputPath.toAbsolutePath());

            var ravvPerSource = new ListRandomAccessVectorValues(vectorsPerSource, ds.getDimension());
            var bsp = BuildScoreProvider.randomAccessScoreProvider(ravvPerSource, ds.getSimilarityFunction());

            var builder = new GraphIndexBuilder(bsp, ds.getDimension(), graphDegree, beamWidth, 1.2f, 1.2f, true);
            var graph = builder.build(ravvPerSource);

            AbstractGraphIndexWriter.Builder<?, ?> writerBuilder =
                    (parallelWriteThreads > 1)
                            ? new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
                                .withParallelWorkerThreads(parallelWriteThreads)
                            : new OnDiskGraphIndexWriter.Builder(graph, outputPath);

            writerBuilder.with(new InlineVectors(ds.getDimension()));

            ProductQuantization pq = null;
            PQVectors pqVectors = null;
            if (indexPrecision == IndexPrecision.FUSEDPQ) {
                boolean centerData = ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
                pq = ProductQuantization.compute(ravvPerSource, ds.getDimension() / 8, 256, centerData);
                pqVectors = (PQVectors) pq.encodeAll(ravvPerSource);
                writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
            }

            try (var writer = writerBuilder.build()) {
                var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(ravvPerSource.getVector(ord)));

                if (indexPrecision == IndexPrecision.FUSEDPQ) {
                    var view = graph.getView();
                    var finalPQ = pqVectors;
                    suppliers.put(FeatureId.FUSED_PQ, ord -> new FusedPQ.State(view, finalPQ, ord));
                }

                writer.write(suppliers);
            }
        }

        log.info("Done. Segments written under {}", outDir.toAbsolutePath());
    }
}
