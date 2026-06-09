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

package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import me.tongfei.progressbar.ProgressBar;

// Demonstrates using Non-uniform Vector Quantization (NVQ) for reducing the footprint of the disk graph.
public class NvqExample {
    public static void main(String[] args) throws IOException {
        // Load a preconfigured dataset
        var ds = DataSets.loadDataSet("ada002-100k").orElseThrow(() -> 
            new RuntimeException("dataset not found"))
        .getDataSet();
        var dim = ds.getDimension();
        var vsf = ds.getSimilarityFunction();
        var base = ds.getBaseRavv();

        var numSubVectors = 2;

        // Setup NVQ parameters.
        // The base vectors RAVV instance is used only for computing the global mean
        var nvq = NVQuantization.compute(base, numSubVectors);
        // Use this method instead if you don't have all the vectors up-front but can estimate the mean
        // var nvq = NVQuantization.create(scaledGlobalMean, numSubVectors);

        // Graph construction parameters
        var M = 32;
        var ef = 100;
        var nOv = 1.2f;
        var alpha = 1.2f;
        var addHierarchy = true;

        var pqMFactor = 8;
        var pqM = (ds.getDimension() + pqMFactor - 1) / pqMFactor;
        var pqClusterCount = 256;
        var pqGloballyCenter = false;

        // PQ is used for graph building and first-stage scoring during query
        var pq = ProductQuantization.compute(base, pqM, pqClusterCount, pqGloballyCenter);

        // Empty PQVectors instance, will be updated as we stream in vectors
        var pqv = new MutablePQVectors(pq);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(vsf, pqv);

        Path graphPath = Files.createTempFile("jvector-nvq-graph", null);

        System.out.println("Building graph in streaming mode...");
        try (
            // Create the graph builder using PQ-based scoring
            var builder = new GraphIndexBuilder(bsp, dim, M, ef, nOv, alpha, addHierarchy);
            // Create the on-disk writer configured with NVQ feature
            // This allows us to write both the graph structure and NVQ-compressed vectors
            var writer = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), graphPath)
                .with(new NVQ(nvq))
                .withMapper(new OrdinalMapper.IdentityMapper(base.size() - 1))
                .build();
            var pb = new ProgressBar("Build graph", base.size());
        ) {

            PhysicalCoreExecutor.pool().submit(() -> {
                IntStream.range(0, base.size())
                    .parallel()
                    .forEach(ordinal -> {
                        var vec = base.getVector(ordinal);
                        
                        // Encode the PQ vector first, then add the graph node
                        pqv.encodeAndSet(ordinal, vec);
                        builder.addGraphNode(ordinal, vec);

                        // Encode and write NVQ vectors for later re-ranking
                        var nvqVec = nvq.encode(vec);
                        Map<FeatureId, Feature.State> featureMap = Map.of(
                            FeatureId.NVQ_VECTORS, new NVQ.State(nvqVec)
                        );
                        try {
                            writer.writeFeaturesInline(ordinal, featureMap);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                        pb.step();
                    });
            }).join();
            pb.close();

            // cleanup
            System.out.println("Cleanup...");
            builder.cleanup();
            writer.write(Map.of());
        }

        // Search parameters
        var topK = 10;
        var rerankK = 100;

        List<SearchResult> results;

        System.out.println("Loading and searching the graph...");
        try (
            var rs = ReaderSupplierFactory.open(graphPath);
            var graph = OnDiskGraphIndex.load(rs);
            var searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(graph));
        ) {
            results = ds.getQueryVectors()
                .parallelStream()
                .map(query -> {
                    var searcher = searchers.get();
                    var scoringView = (ImmutableGraphIndex.ScoringView) searcher.getView();

                    // Two-phase search with NVQ:
                    // 1. Use PQ for fast approximate search to get rerankK candidates
                    var asf = pqv.precomputedScoreFunctionFor(query, vsf);
                    // 2. Use NVQ-compressed vectors from disk for accurate reranking to topK
                    // The reranker automatically uses the NVQ vectors stored in the graph
                    var reranker = scoringView.rerankerFor(query, vsf);
                    var ssp = new DefaultSearchScoreProvider(asf, reranker);
                    return searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
                })
                .collect(Collectors.toList());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Evaluate search accuracy
        var recall = AccuracyMetrics.recallFromSearchResults(ds.getGroundTruth(), results, topK, topK);
        System.out.println("Recall: " + recall);

        // cleanup
        Files.deleteIfExists(graphPath);
    }
}
