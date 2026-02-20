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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

// Provides the code used in disk-tutorial.md.
// If you edit this file you may also want to edit disk-tutoral.md
public class DiskIntro {
    public static void main(String[] args) throws IOException {
        // This is a preconfigured dataset that will be downloaded automatically.
        DataSet dataset = DataSets.loadDataSet("ada002-100k").orElseThrow(() ->
            new RuntimeException("Dataset doesn't exist or wasn't configured correctly")
        ).getDataSet();

        // The loaded DataSet provides a RAVV over the base vectors
        RandomAccessVectorValues ravv = dataset.getBaseRavv();
        VectorSimilarityFunction vsf = dataset.getSimilarityFunction();
        int dim = dataset.getDimension();

        // reasonable defaults
        int M = 32;
        int ef = 100;
        float overflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = true;

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);

        System.out.println("Building the graph, may take a few minutes");

        // nothing new here
        ImmutableGraphIndex heapGraph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, M, ef, overflow, alpha, addHierarchy)) {
            heapGraph = builder.build(ravv);
        }

        Path graphPath = Files.createTempFile("jvector-example-graph", null);  // or wherever you want to save the graph
        try (
            // Create a writer for the on-heap graph we just built.
            GraphIndexWriter writer = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, heapGraph, graphPath)
                // Let the writer know that we'll also be passing in the actual vector data
                // to be saved "inline" with the data for each corresponding graph node.
                .with(new InlineVectors(dim))
                .build();
        ) {
            // Supply one map entry for each feature.
            // The key is a FeatureId enum corresponding to the feature
            // and the value is a function which generates the feature state for each graph node.
            writer.write(Map.of(
                FeatureId.INLINE_VECTORS,
                // we already have a RAVV, so we'll just use that to supply the writer.
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))));
        }

        // ReaderSupplierFactory automatically picks an available RandomAccessReader implementation
        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath);
        OnDiskGraphIndex graph = OnDiskGraphIndex.load(readerSupplier);

        System.out.println("Performing searches with increasing overquery factor");

        // number of search results we want
        int topK = 10;
        for (float overqueryFactor : new float[]{1.0f, 1.5f, 2.0f, 5.0f, 10.0f}) {
            // `rerankK` controls the number of nodes to fetch from the initial graph search.
            // which are then re-ranked to return the actual topK results.
            // Increasing rerankK improves accuracy at the cost of latency and throughput.
            int rerankK = (int) (topK * overqueryFactor);

            try (GraphSearcher searcher = new GraphSearcher(graph)) {
                // Views of an OnDiskGraphIndex with inline or separated vectors can be used as RAVVs!
                // In multi-threaded scenarios you should have one searcher per thread
                // and extract a view for each thread from the associated searcher.
                var graphRavv = (RandomAccessVectorValues) searcher.getView();

                List<SearchResult> results = new ArrayList<>();
                for (VectorFloat<?> query : dataset.getQueryVectors()) {
                    // use the RAVV from the graph instead of the one from the original dataSet
                    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(query, vsf, graphRavv);
                    // A slightly more complex overload of `search` which adds three extra parameters.
                    // Right now we only care about `rerankK`.
                    SearchResult sr = searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
                    results.add(sr);
                }

                double recall = AccuracyMetrics.recallFromSearchResults(dataset.getGroundTruth(), results, topK, topK);
                System.out.println(String.format("Recall@%d for overquery by %f = %f", topK, overqueryFactor, recall));
            }
        }

        // cleanup
        readerSupplier.close();
        Files.deleteIfExists(graphPath);
    }
}
