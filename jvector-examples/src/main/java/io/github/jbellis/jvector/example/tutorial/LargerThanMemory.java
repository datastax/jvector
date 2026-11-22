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
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

// Provides the code used in 3-larger-than-memory-tutorial.md.
// If you edit this file you may also want to edit 3-larger-than-memory-tutorial.md
public class LargerThanMemory {
    public static void main(String[] args) throws IOException {
        // The DataSet provided by loadDataSet is in-memory,
        // but you can apply the same technique even when you don't have
        // the base vectors in-memory.
        String datasetName = args.length == 0 ? "e5-small-v2-100k" : args[0];
        DataSet dataset = DataSets.loadDataSet(datasetName).orElseThrow(() ->
            new RuntimeException("Dataset doesn't exist or wasn't configured correctly")
        ).getDataSet();

        // RAVVs need not be in-memory, but this source must remain readable through final writing.
        // We will sample from this RAVV to compute the PQ codebooks.
        // In general you don't need to have a RAVV over all the vectors to
        // build PQ codebooks, but you do need a "representative set".
        RandomAccessVectorValues ravv = dataset.getBaseRavv();
        VectorSimilarityFunction vsf = dataset.getSimilarityFunction();
        int dim = dataset.getDimension();

        // PQ parameters
        int subspaces = 64;  // number of subspaces to divide each vector into
        int centroidsPerSubspace = 256;  // number of centroids per subspace (256 => 1 byte)
        boolean centerDataset = false;  // we won't ask to center the dataset before quantization

        System.out.println("Computing PQ codebooks...");
        // This method randomly samples at most MAX_PQ_TRAINING_SET_SIZE vectors
        // from the RAVV and considers that a "representative set" used to build the codebooks.
        ProductQuantization pq = ProductQuantization.compute(ravv, subspaces, centroidsPerSubspace, centerDataset);

        // MutablePQVectors is a thread-safe, dynamically growing container for compressed vectors.
        // As we add vectors to the index, we'll compress them and store them here.
        // These compressed vectors are used during graph construction for approximate distance calculations.
        var pqVectors = new MutablePQVectors(pq);

        // Provides approximate scores during graph construction using the compressed vectors
        BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(vsf, pqVectors);

        // Graph construction parameters
        int M = 32;
        int ef = 100;
        float overflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = true;

        Path graphPath = Files.createTempFile("jvector-ltm-graph", null);

        System.out.println("Building index incrementally...");

        try (
            GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, M, ef, overflow, alpha, addHierarchy)
        ) {
            // Independent views also support sources that borrow a scratch vector.
            var vectors = ravv.threadLocalSupplier();
            // Graph building is best done with threads = number of physical cores
            // PhysicalCoreExecutor assumes hyperthreading by default, i.e. cores = vCPUs / 2
            // If this is not correct, set the system property `jvector.physical_core_count`
            PhysicalCoreExecutor.pool().submit(() -> {
                IntStream.range(0, ravv.size()).parallel().forEach(ordinal -> {
                    VectorFloat<?> v = vectors.get().getVector(ordinal);

                    // Encode and add the vector to the working set of PQ vectors,
                    // which allows the graph builder to access it through the BuildScoreProvder.
                    pqVectors.encodeAndSet(ordinal, v);

                    builder.addGraphNode(ordinal, v);
                });
            }).join();

            // Must be done manually for incrementally built graphs.
            // Enforces maximum degree constraint among other things.
            builder.cleanup();

            // Reread this replayable source for final FP payloads; peers were resident PQ.
            try (var writer = new OnDiskParallelGraphIndexWriter.Builder(builder.getGraph(), graphPath)
                    .withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount())
                    .with(new InlineVectors(dim))
                    .withMapper(new OrdinalMapper.IdentityMapper(ravv.size() - 1))
                    .build()) {
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        ordinal -> new InlineVectors.State(vectors.get().getVector(ordinal))));
            }
        }

        // PQ codebooks and vectors also need to be saved somewhere! (we'll not concern ourselves with Fused PQ)
        Path pqPath = Files.createTempFile("jvector-ltm-pq", null);
        try (var pqOut = new BufferedRandomAccessWriter(pqPath)) {
            pqVectors.write(pqOut);
        }

        System.out.println("Index built successfully!");
        
        // Calculate and display the compression ratio
        int compressedSize = pq.compressedVectorSize();
        int originalSize = dim * Float.BYTES;
        float compressionRatio = (float) originalSize / compressedSize;
        System.out.println(String.format("Compression ratio: %.1fx (%d bytes -> %d bytes per vector)",
                                        compressionRatio, originalSize, compressedSize));

        System.out.println("\nSearching with two-pass approach...");

        try (
            // nothing new here
            ReaderSupplier graphSupplier = ReaderSupplierFactory.open(graphPath);
            OnDiskGraphIndex graph = OnDiskGraphIndex.load(graphSupplier);
            // except that we also need a reader for the PQ vectors
            ReaderSupplier pqSupplier = ReaderSupplierFactory.open(pqPath);
            RandomAccessReader pqReader = pqSupplier.get()
        ) {
            // we need to have the PQ vectors in memory
            PQVectors pqVectorsSearch = PQVectors.load(pqReader);

            int topK = 10;
            for (float overqueryFactor : new float[]{4.0f, 8.0f, 16.0f, 32.0f, 64.0f}) {
                int rerankK = (int) (topK * overqueryFactor);

                try (GraphSearcher searcher = new GraphSearcher(graph)) {
                    var graphRavv = (RandomAccessVectorValues) searcher.getView();

                    List<SearchResult> results = new ArrayList<>();
                    for (VectorFloat<?> query : dataset.getQueryVectors()) {
                        // Two-phase search:
                        // 1. ApproximateScoreFunction (ASF) uses compressed vectors for fast initial search
                        // 2. Reranker uses full-resolution vectors from disk for accurate final ranking
                        var asf = pqVectorsSearch.precomputedScoreFunctionFor(query, vsf);
                        var reranker = graphRavv.rerankerFor(query, vsf);
                        SearchScoreProvider ssp = new DefaultSearchScoreProvider(asf, reranker);

                        SearchResult sr = searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
                        results.add(sr);
                    }

                    double recall = AccuracyMetrics.recallFromSearchResults(dataset.getGroundTruth(), results, topK, topK);
                    System.out.println(String.format("Recall@%d for overquery by %f = %f", topK, overqueryFactor, recall));
                }
            }
        }

        // cleanup after ourselves
        Files.deleteIfExists(graphPath);
        Files.deleteIfExists(pqPath);
    }
}
