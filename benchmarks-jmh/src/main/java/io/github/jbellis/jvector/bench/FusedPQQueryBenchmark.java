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

package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import java.util.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Threads(1)
public class FusedPQQueryBenchmark {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private OnDiskGraphIndex index;
    private ArrayList<VectorFloat<?>> queryVectors;
    private Path indexPath;
    private Path tempDir;

    @Param({"1536"})
    int dimension;

    @Param({"96"})
    int pqM;

    @Param({"100000"})
    int numBaseVectors;

    @Param({"100"})
    int numQueryVectors;

    @Param({"10"})
    int topK;

    @Param({"100"})
    int efSearch;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        System.out.println("Setting up FusedPQ index...");

        // 1. Create base vectors
        var baseVectors = new ArrayList<VectorFloat<?>>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            baseVectors.add(createRandomVector(dimension));
        }
        RandomAccessVectorValues floatVectors = new ListRandomAccessVectorValues(baseVectors, dimension);

        // 2. Create query vectors
        queryVectors = new ArrayList<>(numQueryVectors);
        for (int i = 0; i < numQueryVectors; i++) {
            queryVectors.add(createRandomVector(dimension));
        }

        // 3. Compute PQ compression
        System.out.println("Computing PQ compression...");
        boolean centerData = false; // false for DOT_PRODUCT/COSINE
        var pq = ProductQuantization.compute(floatVectors, pqM, 256, centerData, UNWEIGHTED);
        var pqVectors = (PQVectors) pq.encodeAll(floatVectors);
        System.out.printf("PQ: %d subspaces, 256 clusters%n", pqM);

        // 4. Build graph with PQ-compressed vectors
        System.out.println("Building graph...");
        int M = 16;
        int efConstruction = 100;
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = true;
        boolean refineFinalGraph = true;

        var bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, pqVectors);
        var builder = new GraphIndexBuilder(bsp, dimension, M, efConstruction,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph);
        var graph = builder.build(floatVectors);
        System.out.printf("Graph built: %d nodes%n", graph.size(0));

        // 5. Write FusedPQ index to disk
        System.out.println("Writing FusedPQ index to disk...");
        tempDir = Files.createTempDirectory("fusedpq-bench");
        indexPath = tempDir.resolve("fusedpq-index");

        var fusedPQFeature = new FusedPQ(graph.maxDegree(), pq);
        var inlineVectors = new InlineVectors(dimension);

        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, indexPath)
                .with(fusedPQFeature)
                .with(inlineVectors)
                .withMapper(new OrdinalMapper.IdentityMapper(floatVectors.size() - 1))
                .build()) {

            var view = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
            suppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal));
            suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(floatVectors.getVector(ordinal)));

            writer.write(suppliers);
            view.close();
        }

        builder.close();
        System.out.printf("Index written: %.2f MB%n", Files.size(indexPath) / 1024.0 / 1024.0);

        // 6. Load the index
        System.out.println("Loading index...");
        index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(indexPath));
        System.out.println("Setup complete!");
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (index != null) {
            index.close();
        }
        if (indexPath != null && Files.exists(indexPath)) {
            Files.deleteIfExists(indexPath);
        }
        if (tempDir != null && Files.exists(tempDir)) {
            Files.deleteIfExists(tempDir);
        }
        if (queryVectors != null) {
            queryVectors.clear();
        }
    }

     @Benchmark
    public void queryFusedPQ(Blackhole blackhole) throws IOException {
        // Perform queries on all query vectors
        for (VectorFloat<?> queryVector : queryVectors) {
            try (var view = index.getView()) {
                var scoringView = (ImmutableGraphIndex.ScoringView) view;

                // Get score functions - FusedPQ for approximate, then rerank
                var asf = scoringView.approximateScoreFunctionFor(queryVector, VectorSimilarityFunction.DOT_PRODUCT);
                var reranker = scoringView.rerankerFor(queryVector, VectorSimilarityFunction.DOT_PRODUCT);
                var ssp = new io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider(asf, reranker);

                // Search
                var searcher = new GraphSearcher(index);
                SearchResult result = searcher.search(ssp, topK, efSearch, 1.0f, 0.0f, io.github.jbellis.jvector.util.Bits.ALL);

                blackhole.consume(result);
            }
        }
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}
