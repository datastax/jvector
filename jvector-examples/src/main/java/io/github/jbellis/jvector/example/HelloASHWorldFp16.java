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

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.VectorMean;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.IntStream;

/** A first ASH graph: public 100k vectors, compressed disk storage, and top-10 search.
 *  Uses native FP16 query projection on Intel Sapphire Rapids or newer FP16 hardware.
 */
public class HelloASHWorldFp16 {
    public static void main(String[] args) throws Exception {
        // Select before loading any model. Training/encoding still use FP32.
        System.setProperty("jvector.vectorization_provider", "native");
        System.setProperty("jvector.ash.queryProjection", "fp16");
        Path path = args.length == 0 ? Path.of("ash-fp16.graph") : Path.of(args[0]);
        if (Files.exists(path)) throw new IllegalArgumentException("Choose a new index path: " + path);

        // 1. Download/cache a public dataset: normalized 384-dimensional embeddings.
        try (var dataset = DataSets.loadDataSet("e5-small-v2-100k").orElseThrow().getDataSet()) {
            var base = dataset.getBaseRavv();
            var similarity = dataset.getSimilarityFunction();
            System.out.printf("Loaded %,d vectors, %d dimensions%n", base.size(), base.dimension());

            // 2. Learn ASH(b=2,d=192): half as many dimensions, 2 stored bits each.
            // The API's bit budget includes a small per-vector header (40 bits).
            int b = 2, d = base.dimension() / 2;
            var ash = AsymmetricHashing.initialize(base, AsymmetricHashing.ITQ,
                    AsymmetricHashing.HEADER_BITS + b * d, 1, b);
            // Collect NVQ's mean during encoding instead of scanning the source again.
            var mean = new VectorMean(base.size(), base.dimension());
            var codes = ash.encodeAll(mean.duringScan(base), PhysicalCoreExecutor.pool());
            var nvq = NVQuantization.create(mean.mean(), 1);

            // 3. Construct a hierarchical graph using symmetric ASH code comparisons.
            var scores = BuildScoreProvider.ashBuildScoreProvider(similarity, codes);
            try (var builder = new GraphIndexBuilder(scores, base.dimension(),
                    32, 100, 1.2f, 1.2f, true, true)) {
                PhysicalCoreExecutor.pool().submit(() ->
                        IntStream.range(0, base.size()).parallel().forEach(node ->
                                builder.addGraphNode(node, scores.searchProviderFor(node)))).join();
                builder.cleanup();

                // 4. Save neighbor-local ASH blocks plus NVQ vectors for reranking.
                // Encode NVQ while the parallel writer emits complete records.
                var vectors = base.threadLocalSupplier();
                try (var views = ExplicitThreadLocal.withInitial(() -> builder.getGraph().getView());
                     var writer = new OnDiskParallelGraphIndexWriter.Builder(builder.getGraph(), path)
                             .withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount())
                             .withMapper(new OrdinalMapper.IdentityMapper(base.size() - 1))
                             .with(new FusedASH(32, ash)).with(new NVQ(nvq)).build()) {
                    writer.write(Map.of(
                            FeatureId.FUSED_ASH, node -> new FusedASH.State(views.get(), codes, node),
                            FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvq.encode(vectors.get().getVector(node)))));
                }
            }
            System.out.println("Saved " + path.toAbsolutePath());

            // 5. Reopen the self-contained index; the model is stored in its header.
            try (var readers = ReaderSupplierFactory.open(path);
                 var graph = OnDiskGraphIndex.load(readers);
                 var searcher = new GraphSearcher(graph)) {
                var preparer = graph.queryPreparer(similarity).orElseThrow();
                var prepared = preparer.newState();
                preparer.prepare(dataset.getQueryVectors().get(0), prepared);
                var view = (ImmutableGraphIndex.ScoringView) searcher.getView();
                // Fused ASH finds candidates; NVQ reranks 20 candidates down to 10.
                // One searcher/prepared state per in-flight query; reuse after completion.
                var result = searcher.search(view.scoreProviderFor(prepared, true),
                        10, 20, 0.0f, 0.0f, Bits.ALL);
                System.out.println("Synchronous top 10 for query 0 (vector ordinal, similarity score):");
                for (var hit : result.getNodes()) System.out.printf("  %d  %.6f%n", hit.node, hit.score);
                System.out.println("Visited nodes: " + result.getVisitedCount());

                // 6. The same search through the queued API. The application assigns
                // two search workers; the queue uses admission=8C and maximum batch=8.
                // Submit eight independent queries without waiting between submissions.
                var queued = HelloASHQueuedSearch.search(graph, similarity,
                        dataset.getQueryVectors().subList(0, 8), 2);
                System.out.println("Queued searches completed: " + queued.size());
                System.out.println("Queued top 10 for query 0 (vector ordinal, similarity score):");
                for (var hit : queued.get(0).getNodes()) System.out.printf("  %d  %.6f%n", hit.node, hit.score);
                System.out.println("Visited nodes: " + queued.get(0).getVisitedCount());
            }
        }
    }
}
