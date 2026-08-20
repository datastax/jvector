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

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.SQFeature;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.ScalarQuantizer;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Scalar Quantization (SQ) index build-and-search tutorial using siftsmall float32 vectors.
 *
 * Loads float32 base and query vectors from .fvecs files, fits a ScalarQuantizer,
 * quantizes to int8, builds the graph in one parallel shot, then saves the graph with
 * the ScalarQuantizer embedded in the index header via SQFeature. On reload, the
 * quantizer is recovered directly from the index — no sidecar file needed.
 *
 * Run via TutorialRunner:
 *   ./mvnw -pl jvector-examples -am -Pjdk22 compile exec:exec@tutorial -Dtutorial=sq
 */
public class SQExample {

    /** Siftsmall vectors are 128-dimensional. */
    private static final int DIM = 128;

    public static void main(String[] args) throws IOException {
        // ── 1. Load float32 base and query vectors from siftsmall fvecs files ──
        String siftPath = "siftsmall";
        List<VectorFloat<?>> baseVectors  = SiftLoader.readFvecs(siftPath + "/siftsmall_base.fvecs");
        List<VectorFloat<?>> queryVectors = SiftLoader.readFvecs(siftPath + "/siftsmall_query.fvecs");
        System.out.printf("Loaded %d base vectors and %d query vectors, dim=%d%n",
                baseVectors.size(), queryVectors.size(), DIM);

        // ── 2. Fit a ScalarQuantizer on the base vectors ──────────────────────
        // ScalarQuantizer.fit() scans all base vectors to compute per-dimension
        // min/max, then maps each float component linearly into [-128, 127].
        var floatRavv = new ListRandomAccessVectorValues(baseVectors, DIM);
        ScalarQuantizer sq = ScalarQuantizer.fit(floatRavv);
        System.out.println("Quantizer fitted: " + sq);

        // ── 3. Quantize all base vectors to signed int8 ───────────────────────
        RandomAccessByteVectorValues byteRavv = sq.quantizeAll(floatRavv);
        System.out.printf("Quantized %d base vectors to int8%n", byteRavv.size());

        // ── 4. Build the full graph in one shot ───────────────────────────────
        // builder.build(byteRavv) inserts all nodes in parallel and calls cleanup()
        // internally — no manual cleanup() needed.
        ImmutableGraphIndex graph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
                byteRavv, ByteVectorSimilarityFunction.DOT_PRODUCT, 16, 100, 1.2f, 1.2f, true)) {
            graph = builder.build(byteRavv);
        }
        System.out.printf("Graph built: %d nodes, max level %d%n",
                graph.size(0), graph.getMaxLevel());

        // ── 5. Save the graph to disk ─────────────────────────────────────────
        // SQFeature embeds the ScalarQuantizer (dimMin/dimMax) in the index header so
        // it is recovered automatically when the index is loaded from disk.
        // InlineVectors stores the original float32 vectors for full-fidelity reranking.
        Path graphPath = Files.createTempFile("int8-siftsmall", ".jvector");
        try (GraphIndexWriter writer = GraphIndexWriter
                .getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, graph, graphPath)
                .with(new SQFeature(sq))
                .with(new InlineVectors(DIM))
                .build()) {
            writer.write(Map.of(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(floatRavv.getVector(nodeId))
            ));
        }
        System.out.printf("Graph written to %s (%.1f KB)%n",
                graphPath, Files.size(graphPath) / 1024.0);

        // ── 6. Load the graph from disk ───────────────────────────────────────
        // The ScalarQuantizer is recovered directly from the index header — no
        // sidecar file and no access to the original base vectors needed.
        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath);
        OnDiskGraphIndex diskGraph = OnDiskGraphIndex.load(readerSupplier);
        System.out.printf("Graph loaded: %d nodes, max level %d%n",
                diskGraph.size(0), diskGraph.getMaxLevel());

        ScalarQuantizer loadedSq = ((SQFeature) diskGraph.getFeatures().get(FeatureId.SQ_QUANTIZER))
                .getScalarQuantizer();
        System.out.printf("ScalarQuantizer loaded from index header: %s%n", loadedSq);

        // ── 7. Search with every siftsmall query vector ───────────────────────
        // Each float32 query is encoded on-the-fly with the quantizer recovered from disk.
        // The search uses byte×byte scoring for graph traversal, then reranks
        // the top candidates using the full-precision InlineVectors from disk.
        int topK     = 10;
        int efSearch = 100;
        System.out.printf("%nRunning %d queries (topK=%d, efSearch=%d):%n",
                queryVectors.size(), topK, efSearch);

        try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
            for (int q = 0; q < queryVectors.size(); q++) {
                var queryBytes = loadedSq.encode(queryVectors.get(q));
                var sf = (ScoreFunction.ExactScoreFunction)
                        node2 -> ByteVectorSimilarityFunction.DOT_PRODUCT.compare(queryBytes, byteRavv.getVector(node2));
                var ssp = new DefaultSearchScoreProvider(sf);
                var result = searcher.search(ssp, topK, efSearch, 0.0f, 0.0f, Bits.ALL);

                // Print top-1 result for each query; iterate result.getNodes() for the full list
                var top = result.getNodes()[0];
                System.out.printf("  query %3d → top-1 node %5d  score %.4f  (visited %d nodes)%n",
                        q, top.node, top.score, result.getVisitedCount());
            }
        }

        // ── 8. Cleanup ────────────────────────────────────────────────────────
        readerSupplier.close();
        Files.deleteIfExists(graphPath);
        System.out.println("Done.");
    }
}
