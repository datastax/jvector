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
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Int8 HNSW/Vamana index build-and-search tutorial using real siftsmall int8 vectors.
 *
 * Reads pre-quantised int8 base and query vectors from the siftsmall directory,
 * builds an on-disk graph index, then runs every query vector against it.
 *
 * Run via TutorialRunner:
 *   ./mvnw -pl jvector-examples -am -Pjdk22 compile exec:exec@tutorial -Dtutorial=int8build
 */
public class Int8IndexBuild {

    /** Siftsmall vectors are 128-dimensional. */
    private static final int DIM = 128;

    public static void main(String[] args) throws IOException {
        // ── 1. Vector type support ─────────────────────────────────────────────
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        System.out.println("VectorizationProvider: " + VectorizationProvider.getInstance().getClass().getSimpleName());

        // ── 2. Load int8 vectors from siftsmall directory ─────────────────────
        // The .bvecs files are raw flat binary: numVectors * DIM bytes, no header.
        String siftPath = "siftsmall";
        List<ByteSequence<?>> baseVectors  = SiftLoader.readBvecs(siftPath + "/siftsmall_base.bvecs",  DIM);
        List<ByteSequence<?>> queryVectors = SiftLoader.readBvecs(siftPath + "/siftsmall_query.bvecs", DIM);
        System.out.printf("Loaded %d base vectors and %d query vectors, dim=%d%n",
                baseVectors.size(), queryVectors.size(), DIM);

        // ── 3. Wrap base vectors in a RandomAccessByteVectorValues (RAVV) ─────
        RandomAccessByteVectorValues ravv = new ListRandomAccessByteVectorValues(baseVectors, DIM);

        // ── 4. Build the graph incrementally (int8, DOT_PRODUCT, M=16, efConstruction=100) ─
        ImmutableGraphIndex graph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
                ravv, ByteVectorSimilarityFunction.DOT_PRODUCT, 16, 100, 1.2f, 1.2f, true)) {

            for (int i = 0; i < baseVectors.size(); i++) {
                builder.addGraphNode(i, baseVectors.get(i));
            }
            System.out.printf("Inserted %d nodes%n", builder.getGraph().size(0));

            builder.cleanup();
            graph = builder.getGraph();
        }

        // ── 5. Inspect the result ──────────────────────────────────────────────
        try (var view = graph.getView()) {
            System.out.printf("Graph built: %d nodes, max level %d, entry node %s%n",
                    graph.size(0),
                    graph.getMaxLevel(),
                    view.entryNode());
        }
        System.out.printf("RAM used: %.1f KB%n", graph.ramBytesUsed() / 1024.0);

        // ── 6. Save to disk ───────────────────────────────────────────────────
        // InlineVectors stores float32 versions of the int8 components for on-disk reranking.
        Path graphPath = Files.createTempFile("int8-siftsmall", ".jvector");
        try (GraphIndexWriter writer = GraphIndexWriter
                .getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, graph, graphPath)
                .with(new InlineVectors(DIM))
                .build()) {
            writer.write(Map.of(
                FeatureId.INLINE_VECTORS,
                nodeId -> {
                    var bs = ravv.getVector(nodeId);
                    var fv = vts.createFloatVector(DIM);
                    for (int i = 0; i < DIM; i++) fv.set(i, bs.get(i));
                    return new InlineVectors.State(fv);
                }
            ));
        }
        System.out.printf("Graph written to %s (%.1f KB)%n",
                graphPath, Files.size(graphPath) / 1024.0);

        // ── 7. Load from disk ─────────────────────────────────────────────────
        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath);
        OnDiskGraphIndex diskGraph = OnDiskGraphIndex.load(readerSupplier);
        System.out.printf("Graph loaded: %d nodes, max level %d%n",
                diskGraph.size(0), diskGraph.getMaxLevel());

        // ── 8. Search with every query vector ─────────────────────────────────
        int topK     = 10;
        int efSearch = 100;
        System.out.printf("%nRunning %d queries (topK=%d, efSearch=%d):%n",
                queryVectors.size(), topK, efSearch);

        try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
            for (int q = 0; q < queryVectors.size(); q++) {
                ByteSequence<?> query = queryVectors.get(q);
                var sf = (ScoreFunction.ExactScoreFunction)
                        node2 -> ByteVectorSimilarityFunction.DOT_PRODUCT.compare(query, ravv.getVector(node2));
                var ssp = new DefaultSearchScoreProvider(sf);
                var result = searcher.search(ssp, topK, efSearch, 0.0f, 0.0f, Bits.ALL);

                // Print top-1 result for each query; change to result.getNodes() for full list
                var top = result.getNodes()[0];
                System.out.printf("  query %3d → top-1 node %5d  score %.4f  (visited %d nodes)%n",
                        q, top.node, top.score, result.getVisitedCount());
            }
        }

        // ── 9. Cleanup ────────────────────────────────────────────────────────
        readerSupplier.close();
        Files.deleteIfExists(graphPath);
        System.out.println("Done.");
    }
}
