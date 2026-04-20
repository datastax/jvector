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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Tests the two-phase NVQ write pattern used in Grid.buildOnDisk:
 *   1. Parallel writeFeaturesInline (NVQ data)
 *   2. Sequential write(Map.of()) (graph structure)
 *
 * This exercises the lazy-writer initialization in GraphIndexPersister which was
 * unsynchronized and could cause a race condition in the original code.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestNVQTwoPhaseWrite extends LuceneTestCase {

    private Path testDirectory;
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    /**
     * Mimics Grid.buildOnDisk with NVQ_VECTORS:
     * - PQ used for graph construction scoring
     * - NVQ encoded inline via parallel writeFeaturesInline
     * - Graph structure written via write(Map.of())
     * - Search uses NVQ refinement at multiple overquery factors
     */
    @Test
    public void testParallelNVQTwoPhaseWriteAndSearch() throws IOException {
        int dimension = 32;
        int size = 200;
        int M = 8;
        int efConstruction = 50;
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;

        List<VectorFloat<?>> vectorList = new ArrayList<>(TestUtil.createRandomVectors(size, dimension));
        var ravv = new ListRandomAccessVectorValues(vectorList, dimension);

        // PQ for build scoring (like Grid.buildOnDisk)
        var pq = ProductQuantization.compute(ravv, dimension / 4, 64, false);
        PQVectors pqVectors = (PQVectors) pq.encodeAll(ravv);
        BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);

        // NVQ for inline refinement
        NVQuantization nvq = NVQuantization.compute(ravv, 2);

        GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dimension, M, efConstruction, neighborOverflow, alpha, true);

        Path graphPath = testDirectory.resolve("nvq_graph");
        // Use an IdentityMapper sized to the full dataset, matching Grid.buildOnDisk's pattern
        var identityMapper = new OrdinalMapper.IdentityMapper(size - 1);
        GraphIndex.WriteBuilder writer = builder.getGraph().writer(graphPath)
                .with(new NVQ(nvq))
                .withMapper(identityMapper);

        // Parallel phase 1: writeFeaturesInline (exercises the lazy-writer race condition)
        IntStream.range(0, size).parallel().forEach(node -> {
            try {
                var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                stateMap.put(FeatureId.NVQ_VECTORS, new NVQ.State(nvq.encode(ravv.getVector(node))));
                writer.writeFeaturesInline(node, stateMap);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            try {
                builder.addGraphNode(node, ravv.getVector(node));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        builder.cleanup();

        // Phase 2: write graph structure (no NVQ suppliers needed, already written inline)
        writer.write(Map.of());
        writer.close();
        builder.close();

        // Load and search with NVQ refinement
        // For NVQ_VECTORS (non-FUSED_PQ), approximate scoring comes from PQ, refining from the NVQ view.
        try (var readerSupplier = new SimpleMappedReader.Supplier(graphPath);
             var diskGraph = OnDiskGraphIndex.load(readerSupplier)) {

            int topK = 5;
            List<VectorFloat<?>> queries = TestUtil.createRandomVectors(10, dimension);

            for (float overqueryFactor : new float[]{1.0f, 2.0f, 5.0f}) {
                int refineK = (int) (topK * overqueryFactor);
                try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
                    var view = (OnDiskGraphIndex.View) searcher.getView();
                    for (VectorFloat<?> query : queries) {
                        var asf = pqVectors.precomputedScoreFunctionFor(query, VectorSimilarityFunction.EUCLIDEAN);
                        var refiner = view.refinerFor(query, VectorSimilarityFunction.EUCLIDEAN);
                        SearchScoreProvider ssp = new DefaultSearchScoreProvider(asf, refiner);
                        // Must not throw BitsPerDimension 0 or any other exception
                        SearchResult result = searcher.search(ssp, topK, refineK, 0.0f, 0.0f, Bits.ALL);
                        assertNotNull(result);
                        assertTrue("Expected at least 1 result", result.getNodes().length > 0);
                    }
                }
            }
        }
    }
}
