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

package io.github.jbellis.jvector.quantization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestADCGraphIndex extends RandomizedTest {

    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testFusedGraph() throws Exception {
        // generate random graph, M=32, 256-dimension vectors
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var vectors = createRandomVectors(1000,  512);
        var ravv = new ListRandomAccessVectorValues(vectors, 512);
        var pq = ProductQuantization.compute(ravv, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(ravv);

        TestUtil.writeFusedGraph(graph, ravv, pqv, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier, 0))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var cachedOnDiskView = onDiskGraph.getView())
            {
                for (var similarityFunction : VectorSimilarityFunction.values()) {
                    var queryVector = TestUtil.randomVector(getRandom(), 512);
                    var pqScoreFunction = pqv.precomputedScoreFunctionFor(queryVector, similarityFunction);
                    var reranker = cachedOnDiskView.rerankerFor(queryVector, similarityFunction);
                    for (int i = 0; i < 50; i++) {
                        var fusedScoreFunction = cachedOnDiskView.approximateScoreFunctionFor(queryVector, similarityFunction);
                        var ordinal = getRandom().nextInt(graph.size());
                        // first pass compares fused ADC's direct similarity to reranker's similarity, used for comparisons to a specific node
                        var neighbors = cachedOnDiskView.getNeighborsIterator(0, ordinal); // TODO
                        for (; neighbors.hasNext(); ) {
                            var neighbor = neighbors.next();
                            var similarity = fusedScoreFunction.similarityTo(neighbor);
                            assertEquals(reranker.similarityTo(neighbor), similarity, 0.01);
                        }
                        // second pass compares fused ADC's edge similarity prior to having enough information for quantization to PQ
                        neighbors = cachedOnDiskView.getNeighborsIterator(0, ordinal); // TODO
                        var edgeSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal);
                        for (int j = 0; neighbors.hasNext(); j++) {
                            var neighbor = neighbors.next();
                            assertEquals(pqScoreFunction.similarityTo(neighbor), edgeSimilarities.get(j), 0.01);
                        }
                        // third pass compares fused ADC's edge similarity after quantization to edge similarity before quantization
                        var fusedEdgeSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal);
                        for (int j = 0; j < fusedEdgeSimilarities.length(); j++) {
                            assertEquals(fusedEdgeSimilarities.get(j), edgeSimilarities.get(j), 0.01);
                        }
                    }
                }
            }
        }
    }
}
