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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.disk.NeighborsScoreCache;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.logging.log4j.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class OnHeapGraphIndexTest extends RandomizedTest  {
    private final static Logger log = org.apache.logging.log4j.LogManager.getLogger(OnHeapGraphIndexTest.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int numBaseVectors = 100;
    private static final int numNewVectors = 100;
    private static final int numAllVectors = numBaseVectors + numNewVectors;
    private static final int dimension = 16;
    private static final int M = 8;
    private static final int beamWidth = 100;
    private static final float alpha = 1.2f;
    private static final float neighborOverflow = 1.2f;
    private static final boolean addHierarchy = false;

    private Path testDirectory;

    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> newVectors;
    private ArrayList<VectorFloat<?>> allVectors;
    private RandomAccessVectorValues baseVectorsRavv;
    private RandomAccessVectorValues newVectorsRavv;
    private RandomAccessVectorValues allVectorsRavv;
    private BuildScoreProvider baseBuildScoreProvider;
    private BuildScoreProvider newBuildScoreProvider;
    private BuildScoreProvider allBuildScoreProvider;
    private OnHeapGraphIndex baseGraphIndex;
    private OnHeapGraphIndex newGraphIndex;
    private OnHeapGraphIndex allGraphIndex;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        baseVectors = new ArrayList<>(numBaseVectors);
        newVectors = new ArrayList<>(numNewVectors);
        allVectors = new ArrayList<>(numAllVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(dimension);
            baseVectors.add(vector);
            allVectors.add(vector);
        }
        for (int i = 0; i < numNewVectors; i++) {
            VectorFloat<?> vector = createRandomVector(dimension);
            newVectors.add(vector);
            allVectors.add(vector);
        }

        // wrap the raw vectors in a RandomAccessVectorValues
        baseVectorsRavv = new ListRandomAccessVectorValues(baseVectors, dimension);
        newVectorsRavv = new ListRandomAccessVectorValues(newVectors, dimension);
        allVectorsRavv = new ListRandomAccessVectorValues(allVectors, dimension);

        baseBuildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(baseVectorsRavv, VectorSimilarityFunction.EUCLIDEAN);
        newBuildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(newVectorsRavv, VectorSimilarityFunction.EUCLIDEAN);
        allBuildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(allVectorsRavv, VectorSimilarityFunction.EUCLIDEAN);
        var baseGraphIndexBuilder = new GraphIndexBuilder(baseBuildScoreProvider,
                baseVectorsRavv.dimension(),
                M, // graph degree
                beamWidth, // construction search depth
                neighborOverflow, // allow degree overflow during construction by this factor
                alpha, // relax neighbor diversity requirement by this factor
                addHierarchy); // add the hierarchy
        var allGraphIndexBuilder = new GraphIndexBuilder(allBuildScoreProvider,
                allVectorsRavv.dimension(),
                M, // graph degree
                beamWidth, // construction search depth
                neighborOverflow, // allow degree overflow during construction by this factor
                alpha, // relax neighbor diversity requirement by this factor
                addHierarchy); // add the hierarchy

        baseGraphIndex = baseGraphIndexBuilder.build(baseVectorsRavv);
        allGraphIndex = allGraphIndexBuilder.build(allVectorsRavv);
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }


    /**
     * Create an {@link OnHeapGraphIndex} persist it as a {@link OnDiskGraphIndex} and reconstruct back to a mutable {@link OnHeapGraphIndex}
     * Make sure that both graphs are equivalent
     * @throws IOException
     */
    @Test
    public void testReconstructionOfOnHeapGraphIndex() throws IOException {
        var graphOutputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + baseGraphIndex.getClass().getSimpleName());
        var neighborsScoreCacheOutputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + NeighborsScoreCache.class.getSimpleName());
        log.info("Writing graph to {}", graphOutputPath);
        TestUtil.writeGraph(baseGraphIndex, baseVectorsRavv, graphOutputPath);

        log.info("Writing neighbors score cache to {}", neighborsScoreCacheOutputPath);
        final NeighborsScoreCache neighborsScoreCache = new NeighborsScoreCache(baseGraphIndex);
        try (SimpleWriter writer = new SimpleWriter(neighborsScoreCacheOutputPath.toAbsolutePath())) {
            neighborsScoreCache.write(writer);
        }

        log.info("Reading neighbors score cache from {}", neighborsScoreCacheOutputPath);
        final NeighborsScoreCache neighborsScoreCacheRead;
        try (var readerSupplier = new SimpleMappedReader.Supplier(neighborsScoreCacheOutputPath.toAbsolutePath())) {
            neighborsScoreCacheRead = new NeighborsScoreCache(readerSupplier.get());
        }

        try (var readerSupplier = new SimpleMappedReader.Supplier(graphOutputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier)) {
            TestUtil.assertGraphEquals(baseGraphIndex, onDiskGraph);
            try (var onDiskView = onDiskGraph.getView()) {
                validateVectors(onDiskView, baseVectorsRavv);
            }

            OnHeapGraphIndex reconstructedOnHeapGraphIndex = OnHeapGraphIndex.convertToHeap(onDiskGraph, neighborsScoreCacheRead, baseBuildScoreProvider, neighborOverflow, alpha);
            TestUtil.assertGraphEquals(baseGraphIndex, reconstructedOnHeapGraphIndex);
            TestUtil.assertGraphEquals(onDiskGraph, reconstructedOnHeapGraphIndex);

        }
    }

    /**
     * Create {@link OnDiskGraphIndex} then append to it via {@link OnHeapGraphIndex#addNewNodes}
     * Verify that the resulting OnHeapGraphIndex is equivalent to the graph that would have been alternatively generated by bulk index into a new {@link OnDiskGraphIndex}
     */
    @Test
    public void testIncrementalInsertionFromOnDiskIndex() throws IOException {
        var outputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + baseGraphIndex.getClass().getSimpleName());
        log.info("Writing graph to {}", outputPath);
        final NeighborsScoreCache neighborsScoreCache = new NeighborsScoreCache(baseGraphIndex);
        TestUtil.writeGraph(baseGraphIndex, baseVectorsRavv, outputPath);
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier)) {
            TestUtil.assertGraphEquals(baseGraphIndex, onDiskGraph);
            OnHeapGraphIndex reconstructedAllNodeOnHeapGraphIndex = OnHeapGraphIndex.addNewNodes(onDiskGraph, neighborsScoreCache, newVectorsRavv, allBuildScoreProvider, numBaseVectors, beamWidth, neighborOverflow, alpha, addHierarchy);

            try (GraphSearcher reconstructedAllGraphSearcher = new GraphSearcher(reconstructedAllNodeOnHeapGraphIndex);
                 GraphSearcher allGraphSearcher = new GraphSearcher(allGraphIndex)) {
                final int topK = 10;
                VectorFloat<?> queryVector = createRandomVector(dimension);
                var resultFromReconstructed = reconstructedAllGraphSearcher.search(allBuildScoreProvider.searchProviderFor(queryVector), topK, Bits.ALL);
                var resultFromAll = allGraphSearcher.search(allBuildScoreProvider.searchProviderFor(queryVector), topK, Bits.ALL);
                log.info("Reconstructed result: {}, all result: {}", resultFromReconstructed, resultFromAll);
                assertEquals(resultFromReconstructed.getNodes().length, resultFromAll.getNodes().length);
                final Set<Integer> reconstructedResultSet = Arrays.stream(resultFromReconstructed.getNodes()).map(nodeScore -> nodeScore.node).collect(Collectors.toSet());
                final Set<Integer> allResultSet = Arrays.stream(resultFromAll.getNodes()).map(nodeScore -> nodeScore.node).collect(Collectors.toSet());
                reconstructedResultSet.retainAll(allResultSet);
                final float resultSetsOverlap = (1.0f * reconstructedResultSet.size()) / (1.0f * allResultSet.size());
                assertTrue(String.format("expected result set overlap is >= 0.9 but overlap is: %s", resultSetsOverlap),  resultSetsOverlap >= 0.90f);
            }
        }
    }

    public static void validateVectors(OnDiskGraphIndex.View view, RandomAccessVectorValues ravv) {
        for (int i = 0; i < view.size(); i++) {
            assertEquals("Incorrect vector at " + i, ravv.getVector(i), view.getVector(i));
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
