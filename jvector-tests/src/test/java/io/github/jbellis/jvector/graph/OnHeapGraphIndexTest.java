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
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.logging.log4j.Logger;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class OnHeapGraphIndexTest extends RandomizedTest  {
    private final static Logger log = org.apache.logging.log4j.LogManager.getLogger(OnHeapGraphIndexTest.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int NUM_BASE_VECTORS = 100;
    private static final int NUM_NEW_VECTORS = 100;
    private static final int NUM_ALL_VECTORS = NUM_BASE_VECTORS + NUM_NEW_VECTORS;
    private static final int DIMENSION = 16;
    private static final int M = 8;
    private static final int BEAM_WIDTH = 100;
    private static final float ALPHA = 1.2f;
    private static final float NEIGHBOR_OVERFLOW = 1.2f;
    private static final boolean ADD_HIERARCHY = false;
    private static final int TOP_K = 10;

    private Path testDirectory;

    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> newVectors;
    private ArrayList<VectorFloat<?>> allVectors;
    private RandomAccessVectorValues baseVectorsRavv;
    private RandomAccessVectorValues newVectorsRavv;
    private RandomAccessVectorValues allVectorsRavv;
    private VectorFloat<?> queryVector;
    private int[] groundTruthAllVectors;
    private BuildScoreProvider baseBuildScoreProvider;
    private BuildScoreProvider allBuildScoreProvider;
    private ImmutableGraphIndex baseGraphIndex;
    private ImmutableGraphIndex allGraphIndex;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        baseVectors = new ArrayList<>(NUM_BASE_VECTORS);
        newVectors = new ArrayList<>(NUM_NEW_VECTORS);
        allVectors = new ArrayList<>(NUM_ALL_VECTORS);
        for (int i = 0; i < NUM_BASE_VECTORS; i++) {
            VectorFloat<?> vector = createRandomVector(DIMENSION);
            baseVectors.add(vector);
            allVectors.add(vector);
        }
        for (int i = 0; i < NUM_NEW_VECTORS; i++) {
            VectorFloat<?> vector = createRandomVector(DIMENSION);
            newVectors.add(vector);
            allVectors.add(vector);
        }

        // wrap the raw vectors in a RandomAccessVectorValues
        baseVectorsRavv = new ListRandomAccessVectorValues(baseVectors, DIMENSION);
        newVectorsRavv = new ListRandomAccessVectorValues(newVectors, DIMENSION);
        allVectorsRavv = new ListRandomAccessVectorValues(allVectors, DIMENSION);

        queryVector = createRandomVector(DIMENSION);
        groundTruthAllVectors = getGroundTruth(allVectorsRavv, queryVector, TOP_K, VectorSimilarityFunction.EUCLIDEAN);

        // score provider using the raw, in-memory vectors
        baseBuildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(baseVectorsRavv, VectorSimilarityFunction.EUCLIDEAN);
        allBuildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(allVectorsRavv, VectorSimilarityFunction.EUCLIDEAN);
        var baseGraphIndexBuilder = new GraphIndexBuilder(baseBuildScoreProvider,
                baseVectorsRavv.dimension(),
                M, // graph degree
                BEAM_WIDTH, // construction search depth
                NEIGHBOR_OVERFLOW, // allow degree overflow during construction by this factor
                ALPHA, // relax neighbor diversity requirement by this factor
                ADD_HIERARCHY); // add the hierarchy
        var allGraphIndexBuilder = new GraphIndexBuilder(allBuildScoreProvider,
                allVectorsRavv.dimension(),
                M, // graph degree
                BEAM_WIDTH, // construction search depth
                NEIGHBOR_OVERFLOW, // allow degree overflow during construction by this factor
                ALPHA, // relax neighbor diversity requirement by this factor
                ADD_HIERARCHY); // add the hierarchy

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
        var heapGraphOutputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + baseGraphIndex.getClass().getSimpleName() + "_onHeap");

        log.info("Writing graph to {}", graphOutputPath);
        TestUtil.writeGraph(baseGraphIndex, baseVectorsRavv, graphOutputPath);

        log.info("Writing on-heap graph to {}", heapGraphOutputPath);
        try (SimpleWriter writer = new SimpleWriter(heapGraphOutputPath.toAbsolutePath())) {
            ((OnHeapGraphIndex) baseGraphIndex).save(writer);
        }

        log.info("Reading on-heap graph from {}", heapGraphOutputPath);
        MutableGraphIndex reconstructedOnHeapGraphIndex;
        try (var readerSupplier = new SimpleMappedReader.Supplier(heapGraphOutputPath.toAbsolutePath())) {
            reconstructedOnHeapGraphIndex = OnHeapGraphIndex.load(readerSupplier.get(), NEIGHBOR_OVERFLOW, new VamanaDiversityProvider(baseBuildScoreProvider, ALPHA));
        }

        try (var readerSupplier = new SimpleMappedReader.Supplier(graphOutputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier)) {
            TestUtil.assertGraphEquals(baseGraphIndex, onDiskGraph);
            try (var onDiskView = onDiskGraph.getView()) {
                validateVectors(onDiskView, baseVectorsRavv);
            }

            TestUtil.assertGraphEquals(baseGraphIndex, reconstructedOnHeapGraphIndex);
            TestUtil.assertGraphEquals(onDiskGraph, reconstructedOnHeapGraphIndex);
        }
    }

    /**
     * Create {@link OnDiskGraphIndex} then append to it via {@link GraphIndexBuilder#buildAndMergeNewNodes}
     * Verify that the resulting OnHeapGraphIndex is equivalent to the graph that would have been alternatively generated by bulk index into a new {@link OnDiskGraphIndex}
     */
    @Test
    public void testIncrementalInsertionFromOnDiskIndex() throws IOException {
        var outputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + baseGraphIndex.getClass().getSimpleName());
        var heapGraphOutputPath = testDirectory.resolve("testReconstructionOfOnHeapGraphIndex_" + baseGraphIndex.getClass().getSimpleName() + "_onHeap");

        log.info("Writing graph to {}", outputPath);
        TestUtil.writeGraph(baseGraphIndex, baseVectorsRavv, outputPath);

        log.info("Writing on-heap graph to {}", heapGraphOutputPath);
        try (SimpleWriter writer = new SimpleWriter(heapGraphOutputPath.toAbsolutePath())) {
            ((OnHeapGraphIndex) baseGraphIndex).save(writer);
        }

        log.info("Reading on-heap graph from {}", heapGraphOutputPath);
        try (var readerSupplier = new SimpleMappedReader.Supplier(heapGraphOutputPath.toAbsolutePath())) {
            // We will create a trivial 1:1 mapping between the new graph and the ravv
            final int[] graphToRavvOrdMap = IntStream.range(0, allVectorsRavv.size()).toArray();
            ImmutableGraphIndex reconstructedAllNodeOnHeapGraphIndex = GraphIndexBuilder.buildAndMergeNewNodes(readerSupplier.get(), allVectorsRavv, allBuildScoreProvider, NUM_BASE_VECTORS, graphToRavvOrdMap, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY);

            // Verify that the recall is similar
            float recallFromReconstructedAllNodeOnHeapGraphIndex = calculateRecall(reconstructedAllNodeOnHeapGraphIndex, allBuildScoreProvider, queryVector, groundTruthAllVectors, TOP_K);
            float recallFromAllGraphIndex = calculateRecall(allGraphIndex, allBuildScoreProvider, queryVector, groundTruthAllVectors, TOP_K);
            Assert.assertEquals(recallFromReconstructedAllNodeOnHeapGraphIndex, recallFromAllGraphIndex, 0.01f);
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

    /**
     * Get the ground truth for a query vector
     * @param ravv the vectors to search
     * @param queryVector the query vector
     * @param topK the number of results to return
     * @param similarityFunction the similarity function to use

     * @return the ground truth
     */
    private static int[] getGroundTruth(RandomAccessVectorValues ravv, VectorFloat<?> queryVector, int topK, VectorSimilarityFunction similarityFunction) {
        var exactResults = new ArrayList<SearchResult.NodeScore>();
        for (int i = 0; i < ravv.size(); i++) {
            float similarityScore = similarityFunction.compare(queryVector, ravv.getVector(i));
            exactResults.add(new SearchResult.NodeScore(i, similarityScore));
        }
        exactResults.sort((a, b) -> Float.compare(b.score, a.score));
        return exactResults.stream().limit(topK).mapToInt(nodeScore -> nodeScore.node).toArray();
    }

    private static float calculateRecall(ImmutableGraphIndex graphIndex, BuildScoreProvider buildScoreProvider, VectorFloat<?> queryVector, int[] groundTruth, int k) throws IOException {
        try (GraphSearcher graphSearcher = new GraphSearcher(graphIndex)){
            SearchScoreProvider ssp = buildScoreProvider.searchProviderFor(queryVector);
            var searchResults = graphSearcher.search(ssp, k, Bits.ALL);
            var predicted = Arrays.stream(searchResults.getNodes()).mapToInt(nodeScore -> nodeScore.node).boxed().collect(Collectors.toSet());
            return calculateRecall(predicted, groundTruth, k);
        }
    }
    /**
     * Calculate the recall for a set of predicted results
     * @param predicted the predicted results
     * @param groundTruth the ground truth
     * @param k the number of results to consider
     * @return the recall
     */
    private static float calculateRecall(Set<Integer> predicted, int[] groundTruth, int k) {
        int hits = 0;
        int actualK = Math.min(k, Math.min(predicted.size(), groundTruth.length));

        for (int i = 0; i < actualK; i++) {
            for (int j = 0; j < actualK; j++) {
                if (predicted.contains(groundTruth[j])) {
                    hits++;
                    break;
                }
            }
        }

        return ((float) hits) / (float) actualK;
    }
}
