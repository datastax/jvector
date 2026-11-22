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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Graphs built outside {@link GraphIndexBuilder}'s incremental insertion, through
 * {@link GraphIndexBuilder#addGraphNodeWithNeighbors} and {@link GraphBuildAccelerator}.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestExternalGraphBuild extends LuceneTestCase {

    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    @After
    public void clearProperty() {
        System.clearProperty(GraphBuildAccelerators.PROPERTY);
    }

    /** The exact top-{@code maxDegree} neighbors of every node, as a test-only "accelerator". */
    static void buildExact(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int maxDegree = builder.getMaxDegree();
        IntStream.range(0, n).parallel().forEach(i -> {
            VectorFloat<?> v = vectors.getVector(i);
            Integer[] others = IntStream.range(0, n).filter(j -> j != i).boxed().toArray(Integer[]::new);
            float[] scores = new float[n];
            for (int j : others) {
                scores[j] = similarity.compare(v, vectors.getVector(j));
            }
            Arrays.sort(others, (a, b) -> Float.compare(scores[b], scores[a]));
            var neighbors = new NodeArray(maxDegree);
            for (int c = 0; c < maxDegree; c++) {
                neighbors.addInOrder(others[c], scores[others[c]]);
            }
            builder.addGraphNodeWithNeighbors(i, neighbors);
        });
        builder.completeExternalBuild(0);
    }

    /** The exact top-{@code maxDegree} neighbors of every node among the nodes of its layer, in every layer. */
    static void buildExactHierarchy(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        int n = vectors.size();
        int[] levels = builder.sampleGraphLevels(n);
        int maxLevel = Arrays.stream(levels).max().orElse(0);
        for (int level = 0; level <= maxLevel; level++) {
            final int lvl = level;
            int[] members = IntStream.range(0, n).filter(i -> levels[i] >= lvl).toArray();
            int degree = Math.min(builder.getMaxDegree(), members.length - 1);
            IntStream.range(0, members.length).parallel().forEach(x -> {
                int i = members[x];
                VectorFloat<?> v = vectors.getVector(i);
                Integer[] others = Arrays.stream(members).filter(j -> j != i).boxed().toArray(Integer[]::new);
                float[] scores = new float[n];
                for (int j : others) {
                    scores[j] = similarity.compare(v, vectors.getVector(j));
                }
                Arrays.sort(others, (a, b) -> Float.compare(scores[b], scores[a]));
                var neighbors = new NodeArray(Math.max(degree, 1));
                for (int c = 0; c < degree; c++) {
                    neighbors.addInOrder(others[c], scores[others[c]]);
                }
                builder.addGraphNodeWithNeighbors(lvl, i, neighbors);
            });
        }
        builder.completeExternalBuild(0);
    }

    private static double recallAt10(ImmutableGraphIndex graph, RandomAccessVectorValues vectors, int queries) {
        double hits = 0;
        for (int q = 0; q < queries; q++) {
            VectorFloat<?> query = vectors.getVector(q);
            float[] scores = new float[vectors.size()];
            Integer[] all = IntStream.range(0, vectors.size()).boxed().toArray(Integer[]::new);
            for (int j = 0; j < vectors.size(); j++) {
                scores[j] = SIMILARITY.compare(query, vectors.getVector(j));
            }
            Arrays.sort(all, (a, b) -> Float.compare(scores[b], scores[a]));
            Set<Integer> truth = new HashSet<>(Arrays.asList(all).subList(0, 10));
            var result = GraphSearcher.search(query, 10, vectors, SIMILARITY, graph, Bits.ALL);
            for (var ns : result.getNodes()) {
                if (truth.contains(ns.node)) {
                    hits++;
                }
            }
        }
        return hits / (10.0 * queries);
    }

    @Test
    public void testExternalBuildIsSearchable() {
        List<VectorFloat<?>> data = List.of(createRandomFloatVectors(2000, 32, getRandom()));
        var vectors = new ListRandomAccessVectorValues(data, 32);
        var builder = new GraphIndexBuilder(vectors, SIMILARITY, 16, 100, 1.2f, 1.2f, false);
        buildExact(builder, vectors, SIMILARITY);
        builder.cleanup();

        ImmutableGraphIndex graph = builder.getGraph();
        assertEquals(2000, graph.size(0));
        try (var view = graph.getView()) {
            for (int i = 0; i < 2000; i++) {
                assertTrue(view.getNeighborsIterator(0, i).size() <= builder.getMaxDegree());
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        assertTrue(recallAt10(graph, vectors, 100) > 0.9);
    }

    @Test
    public void testExternalBuildWithExternalHierarchy() {
        List<VectorFloat<?>> data = List.of(createRandomFloatVectors(4000, 32, getRandom()));
        var vectors = new ListRandomAccessVectorValues(data, 32);
        var builder = new GraphIndexBuilder(vectors, SIMILARITY, 16, 100, 1.2f, 1.2f, true);
        buildExactHierarchy(builder, vectors, SIMILARITY);
        builder.cleanup();

        ImmutableGraphIndex graph = builder.getGraph();
        assertEquals(4000, graph.size(0));
        assertTrue(graph.getMaxLevel() > 0);
        try (var view = graph.getView()) {
            assertEquals(graph.getMaxLevel(), view.entryNode().level);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        assertTrue(recallAt10(graph, vectors, 100) > 0.9);
    }

    @Test
    public void testAcceleratorsAreOptIn() {
        assertTrue(GraphBuildAccelerators.find().isEmpty());
        System.setProperty(GraphBuildAccelerators.PROPERTY, "none");
        assertTrue(GraphBuildAccelerators.find().isEmpty());
        System.setProperty(GraphBuildAccelerators.PROPERTY, "does-not-exist");
        assertTrue(GraphBuildAccelerators.find().isEmpty());
    }

    @Test
    public void testAcceleratorIsDiscoveredAndBuilds() {
        System.setProperty(GraphBuildAccelerators.PROPERTY, TestGraphBuildAccelerator.NAME);
        var accelerator = GraphBuildAccelerators.find();
        assertTrue(accelerator.isPresent());

        List<VectorFloat<?>> data = List.of(createRandomFloatVectors(500, 16, getRandom()));
        var vectors = new ListRandomAccessVectorValues(data, 16);
        var builder = new GraphIndexBuilder(vectors, SIMILARITY, 8, 100, 1.2f, 1.2f, false);
        assertTrue(accelerator.get().supports(builder, vectors, SIMILARITY));
        accelerator.get().build(builder, vectors, SIMILARITY);
        builder.cleanup();
        assertEquals(500, builder.getGraph().size(0));
    }

    /** Registered in META-INF/services for {@link #testAcceleratorIsDiscoveredAndBuilds}. */
    public static final class TestGraphBuildAccelerator implements GraphBuildAccelerator {
        static final String NAME = "test-exact";

        @Override
        public String name() {
            return NAME;
        }

        @Override
        public boolean supports(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
            return true;
        }

        @Override
        public void build(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
            buildExact(builder, vectors, similarity);
        }
    }
}
