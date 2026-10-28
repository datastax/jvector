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

import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestPQRecallEvaluator {
    @Test
    public void resolvesExactStoredByteBudgetsAndPreservesLegacyFactor() {
        String exact = System.getProperty("jvector.pq.subspaces");
        String factor = System.getProperty("jvector.pq.mFactor");
        try {
            System.clearProperty("jvector.pq.subspaces");
            System.clearProperty("jvector.pq.mFactor");
            assertEquals(96, DistancesPQ.subspaceCount(768));
            System.setProperty("jvector.pq.mFactor", "4");
            assertEquals(192, DistancesPQ.subspaceCount(768));
            assertThrows(IllegalArgumentException.class, () -> DistancesPQ.subspaceCount(767));
            System.clearProperty("jvector.pq.mFactor");
            for (int dimension : new int[] {768, 1024, 1536, 3072}) {
                for (int bits : new int[] {2, 4}) {
                    int bytes = 5 + (dimension / 2) * bits / 8;
                    System.setProperty("jvector.pq.subspaces", Integer.toString(bytes));
                    assertEquals(bytes, DistancesPQ.subspaceCount(dimension));
                }
            }
            for (String bad : new String[] {"0", "-1", "769", "bad"}) {
                System.setProperty("jvector.pq.subspaces", bad);
                assertThrows(IllegalArgumentException.class, () -> DistancesPQ.subspaceCount(768));
            }
            System.setProperty("jvector.pq.subspaces", "101");
            System.setProperty("jvector.pq.mFactor", "8");
            assertThrows(IllegalArgumentException.class, () -> DistancesPQ.subspaceCount(768));
        } finally {
            restore("jvector.pq.subspaces", exact);
            restore("jvector.pq.mFactor", factor);
        }
    }

    @Test
    public void scansFinalOrdinalAndResolvesBoundaryTiesById() {
        float[] scores = {.1f, .8f, .9f, .8f, .8f, .2f, 1f};
        ScoreFunction.ApproximateScoreFunction scorer = i -> scores[i];
        var result = PQRecallEvaluator.rank(scorer, scores.length, List.of(6, 3, 1), 3);
        assertArrayEquals(new int[] {6, 2, 1}, result.ids);
        assertArrayEquals(new float[] {1f, .9f, .8f}, result.scores, 0f);
        assertEquals(2, result.hits);
        scores[6] = Float.NaN;
        assertThrows(IllegalStateException.class, () -> PQRecallEvaluator.rank(scorer, scores.length, List.of(1), 1));
    }

    @Test
    public void unevenPqRoundTripPreservesScoresAndPerQueryRankings() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> base = new ArrayList<>();
        var random = new Random(1307);
        for (int i = 0; i < 96; i++) {
            float[] values = new float[10];
            for (int d = 0; d < values.length; d++)
                values[d] = random.nextFloat() - .5f;
            var vector = types.createFloatVector(values);
            VectorUtil.l2normalize(vector);
            base.add(vector);
        }
        var executor = new ForkJoinPool(2);
        var folder = Files.createTempDirectory("pq-full-recall-");
        var artifact = folder.resolve("vectors.pqv");
        var rankings = folder.resolve("rankings.tsv");
        String oldArtifact = System.getProperty("jvector.bench.pq.artifact");
        String oldOutput = System.getProperty("jvector.bench.recall.output");
        try {
            var ravv = new ListRandomAccessVectorValues(base, 10);
            var pq = ProductQuantization.compute(ravv, 3, 16, false, -1f, executor, executor);
            var encoded = pq.encodeAll(ravv, executor);
            assertEquals(3, pq.compressedVectorSize());
            var queries = base.subList(0, 3);
            PQRecallEvaluator.validate(encoded, queries);
            List<List<Integer>> truth = new ArrayList<>();
            for (var query : queries) {
                var scorer = encoded.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                Integer[] all = new Integer[base.size()];
                Arrays.setAll(all, i -> i);
                Arrays.sort(all, (a, b) -> {
                    int c = Float.compare(scorer.similarityTo(b), scorer.similarityTo(a));
                    return c != 0 ? c : Integer.compare(a, b);
                });
                var top = Arrays.asList(Arrays.copyOf(all, 10));
                truth.add(top);
                var result = PQRecallEvaluator.rank(scorer, base.size(), top, 10);
                assertArrayEquals(top.stream().mapToInt(Integer::intValue).toArray(), result.ids);
                assertEquals(10, result.hits);
            }
            System.setProperty("jvector.bench.pq.artifact", artifact.toString());
            System.setProperty("jvector.bench.recall.output", rankings.toString());
            PQRecallEvaluator.run(encoded, queries, truth, 10, executor);
            var rows = Files.readAllLines(rankings);
            assertEquals(4, rows.size());
            for (int q = 0; q < queries.size(); q++)
                assertTrue(rows.get(q + 1).startsWith(q + "\t10\t"));
            try (var supplier = new SimpleMappedReader.Supplier(artifact); var reader = supplier.get()) {
                var loaded = PQVectors.load(reader);
                assertEquals(encoded.count(), loaded.count());
                for (var query : queries) {
                    var original = encoded.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                    var restored = loaded.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                    for (int i = 0; i < base.size(); i++) {
                        assertEquals(original.similarityTo(i), restored.similarityTo(i), 0f);
                    }
                }
            }
        } finally {
            executor.shutdown();
            restore("jvector.bench.pq.artifact", oldArtifact);
            restore("jvector.bench.recall.output", oldOutput);
            Files.deleteIfExists(artifact);
            Files.deleteIfExists(rankings);
            Files.deleteIfExists(folder);
        }
    }

    private static void restore(String key, String value) {
        if (value == null)
            System.clearProperty(key);
        else
            System.setProperty(key, value);
    }
}
