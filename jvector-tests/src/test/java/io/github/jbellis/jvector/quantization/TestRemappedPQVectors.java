/*
 * Copyright IBM Corp.
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
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestRemappedPQVectors extends RandomizedTest {

    private static final int DIM = 4;
    private static final int CLUSTERS = 4;

    /** Build a PQVectors over {@code n} random 4-D vectors using a small cluster count. */
    private static PQVectors buildSource(int n) {
        List<VectorFloat<?>> vectors = createRandomVectors(n, DIM);
        ListRandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, DIM);
        ProductQuantization pq = ProductQuantization.compute(ravv, 2, CLUSTERS, false);
        return (PQVectors) pq.encodeAll(ravv);
    }

    private VectorFloat<?> randomQuery(int dim) {
        return io.github.jbellis.jvector.TestUtil.randomVector(getRandom(), dim);
    }

    // -----------------------------------------------------------------------
    // count / get
    // -----------------------------------------------------------------------

    @Test
    public void testCountAndGetWithIdentityMapper() {
        PQVectors source = buildSource(10);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);

        assertEquals(source.count(), remapped.count());
        for (int i = 0; i < source.count(); i++) {
            assertEquals("bytes differ at ordinal " + i, source.get(i), remapped.get(i));
        }
    }

    @Test
    public void testCountAndGetWithMapMapper() {
        // source has 5 vectors; we expose only 3 of them in a different order
        PQVectors source = buildSource(5);

        // new ordinal 0 -> old 4, 1 -> old 2, 2 -> old 0
        Map<Integer, Integer> oldToNew = new HashMap<>();
        oldToNew.put(4, 0);
        oldToNew.put(2, 1);
        oldToNew.put(0, 2);
        OrdinalMapper mapper = new OrdinalMapper.MapMapper(oldToNew);
        RemappedPQVectors remapped = source.remap(mapper);

        assertEquals(3, remapped.count());
        assertEquals(source.get(4), remapped.get(0));
        assertEquals(source.get(2), remapped.get(1));
        assertEquals(source.get(0), remapped.get(2));
    }

    @Test
    public void testGetOutOfBoundsThrows() {
        PQVectors source = buildSource(5);
        RemappedPQVectors remapped = source.remap(new OrdinalMapper.IdentityMapper(4));

        assertThrows(IndexOutOfBoundsException.class, () -> remapped.get(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> remapped.get(5));
    }

    // -----------------------------------------------------------------------
    // remap() factory on PQVectors
    // -----------------------------------------------------------------------

    @Test
    public void testRemapFactoryMethod() {
        PQVectors source = buildSource(8);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);

        // same content as source when identity-mapped
        assertEquals(source, remapped);
        assertEquals(source.hashCode(), remapped.hashCode());
    }

    // -----------------------------------------------------------------------
    // equals / hashCode
    // -----------------------------------------------------------------------

    @Test
    public void testEqualsAgainstSource() {
        PQVectors source = buildSource(6);
        RemappedPQVectors remapped = source.remap(new OrdinalMapper.IdentityMapper(source.count() - 1));

        assertEquals(remapped, source);
        assertEquals(source, remapped);
    }

    @Test
    public void testEqualsRemappedSubset() {
        // Two remapped views over the same source with the same mapping must be equal to each other.
        PQVectors source = buildSource(6);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors r1 = source.remap(mapper);
        RemappedPQVectors r2 = source.remap(mapper);

        assertEquals(r1, r2);
        assertEquals(r1.hashCode(), r2.hashCode());
    }

    // -----------------------------------------------------------------------
    // score functions
    // -----------------------------------------------------------------------

    @Test
    public void testScoreFunctionMatchesSource() {
        PQVectors source = buildSource(10);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);
        VectorFloat<?> query = randomQuery(4);

        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            ScoreFunction.ApproximateScoreFunction srcFn = source.scoreFunctionFor(query, vsf);
            ScoreFunction.ApproximateScoreFunction remFn = remapped.scoreFunctionFor(query, vsf);
            for (int i = 0; i < source.count(); i++) {
                assertEquals("scoreFunctionFor " + vsf + " ordinal " + i,
                        srcFn.similarityTo(i), remFn.similarityTo(i), 1e-6f);
            }
        }
    }

    @Test
    public void testPrecomputedScoreFunctionMatchesSource() {
        PQVectors source = buildSource(10);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);
        VectorFloat<?> query = randomQuery(4);

        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            ScoreFunction.ApproximateScoreFunction srcFn = source.precomputedScoreFunctionFor(query, vsf);
            ScoreFunction.ApproximateScoreFunction remFn = remapped.precomputedScoreFunctionFor(query, vsf);
            for (int i = 0; i < source.count(); i++) {
                assertEquals("precomputedScoreFunctionFor " + vsf + " ordinal " + i,
                        srcFn.similarityTo(i), remFn.similarityTo(i), 1e-6f);
            }
        }
    }

    @Test
    public void testDiversityFunctionMatchesSource() {
        PQVectors source = buildSource(10);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);

        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            for (int node1 = 0; node1 < source.count(); node1++) {
                ScoreFunction.ApproximateScoreFunction srcFn = source.diversityFunctionFor(node1, vsf);
                ScoreFunction.ApproximateScoreFunction remFn = remapped.diversityFunctionFor(node1, vsf);
                for (int node2 = 0; node2 < source.count(); node2++) {
                    assertEquals("diversityFunctionFor " + vsf + " nodes " + node1 + "," + node2,
                            srcFn.similarityTo(node2), remFn.similarityTo(node2), 1e-6f);
                }
            }
        }
    }

    @Test
    public void testScoreFunctionReordersCorrectly() {
        // Remap so that new ordinal i sees old ordinal (n-1-i), i.e. reverse order.
        PQVectors source = buildSource(6);
        int n = source.count();
        Map<Integer, Integer> oldToNew = new HashMap<>();
        for (int old = 0; old < n; old++) {
            oldToNew.put(old, n - 1 - old);
        }
        OrdinalMapper mapper = new OrdinalMapper.MapMapper(oldToNew);
        RemappedPQVectors remapped = source.remap(mapper);
        VectorFloat<?> query = randomQuery(4);

        ScoreFunction.ApproximateScoreFunction srcFn = source.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
        ScoreFunction.ApproximateScoreFunction remFn = remapped.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
        for (int newOrd = 0; newOrd < n; newOrd++) {
            int oldOrd = n - 1 - newOrd;
            assertEquals("reversed mapping at new=" + newOrd,
                    srcFn.similarityTo(oldOrd), remFn.similarityTo(newOrd), 1e-6f);
        }
    }

    // -----------------------------------------------------------------------
    // write / load round-trip
    // -----------------------------------------------------------------------

    @Test
    public void testWriteAndLoad() throws Exception {
        PQVectors source = buildSource(20);
        OrdinalMapper mapper = new OrdinalMapper.IdentityMapper(source.count() - 1);
        RemappedPQVectors remapped = source.remap(mapper);

        File tmp = File.createTempFile("remapped-pq", ".cv");
        try {
            try (SimpleWriter writer = new SimpleWriter(tmp.toPath())) {
                remapped.write(writer);
            }
            try (SimpleMappedReader.Supplier rs = new SimpleMappedReader.Supplier(tmp.toPath())) {
                PQVectors loaded = PQVectors.load(rs.get(), 0);
                assertEquals(remapped, loaded);
            }
        } finally {
            tmp.delete();
        }
    }

    @Test
    public void testWriteAndLoadReorderedSubset() throws Exception {
        PQVectors source = buildSource(10);
        // Expose only even-indexed source vectors, in reversed order
        Map<Integer, Integer> oldToNew = new HashMap<>();
        int newOrd = 0;
        for (int old = 8; old >= 0; old -= 2) {
            oldToNew.put(old, newOrd++);
        }
        OrdinalMapper mapper = new OrdinalMapper.MapMapper(oldToNew);
        RemappedPQVectors remapped = source.remap(mapper);

        File tmp = File.createTempFile("remapped-pq-subset", ".cv");
        try {
            try (SimpleWriter writer = new SimpleWriter(tmp.toPath())) {
                remapped.write(writer);
            }
            try (SimpleMappedReader.Supplier rs = new SimpleMappedReader.Supplier(tmp.toPath())) {
                PQVectors loaded = PQVectors.load(rs.get(), 0);
                assertEquals(remapped.count(), loaded.count());
                assertEquals(remapped, loaded);
            }
        } finally {
            tmp.delete();
        }
    }
}
