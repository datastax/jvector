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
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2IntHashMap;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static io.github.jbellis.jvector.TestUtil.randomVector;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestRemappedPQVectors extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void testRemappedGetAndWrite() throws IOException {
        Random random = getRandom();
        int dimension = 16;
        int subspaceCount = 4;
        int numOriginalVectors = 200;

        List<VectorFloat<?>> originalVectorList = new ArrayList<>(numOriginalVectors);
        for (int i = 0; i < numOriginalVectors; i++) {
            originalVectorList.add(randomVector(random, dimension));
        }

        var originalValues = new ListRandomAccessVectorValues(originalVectorList, dimension);
        var pq = ProductQuantization.compute(originalValues, subspaceCount, 256, false);
        var mutablePQ = new MutablePQVectors(pq);
        for (int i = 0; i < numOriginalVectors; i++) {
            mutablePQ.encodeAndSet(i, originalVectorList.get(i));
        }

        // Map: new ordinal -> old ordinal
        // new 0 -> old 10
        // new 1 -> OMITTED (hole)
        // new 2 -> old 150
        // new 3 -> old 5
        Int2IntHashMap oldToNew = new Int2IntHashMap(Integer.MIN_VALUE);
        oldToNew.put(10, 0);
        oldToNew.put(150, 2);
        oldToNew.put(5, 3);
        OrdinalMapper mapper = new OrdinalMapper.MapMapper(oldToNew);

        int outputCount = 4;
        PQVectors remapped = mutablePQ.remap(outputCount, mapper);

        assertEquals(outputCount, remapped.count());

        // Verify .get() semantics
        assertByteSequenceEquals(mutablePQ.get(10), remapped.get(0));
        assertZeroSequence(remapped.get(1), subspaceCount);
        assertByteSequenceEquals(mutablePQ.get(150), remapped.get(2));
        assertByteSequenceEquals(mutablePQ.get(5), remapped.get(3));

        // Write and reload
        File tempFile = Files.createTempFile("remapped_pq", ".tmp").toFile();
        tempFile.deleteOnExit();

        try (var writer = new SimpleWriter(tempFile.getAbsolutePath())) {
            remapped.write(writer, 4);
        }

        try (var reader = new SimpleMappedReader(tempFile.getAbsolutePath())) {
            PQVectors loaded = PQVectors.load(reader);
            assertEquals(outputCount, loaded.count());
            assertByteSequenceEquals(mutablePQ.get(10), loaded.get(0));
            assertZeroSequence(loaded.get(1), subspaceCount);
            assertByteSequenceEquals(mutablePQ.get(150), loaded.get(2));
            assertByteSequenceEquals(mutablePQ.get(5), loaded.get(3));
        }
    }

    @Test
    public void testRemappedScoring() {
        Random random = getRandom();
        int dimension = 16;
        int subspaceCount = 4;
        int numOriginalVectors = 100;

        List<VectorFloat<?>> originalVectorList = new ArrayList<>(numOriginalVectors);
        for (int i = 0; i < numOriginalVectors; i++) {
            originalVectorList.add(randomVector(random, dimension));
        }

        var originalValues = new ListRandomAccessVectorValues(originalVectorList, dimension);
        var pq = ProductQuantization.compute(originalValues, subspaceCount, 256, false);
        var mutablePQ = new MutablePQVectors(pq);
        for (int i = 0; i < numOriginalVectors; i++) {
            mutablePQ.encodeAndSet(i, originalVectorList.get(i));
        }

        Int2IntHashMap oldToNew = new Int2IntHashMap(Integer.MIN_VALUE);
        oldToNew.put(20, 0);
        oldToNew.put(80, 1);
        OrdinalMapper mapper = new OrdinalMapper.MapMapper(oldToNew);

        PQVectors remapped = mutablePQ.remap(3, mapper); // 3rd is omitted
        VectorFloat<?> query = randomVector(random, dimension);

        var scoreOrig = mutablePQ.scoreFunctionFor(query, VectorSimilarityFunction.COSINE);
        var scoreRemapped = remapped.scoreFunctionFor(query, VectorSimilarityFunction.COSINE);

        assertEquals(scoreOrig.similarityTo(20), scoreRemapped.similarityTo(0), 1e-6f);
        assertEquals(scoreOrig.similarityTo(80), scoreRemapped.similarityTo(1), 1e-6f);
        assertEquals(0.0f, scoreRemapped.similarityTo(2), 1e-6f);
    }

    private static void assertByteSequenceEquals(ByteSequence<?> expected, ByteSequence<?> actual) {
        assertEquals(expected.length(), actual.length());
        for (int i = 0; i < expected.length(); i++) {
            assertEquals(expected.get(i), actual.get(i));
        }
    }

    private static void assertZeroSequence(ByteSequence<?> seq, int length) {
        assertEquals(length, seq.length());
        for (int i = 0; i < length; i++) {
            assertEquals(0, seq.get(i));
        }
    }
}
