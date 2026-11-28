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

import org.junit.Test;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestBinaryQuantization extends RandomizedTest
{
    @Test
    public void testMutableImmutableBQEquality()
    {
        var vectors = createRandomVectors(512, 64);
        var ravv = new ListRandomAccessVectorValues(vectors, 64);
        var bq = new BinaryQuantization(ravv.dimension());
        var immutableCompressedVectors = bq.encodeAll(ravv);
        var mutableCompressedVectors = new MutableBQVectors(bq);
        for (int i = 0; i < ravv.size(); i++)
        {
            mutableCompressedVectors.encodeAndSet(i, ravv.getVector(i));
        }
        assertEquals(mutableCompressedVectors.count(), immutableCompressedVectors.count());
        var randomVector = TestUtil.randomVector(getRandom(), 64);
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values())
        {
            var immutableScoreFunction = immutableCompressedVectors.scoreFunctionFor(randomVector, vsf);
            var mutableScoreFunction = mutableCompressedVectors.scoreFunctionFor(randomVector, vsf);
            for (int i = 0; i < ravv.size(); i++)
            {
                assertEquals(immutableScoreFunction.similarityTo(i), mutableScoreFunction.similarityTo(i));
            }
        }
    }

    @Test
    public void testConcurrentEncodeAndSet() throws Exception
    {
        // Enough ordinals to force several array growths while threads insert.
        int size = 20_000;
        int dimension = 64;
        int threads = 16;
        var vectors = createRandomVectors(size, dimension);
        var bq = new BinaryQuantization(dimension);
        var mutable = new MutableBQVectors(bq);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        try
        {
            var futures = new ArrayList<Future<?>>();
            for (int t = 0; t < threads; t++)
            {
                int start = t;
                futures.add(executor.submit(() -> {
                    // Interleaved ordinals so threads race on the same growth boundaries.
                    for (int i = start; i < size; i += threads)
                    {
                        if (i % 7 == 0)
                            mutable.setZero(i);
                        else
                            mutable.encodeAndSet(i, vectors.get(i));
                    }
                }));
            }
            for (var f : futures)
                f.get();
        }
        finally
        {
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }

        assertEquals(size, mutable.count());
        var zero = new long[bq.compressedVectorSize() / Long.BYTES];
        for (int i = 0; i < size; i++)
        {
            int ordinal = i;
            long[] expected = i % 7 == 0 ? zero : bq.encode(vectors.get(i));
            assertArrayEquals(expected, mutable.get(i), () -> "ordinal " + ordinal);
        }
    }

    @Test
    public void testInvalidOrdinal()
    {
        var bq = new BinaryQuantization(64);
        var mutable = new MutableBQVectors(bq);
        var vector = TestUtil.randomVector(getRandom(), 64);
        assertThrows(IllegalArgumentException.class, () -> mutable.encodeAndSet(-1, vector));
        assertThrows(IllegalArgumentException.class, () -> mutable.setZero(Integer.MAX_VALUE));
        assertEquals(0, mutable.count());
    }
}
