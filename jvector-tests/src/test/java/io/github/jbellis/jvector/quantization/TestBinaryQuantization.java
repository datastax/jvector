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
import io.github.jbellis.jvector.graph.ListRandomAccessVectorRepresentations;
import io.github.jbellis.jvector.vector.VectorSimilarityType;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestBinaryQuantization extends RandomizedTest
{
    @Test
    public void testMutableImmutableBQEquality()
    {
        var vectors = createRandomVectors(512, 64);
        var ravv = new ListRandomAccessVectorRepresentations(vectors, 64);
        var bq = new BinaryQuantization(ravv.dimension());
        var immutableCompressedVectors = bq.encodeAll(ravv);
        var mutableCompressedVectors = new MutableBQVectors(bq);
        for (int i = 0; i < ravv.size(); i++)
        {
            mutableCompressedVectors.encodeAndSet(i, ravv.getVector(i));
        }
        assertEquals(mutableCompressedVectors.count(), immutableCompressedVectors.count());
        var randomVector = TestUtil.randomVector(getRandom(), 64);
        for (VectorSimilarityType vsf : VectorSimilarityType.values())
        {
            var immutableScoreFunction = immutableCompressedVectors.scoreFunctionFor(randomVector, vsf);
            var mutableScoreFunction = mutableCompressedVectors.scoreFunctionFor(randomVector, vsf);
            for (int i = 0; i < ravv.size(); i++)
            {
                assertEquals(immutableScoreFunction.similarityTo(i), mutableScoreFunction.similarityTo(i));
            }
        }
    }
}
