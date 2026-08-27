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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/** The key is a pure function of the vector: same across instances, so streams of any generation compare. */
public class TestSimilarityKey {
    @Test
    public void testDeterministicAcrossInstances() {
        var a = SimilarityKey.randomProjection(64);
        var b = SimilarityKey.randomProjection(64);
        var rnd = new Random(3);
        for (int i = 0; i < 100; i++) {
            VectorFloat<?> v = TestUtil.randomVector(rnd, 64);
            assertEquals(a.keyOf(v), b.keyOf(v));
        }
        assertEquals(SimilarityKey.RANDOM_PROJECTION, a.id());
    }

    @Test
    public void testNegationFlipsEveryBit() {
        var k = SimilarityKey.randomProjection(48);
        var rnd = new Random(5);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int i = 0; i < 50; i++) {
            VectorFloat<?> v = TestUtil.randomVector(rnd, 48);
            VectorFloat<?> neg = vts.createFloatVector(48);
            for (int d = 0; d < 48; d++) {
                neg.set(d, -v.get(d));
            }
            // a dot product that is exactly zero would keep its bit; with random vectors it never is
            assertEquals(~k.keyOf(v), k.keyOf(neg));
        }
    }

    @Test
    public void testDifferentVectorsDiffer() {
        var k = SimilarityKey.randomProjection(32);
        var rnd = new Random(9);
        assertNotEquals(k.keyOf(TestUtil.randomVector(rnd, 32)), k.keyOf(TestUtil.randomVector(rnd, 32)));
    }
}
