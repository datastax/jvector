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


package io.github.jbellis.jvector.vector;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** The batched similarity methods must give the same scores as scoring one pair at a time. */
public class TestBatchedSimilarity extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static VectorFloat<?> randomVector(Random random, int dim) {
        var v = vts.createFloatVector(dim);
        for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian());
        return v;
    }

    @Test
    public void testBatchKernelsMatchSinglePairs() {
        Random random = getRandom();
        for (int dim : new int[]{1, 3, 7, 16, 33, 128, 384}) {
            for (int count : new int[]{0, 1, 2, 5, 17, 64}) {
                var query = randomVector(random, dim);
                var vectors = randomVector(random, dim * count);
                float[] out = new float[count + 1];
                out[count] = 42f; // must stay untouched

                VectorUtil.squareL2DistanceBatch(query, vectors, count, out);
                for (int i = 0; i < count; i++) {
                    assertEquals("l2 dim " + dim + " i " + i, VectorUtil.squareL2Distance(query, 0, vectors, i * dim, dim), out[i], 0f);
                }
                assertEquals(42f, out[count], 0f);

                VectorUtil.dotProductBatch(query, vectors, count, out);
                for (int i = 0; i < count; i++) {
                    assertEquals("dot dim " + dim + " i " + i, VectorUtil.dotProduct(query, 0, vectors, i * dim, dim), out[i], 0f);
                }
                assertEquals(42f, out[count], 0f);

                VectorUtil.cosineBatch(query, vectors, count, out);
                var slice = vts.createFloatVector(dim);
                for (int i = 0; i < count; i++) {
                    slice.copyFrom(vectors, i * dim, 0, dim);
                    assertEquals("cosine dim " + dim + " i " + i, VectorUtil.cosine(query, slice), out[i], 0f);
                }
                assertEquals(42f, out[count], 0f);
            }
        }
    }

    @Test
    public void testCompareBatchMatchesCompare() {
        Random random = getRandom();
        int dim = 37, count = 11;
        var query = randomVector(random, dim);
        var one = new VectorFloat<?>[count];
        var vectors = vts.createFloatVector(dim * count);
        for (int i = 0; i < count; i++) {
            one[i] = randomVector(random, dim);
            vectors.copyFrom(one[i], 0, i * dim, dim);
        }
        for (var vsf : VectorSimilarityFunction.values()) {
            float[] out = new float[count];
            vsf.compareBatch(query, vectors, count, out);
            for (int i = 0; i < count; i++) {
                assertEquals(vsf + " i " + i, vsf.compare(query, one[i]), out[i], 1e-6f);
            }
        }
    }

    @Test
    public void testBatchBoundsAreChecked() {
        var query = vts.createFloatVector(8);
        var vectors = vts.createFloatVector(8 * 3);
        assertThrows(IllegalArgumentException.class, () -> VectorUtil.squareL2DistanceBatch(query, vectors, 4, new float[4]));
        assertThrows(IllegalArgumentException.class, () -> VectorUtil.dotProductBatch(query, vectors, 3, new float[2]));
        assertThrows(IllegalArgumentException.class, () -> VectorUtil.cosineBatch(query, vectors, -1, new float[3]));
    }
}
