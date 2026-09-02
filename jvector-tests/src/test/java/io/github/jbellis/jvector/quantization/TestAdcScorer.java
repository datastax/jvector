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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;

/** The scorer over a supplied code equals the PQVectors decoders over the stored one, for every similarity. */
public class TestAdcScorer {
    @Test
    public void testMatchesPqVectorsDecoders() {
        int dim = 24, n = 400;
        var rnd = new Random(17);
        List<VectorFloat<?>> vecs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs.add(TestUtil.randomVector(rnd, dim));
        }
        var ravv = new ListRandomAccessVectorValues(vecs, dim);
        var pq = ProductQuantization.compute(ravv, 4, 32, true);
        var pqv = (PQVectors) pq.encodeAll(ravv, ForkJoinPool.commonPool());
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (var vsf : new VectorSimilarityFunction[] {VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.COSINE}) {
            var scorer = new AdcScorer(pq, vsf);
            assertEquals(pq.getSubspaceCount(), scorer.codeSize());
            for (int q = 0; q < 5; q++) {
                VectorFloat<?> query = TestUtil.randomVector(rnd, dim);
                scorer.setQuery(query);
                var reference = pqv.precomputedScoreFunctionFor(query, vsf);
                for (int i = 0; i < n; i++) {
                    assertEquals(vsf + " node " + i, reference.similarityTo(i), scorer.similarityTo(pq.encode(vecs.get(i))), 1e-5f);
                }
            }
            // decode restores the global centroid, like the PQ itself
            var a = vts.createFloatVector(dim);
            var b = vts.createFloatVector(dim);
            var code = pq.encode(vecs.get(3));
            scorer.decode(code, a);
            pq.decode(code, b);
            for (int d = 0; d < dim; d++) {
                assertEquals(b.get(d), a.get(d), 0f);
            }
        }
    }
}
