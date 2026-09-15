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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/** The code-code similarity must equal the exact similarity of the decoded vectors, centered or not, for every supported function. */
public class TestSymmetricCodeSimilarity {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void testMatchesDecodedSimilarity() {
        int dim = 32, n = 2000; Random rnd = new Random(7);
        List<VectorFloat<?>> vecs = new ArrayList<>();
        for (int i = 0; i < n; i++) { var v = vts.createFloatVector(dim); double s = 0; for (int d = 0; d < dim; d++) { float x = (float) rnd.nextGaussian() + (d < 4 ? 2f : 0f); v.set(d, x); s += x * x; } for (int d = 0; d < dim; d++) v.set(d, (float) (v.get(d) / Math.sqrt(s))); vecs.add(v); }
        var ravv = new ListRandomAccessVectorValues(vecs, dim);
        for (boolean centered : new boolean[]{false, true}) {
            var pq = ProductQuantization.compute(ravv, 8, 16, centered);
            var codes = new ArrayList<byte[]>(); var decoded = new ArrayList<VectorFloat<?>>();
            for (int i = 0; i < 200; i++) { ByteSequence<?> c = pq.encode(vecs.get(i)); byte[] b = new byte[c.length()]; for (int m = 0; m < b.length; m++) b[m] = c.get(m); codes.add(b);
                var d = vts.createFloatVector(dim); pq.decode(c, d); decoded.add(d); }
            for (var vsf : new VectorSimilarityFunction[]{VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.COSINE}) {
                var sim = new SymmetricCodeSimilarity(pq, vsf);
                double maxErr = 0;
                for (int i = 0; i < 200; i++) for (int j = i; j < 200; j += 7) {
                    float exact = vsf.compare(decoded.get(i), decoded.get(j));
                    float approx = sim.similarity(codes.get(i), codes.get(j));
                    maxErr = Math.max(maxErr, Math.abs(exact - approx));
                }
                assertEquals("centered=" + centered + " " + vsf + " max |exact-code| = " + maxErr, 0.0, maxErr, 2e-4);
            }
        }
    }
}
