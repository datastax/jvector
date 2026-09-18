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
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Reproduces <a href="https://github.com/riptano/cndb/issues/18237">cndb#18237</a>: hierarchical
 * search with {@link VectorSimilarityFunction#DOT_PRODUCT} and non-unit-normalized vectors can
 * throw {@code AssertionError: 0} from {@link GraphSearcher#internalSearch}.
 *
 * <p>{@code DOT_PRODUCT} similarity is {@code (1 + dotProduct(v1, v2)) / 2}, which is only
 * guaranteed to land in {@code (0, 1]} when both vectors are unit-length. With larger-magnitude,
 * non-normalized vectors (the deliberate "true dot product" use case this test mirrors, taken
 * from Cassandra's {@code VectorDotProductWithLengthTest#testTrueDotproduct}), the raw score can
 * legitimately be negative. {@code internalSearch}'s hierarchy-descent loop searches each upper
 * layer for a single best candidate using a hardcoded {@code 0.0f} threshold; if every candidate
 * reachable from the entry point in that layer scores below zero for a given query, all of them
 * are filtered out, leaving {@code approximateResults} empty and tripping the
 * {@code assert approximateResults.size() == 1} a few lines later.
 *
 * <p>This only manifests with hierarchy enabled: layer-0 search uses a much larger candidate
 * pool and generally still finds enough positively-scored neighbors, but the hierarchy-descent
 * layers search with {@code rerankK=1} from a single fixed entry point, making it far more
 * likely that every candidate examined happens to score negative for an unlucky query direction.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestHierarchyDotProductSearch extends LuceneTestCase {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    // Mirrors Cassandra's create2DVector(): low dimensionality and a wide magnitude range so
    // dot products routinely fall well outside [-1, 1], unlike unit-normalized test vectors.
    private VectorFloat<?> randomUnnormalizedVector() {
        float x = getRandom().nextFloat() * 200f - 100f;
        float y = getRandom().nextFloat() * 200f - 100f;
        return vts.createFloatVector(new float[]{x, y});
    }

    @Test
    public void testHierarchySearchWithNonUnitDotProductVectors() {
        int nDoc = 2000;
        int dimension = 2;

        List<VectorFloat<?>> vectors = new ArrayList<>(nDoc);
        for (int i = 0; i < nDoc; i++) {
            vectors.add(randomUnnormalizedVector());
        }
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);

        // addHierarchy=true is required to reach the buggy code path; M/beamWidth mirror
        // typical defaults, not anything specific to reproduction.
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.DOT_PRODUCT,
                16, 100, 1.2f, 1.2f, true);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // Enough queries that, pre-fix, at least one is very likely to hit a hierarchy layer
        // where every reachable candidate scores negative against it.
        for (int q = 0; q < 200; q++) {
            VectorFloat<?> query = randomUnnormalizedVector();
            // Pre-fix, this throws AssertionError(0) from GraphSearcher.internalSearch (line 263)
            // via searchOneLayer's hardcoded 0.0f threshold during hierarchy descent.
            GraphSearcher.search(query, 10, ravv.copy(), VectorSimilarityFunction.DOT_PRODUCT, graph, Bits.ALL);
        }
    }
}
