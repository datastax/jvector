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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class BuildScoreProviderTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Test that the ordinal mapping is correctly applied when creating search and diversity score providers.
     */
    @Test
    public void testOrdinalMapping() {
        final VectorSimilarityFunction vsf = VectorSimilarityFunction.DOT_PRODUCT;

        // Create test vectors
        final List<VectorFloat<?>> vectors = new ArrayList<>();
        vectors.add(vts.createFloatVector(new float[]{1.0f, 0.0f}));
        vectors.add(vts.createFloatVector(new float[]{0.0f, 1.0f}));
        vectors.add(vts.createFloatVector(new float[]{-1.0f, 0.0f}));
        var ravv = new ListRandomAccessVectorValues(vectors, 2);

        // Create non-identity mapping: graph node 0 -> ravv ordinal 2, graph node 1 -> ravv ordinal 0, graph node 2 -> ravv ordinal 1
        int[] graphToRavvOrdMap = {2, 0, 1};
        
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, graphToRavvOrdMap, vsf);
        
        // Test that searchProviderFor(graphNode) uses the correct RAVV ordinal
        var ssp0 = bsp.searchProviderFor(0); // should use ravv ordinal 2 (vector [-1, 0])
        var ssp1 = bsp.searchProviderFor(1); // should use ravv ordinal 0 (vector [1, 0])
        var ssp2 = bsp.searchProviderFor(2); // should use ravv ordinal 1 (vector [0, 1])
        
        // Verify by computing similarity between graph nodes
        // Graph node 0 (vector 2:[-1, 0]) vs graph node 1 (vector 0:[1, 0])
        assertEquals(vsf.compare(vectors.get(2), vectors.get(0)), ssp0.exactScoreFunction().similarityTo(1), 1e-6f);
        
        // Graph node 1 (vector 0:[1, 0]) vs graph node 0 (vector 2:[-1, 0])
        assertEquals(vsf.compare(vectors.get(0), vectors.get(2)), ssp1.exactScoreFunction().similarityTo(0), 1e-6f);
        
        // Graph node 2 (vector 1:[0, 1]) vs graph node 1 (vector 0:[1, 0])
        assertEquals(vsf.compare(vectors.get(1), vectors.get(0)), ssp2.exactScoreFunction().similarityTo(1), 1e-6f);
        
        // Test diversityProviderFor uses same mapping, Graph node 0 (vector 2:[-1, 0]) vs graph node 1 (vector 0:[1, 0])
        var dsp0 = bsp.diversityProviderFor(0);
        assertEquals(vsf.compare(vectors.get(2), vectors.get(0)), dsp0.exactScoreFunction().similarityTo(1), 1e-6f);
    }
}