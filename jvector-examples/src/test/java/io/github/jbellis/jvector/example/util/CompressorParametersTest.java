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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class CompressorParametersTest {
    /// Only the name feeds the cache key; the rest is never called.
    private static DataSet named(String name) {
        return new DataSet() {
            public int getDimension() { throw new UnsupportedOperationException(); }
            public String getName() { return name; }
            public VectorSimilarityFunction getSimilarityFunction() { throw new UnsupportedOperationException(); }
            public RandomAccessVectorValues getBaseRavv() { throw new UnsupportedOperationException(); }
            public List<VectorFloat<?>> getBaseVectors() { throw new UnsupportedOperationException(); }
            public List<VectorFloat<?>> getQueryVectors() { throw new UnsupportedOperationException(); }
            public List<List<Integer>> getGroundTruth() { throw new UnsupportedOperationException(); }
        };
    }

    @Test
    public void nvqCacheKeyFormats() {
        var params = new CompressorParameters.NVQParameters(2);
        assertTrue(params.supportsCaching());
        assertEquals("NVQ_ada002-100k_2", params.idStringFor(named("ada002-100k")));
    }
}
