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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class TestASHQueryOptions {
    private static AsymmetricHashing model() throws java.io.IOException {
        var type = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> vectors = new ArrayList<>();
        var random = new Random(4211);
        for (int n = 0; n < 64; n++) {
            var v = type.createFloatVector(32);
            for (int d = 0; d < 32; d++) v.set(d, random.nextFloat() - 0.5f);
            VectorUtil.l2normalize(v);
            vectors.add(v);
        }
        var values = new ListRandomAccessVectorValues(vectors, 32);
        return AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM, AsymmetricHashing.HEADER_BITS + 32, 1, 1);
    }

    @Test
    public void optionsApplyBeforeTheFirstQuery() throws Exception {
        var ash = model();
        ash.setQueryOptions(ASHQueryProjection.FP32, 1);
        assertEquals(ASHQueryProjection.FP32, ash.getQueryProjection());
        assertEquals(1, ash.getQueryBatchSize());

        float[] query = new float[ash.stiefelTransform.cols];
        query[0] = 1;
        float[] projected = new float[ash.stiefelTransform.rows];
        float[] expected = new float[ash.stiefelTransform.rows];
        ash.stiefelTransform.projectQuery(query, projected);
        ash.stiefelTransform.project(query, expected);
        // FP32 query projection matches the FP32 encoding projection exactly.
        assertArrayEquals(expected, projected, 0f);
        assertEquals("fp32", ash.stiefelTransform.initializedQueryProjectionPrecision());
        // A batch size of 1 never combines concurrent queries.
        assertEquals(null, ash.stiefelTransform.queryProjectionBatchHistogram());
    }

    @Test
    public void optionsCannotChangeAfterTheFirstQuery() throws Exception {
        var ash = model();
        ash.stiefelTransform.projectQuery(new float[ash.stiefelTransform.cols], new float[ash.stiefelTransform.rows]);
        assertThrows(IllegalStateException.class, () -> ash.setQueryOptions(ASHQueryProjection.FP32, 4));
    }

    @Test
    public void optionsAreValidated() throws Exception {
        var ash = model();
        assertThrows(NullPointerException.class, () -> ash.setQueryOptions(null, 4));
        assertThrows(IllegalArgumentException.class, () -> ash.setQueryOptions(ASHQueryProjection.AUTO, 0));
        assertEquals(ASHQueryProjection.INT8, ASHQueryProjection.parse(" Int8 "));
        assertThrows(IllegalArgumentException.class, () -> ASHQueryProjection.parse("bf16"));
    }
}
