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

import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RangeRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/// Tests that {@link DataSetPartitioner} produces contiguous, non-copying ranged views.
public class DataSetPartitionerTest {

    @Test
    public void partitionsAreContiguousRangedViews() {
        int count = 23, dimension = 2;
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> vectors = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            vectors.add(vts.createFloatVector(new float[] {i, -i}));
        }
        var base = new ListRandomAccessVectorValues(vectors, dimension);

        var parts = DataSetPartitioner.partition(base, 4, TestDataPartition.Distribution.UNIFORM);
        assertEquals(4, parts.vectors.size());
        assertEquals(4, parts.sizes.size());
        assertEquals(count, parts.sizes.stream().mapToInt(Integer::intValue).sum());

        int globalOrdinal = 0;
        for (int p = 0; p < 4; p++) {
            var view = parts.vectors.get(p);
            assertEquals(parts.sizes.get(p).intValue(), view.size());
            assertTrue(view instanceof RangeRandomAccessVectorValues);
            assertEquals(globalOrdinal, ((RangeRandomAccessVectorValues) view).fromOrdinal());
            for (int i = 0; i < view.size(); i++) {
                assertSame(vectors.get(globalOrdinal), view.getVector(i));
                globalOrdinal++;
            }
        }
        assertEquals(count, globalOrdinal);
    }
}
