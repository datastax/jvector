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

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import static org.junit.Assert.*;

public class TestVectorSourceSampling {
    @Test(timeout=60000) public void floydSampleAndTrainingOrderAreUnchanged() throws Exception {
        int count = 129003;
        var path = Files.createTempFile("pq-sample-source", ".fvecs");
        var buffer = ByteBuffer.allocate(count * 12).order(ByteOrder.LITTLE_ENDIAN);
        var values = new ArrayList<VectorFloat<?>>();
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int i = 0; i < count; i++) {
            buffer.putInt(2).putFloat(i).putFloat(-i);
            values.add(vts.createFloatVector(new float[] {i, -i}));
        }
        Files.write(path, buffer.array());
        var pool = new ForkJoinPool(48);
        try (var source = FvecFileVectorValues.open(path)) {
            var expected = ProductQuantization.extractTrainingVectors(new ListRandomAccessVectorValues(values, 2), pool);
            var actual = ProductQuantization.extractTrainingVectors(source, pool);
            assertEquals(128000, actual.size());
            for (int i = 0; i < actual.size(); i++) assertEquals(expected.get(i), actual.get(i));
            assertTrue(source.statistics().bufferBytes <= source.statistics().bufferLimitBytes);
        } finally { pool.shutdown(); Files.delete(path); }
    }
}
