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

import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.Random;
import org.junit.Test;

import static org.junit.Assert.*;

/** Public scorer construction rejects invalid shapes before projection or native access. */
public class TestASHNeighborhoodScorer {
    @Test
    public void rejectsWrongQueryDimensionsAndInvalidCapacity() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(14932);
        VectorFloat<?>[] input = new VectorFloat<?>[ 3 ];
        for (int i = 0; i < input.length; i++) {
            float[] values = new float[64];
            for (int d = 0; d < values.length; d++)
                values[d] = random.nextFloat() - .5f;
            input[i] = types.createFloatVector(values);
        }
        var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(input), AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 33 * 4, 1, 4);
        assertThrows(IllegalArgumentException.class,
                () -> new ASHNeighborhoodScorer(ash, types.createFloatVector(63), 32, 32));
        assertThrows(IllegalArgumentException.class,
                () -> new ASHNeighborhoodScorer(ash, types.createFloatVector(65), 32, 32));
        assertThrows(IllegalArgumentException.class, () -> new ASHNeighborhoodScorer(ash, input[0], 0, 32));
        assertThrows(IllegalArgumentException.class, () -> new ASHNeighborhoodScorer(ash, input[0], 32, 31));
        assertThrows(ArithmeticException.class, () -> new ASHNeighborhoodScorer(ash, input[0], Integer.MAX_VALUE, 32));
    }

    @Test
    public void layoutSizesFailOnOverflowInsteadOfWrapping() {
        assertEquals(1_073_741_824, FusedASHLayout.codeGroups(Integer.MAX_VALUE, 2));
        assertEquals(67_108_864, FusedASHLayout.blocksPerNode(Integer.MAX_VALUE, 32));
        assertThrows(ArithmeticException.class, () -> FusedASHLayout.blockBodyBytes(Integer.MAX_VALUE, 4, 32));
        assertThrows(ArithmeticException.class, () -> FusedASHLayout.featureSize(Integer.MAX_VALUE, 512, 4, 32));
        assertThrows(IllegalArgumentException.class, () -> FusedASHLayout.blockOffset(-1, 33, 4, 32));
    }
}
