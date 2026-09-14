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

import io.github.jbellis.jvector.quantization.NVQuantization;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class NvqGrowthRateTest {
    @BeforeAll
    static void requireNativeProvider() {
        NativeTestSupport.requireNativeProvider();
    }

    private static float reference(VectorFloatInput input) {
        var v = input.vector;
        float baseline = VectorUtil.nvqUniformLoss(v, input.min, input.max, 8);
        float coarse = 1e-2f, best = Float.MIN_VALUE;
        for (float gr = 1e-6f; gr < 20.f; gr += 1f) {
            float value = baseline / VectorUtil.nvqLoss(v, gr, 0.f, input.min, input.max, 8);
            if (value > best) { best = value; coarse = gr; }
        }
        float fine = coarse;
        for (float gr = coarse - 1; gr < coarse + 1; gr += .1f) {
            float value = baseline / VectorUtil.nvqLoss(v, gr, 0.f, input.min, input.max, 8);
            if (value > best) { best = value; fine = gr; }
        }
        return fine;
    }
    private record VectorFloatInput(io.github.jbellis.jvector.vector.types.VectorFloat<?> vector,
                                    float min, float max) {}

    @Test void preservesNativeSearchAndEncodedBytes() {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(5);
        for (int dim : new int[]{1, 3, 8, 15, 16, 17, 31, 32, 33, 127, 256, 768, 1536, 3072}) {
            for (int trial = 0; trial < 30; trial++) {
                float[] data = new float[dim];
                for (int i = 0; i < dim; i++) data[i] = trial == 0 ? 0 : trial == 1 ? 1 :
                        trial == 2 ? (i % 2 == 0 ? -0.f : 0.f) : (float)random.nextGaussian();
                var vector = types.createFloatVector(data);
                float min = VectorUtil.min(vector), max = VectorUtil.max(vector);
                float expected = reference(new VectorFloatInput(vector, min, max));
                float actual = VectorUtil.tryNvqGrowthRate(vector, min, max, 8);
                assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual),
                             "growth dim=" + dim + " trial=" + trial);
                var destination = NVQuantization.QuantizedSubVector.createEmpty(NVQuantization.BitsPerDimension.EIGHT, dim);
                NVQuantization.QuantizedSubVector.quantizeTo(vector, NVQuantization.BitsPerDimension.EIGHT, true, destination);
                var bytes = types.createByteSequence(dim);
                VectorUtil.nvqQuantize8bit(vector, expected, 0, min, max, bytes);
                assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(destination.growthRate));
                for (int i = 0; i < dim; i++) assertEquals(bytes.get(i), destination.bytes.get(i));
            }
        }
    }
}
