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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.NVQScorer;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class NvqSIMDTest {
    @Test void shufflePreservesEveryBit() {
        NativeTestSupport.requireNativeProvider();
        Random random = new Random(6);
        for (int dim : new int[]{0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1536, 3072}) {
            float[] data = new float[dim];
            for (int i = 0; i < dim; i++) data[i] = Float.intBitsToFloat(random.nextInt());
            var vector = new MemorySegmentVectorFloat(data.clone());
            VectorUtil.nvqShuffleQueryInPlace8bit(vector);
            int[] expected = new int[dim], actual = new int[dim];
            for (int i = 0; i < dim; i++) {
                expected[i] = Float.floatToRawIntBits(data[i]);
                actual[i] = Float.floatToRawIntBits(vector.get(i));
            }
            Arrays.sort(expected);
            Arrays.sort(actual);
            assertArrayEquals(expected, actual, "shuffle bits dim=" + dim);
        }
    }

    @Test void shuffledQueriesMatchScalarScoresAcrossBlockBoundaries() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var scalar = new DefaultVectorUtilSupport();
        Random random = new Random(61);
        for (int dim : new int[]{0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1536, 3072}) {
            float[] data = new float[dim];
            byte[] codes = new byte[dim];
            for (int i = 0; i < dim; i++) data[i] = random.nextFloat() * 2 - 1;
            random.nextBytes(codes);
            var original = new MemorySegmentVectorFloat(data);
            var shuffled = original.copy();
            var bytes = new MemorySegmentByteSequence(codes);
            support.nvqShuffleQueryInPlace8bit(shuffled);
            // Compare the complete shuffle/scoring contract, independent of the
            // JVM vector width or a capped/scalable native ISA's lane count.
            float dot = scalar.nvqDotProduct8bit(original, bytes, 1, 0, -2, 2);
            float distance = scalar.nvqSquareL2Distance8bit(original, bytes, 1, 0, -2, 2);
            assertEquals(dot, support.nvqDotProduct8bit(shuffled, bytes, 1, 0, -2, 2),
                         Math.max(1e-4f, Math.abs(dot) * 1e-5f), "dot dim=" + dim);
            assertEquals(distance, support.nvqSquareL2Distance8bit(shuffled, bytes, 1, 0, -2, 2),
                         Math.max(1e-4f, Math.abs(distance) * 1e-5f), "distance dim=" + dim);
        }
    }

    @Test void packedCosineScorerMatchesUnpackedAndScalarAcrossSubspacesAndTails() {
        var provider = NativeTestSupport.requireNativeProvider();
        var support = provider.getVectorUtilSupport();
        var types = provider.getVectorTypeSupport();
        var scalar = new DefaultVectorUtilSupport();
        assertTrue(support.supportsNvqCosine8bitPacked());
        Random random = new Random(62);
        for (int dim : new int[]{7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 1536}) {
            for (int subspaces : new int[]{1, 2, 3}) {
                float[] mean = new float[dim], data = new float[dim], queryData = new float[dim];
                for (int d = 0; d < dim; d++) {
                    mean[d] = random.nextFloat() - .5f;
                    data[d] = random.nextFloat() * 4 - 2;
                    queryData[d] = random.nextFloat() * 2 - 1;
                }
                var nvq = NVQuantization.create(types.createFloatVector(mean), subspaces);
                nvq.learn = false;
                var encoded = nvq.encode(types.createFloatVector(data));
                for (boolean zeroQuery : new boolean[]{false, true}) {
                    var query = types.createFloatVector(zeroQuery ? new float[dim] : queryData.clone());
                    var originalQuery = nvq.getSubVectors(query);
                    var originalMean = nvq.getSubVectors(nvq.globalMean);
                    var shuffledQuery = nvq.getSubVectors(query);
                    var shuffledMean = nvq.getSubVectors(nvq.globalMean);
                    float dot = 0, magnitude = 0, scalarDot = 0, scalarMagnitude = 0;
                    for (int i = 0; i < subspaces; i++) {
                        support.nvqShuffleQueryInPlace8bit(shuffledQuery[i]);
                        support.nvqShuffleQueryInPlace8bit(shuffledMean[i]);
                        var sv = encoded.subVectors[i];
                        float[] unpacked = support.nvqCosine8bit(shuffledQuery[i], sv.bytes,
                                sv.growthRate, sv.midpoint, sv.minValue, sv.maxValue, shuffledMean[i]);
                        long packed = support.nvqCosine8bitPacked(shuffledQuery[i], sv.bytes,
                                sv.growthRate, sv.midpoint, sv.minValue, sv.maxValue, shuffledMean[i]);
                        assertEquals(Float.floatToRawIntBits(unpacked[0]), (int) packed);
                        assertEquals(Float.floatToRawIntBits(unpacked[1]), (int) (packed >>> 32));
                        dot += unpacked[0];
                        magnitude += unpacked[1];
                        float[] reference = scalar.nvqCosine8bit(originalQuery[i], sv.bytes,
                                sv.growthRate, sv.midpoint, sv.minValue, sv.maxValue, originalMean[i]);
                        scalarDot += reference[0];
                        scalarMagnitude += reference[1];
                    }
                    float norm = (float) Math.sqrt(support.dotProduct(query, query));
                    float expected = (1 + (dot / norm) / (float) Math.sqrt(magnitude)) / 2;
                    var scalarQuery = new ArrayVectorFloat(zeroQuery ? new float[dim] : queryData.clone());
                    float scalarNorm = (float) Math.sqrt(scalar.dotProduct(scalarQuery, scalarQuery));
                    float scalarScore = (1 + (scalarDot / scalarNorm) / (float) Math.sqrt(scalarMagnitude)) / 2;
                    float actual = new NVQScorer(nvq).scoreFunctionFor(query, VectorSimilarityFunction.COSINE)
                            .similarityTo(encoded);
                    String context = "dim=" + dim + " subspaces=" + subspaces + " zeroQuery=" + zeroQuery;
                    assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual), context);
                    assertEquals(scalarScore, actual, 3e-5f, "Scalar cosine " + context);
                }
            }
        }
    }

    @Test void nvqSetupRetainsScopedArithmetic() {
        var methods = Arrays.stream(NativeVectorUtilSupport.class.getDeclaredMethods())
                .map(java.lang.reflect.Method::getName).toList();
        for (String scoped : new String[]{"nvqMin", "nvqScale", "nvqSub"}) assertTrue(methods.contains(scoped), scoped);
    }

    @Test void meanPreservesSequentialOrderAndInputs() {
        NativeTestSupport.requireNativeProvider();
        Random random = new Random(60);
        for (int dim : new int[]{1, 3, 15, 16, 17, 63, 64, 65, 1536}) {
            List<VectorFloat<?>> values = new ArrayList<>();
            float[] expected = new float[dim];
            for (int n = 0; n < 257; n++) {
                float[] data = new float[dim];
                for (int i = 0; i < dim; i++) {data[i]=(float)random.nextGaussian();expected[i]+=data[i];}
                values.add(new MemorySegmentVectorFloat(data));
            }
            var ravv = new ListRandomAccessVectorValues(values, dim);
            for (boolean nativeViews : new boolean[]{false, true}) {
                // PQ has already requested native views before the full NVQ compute phase.
                if (nativeViews) for (var value : values) ((MemorySegmentVectorFloat)value).get();
                var result = NVQuantization.compute(ravv, 1).globalMean;
                for (int i = 0; i < dim; i++) {
                    assertEquals(Float.floatToRawIntBits(expected[i] * (1.0f / values.size())),
                                 Float.floatToRawIntBits(result.get(i)), "mean " + dim + ":" + i);
                }
            }
        }
    }
}
