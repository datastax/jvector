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

import java.util.Arrays;
import java.util.List;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TestKernelContracts {
    private final PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();

    @Test
    void storageHelpersPreserveOffsetsGatherOrderAndMaskedSentinels() {
        for (var species : List.of(FloatVector.SPECIES_64, FloatVector.SPECIES_128,
                FloatVector.SPECIES_256, FloatVector.SPECIES_PREFERRED)) {
            int lanes = species.length();
            float[] source = new float[lanes + 7];
            for (int i = 0; i < source.length; i++) source[i] = i + 0.25f;
            var input = new ArrayVectorFloat(source);
            var loaded = support.fromVectorFloat(species, input, 3);
            float[] target = new float[lanes + 7];
            Arrays.fill(target, -99f);
            support.intoVectorFloat(loaded, new ArrayVectorFloat(target), 2);
            for (int i = 0; i < target.length; i++)
                assertEquals(i >= 2 && i < lanes + 2 ? source[i + 1] : -99f, target[i]);
            int[] indices = new int[lanes + 2];
            for (int i = 0; i < lanes; i++) indices[i + 1] = lanes - i - 1;
            var gathered = support.fromVectorFloat(species, input, 3, indices, 1);
            for (int i = 0; i < lanes; i++) assertEquals(source[3 + lanes - i - 1], gathered.lane(i));
        }
        for (var species : List.of(ByteVector.SPECIES_64, ByteVector.SPECIES_128,
                ByteVector.SPECIES_256, ByteVector.SPECIES_PREFERRED)) {
            int lanes = species.length();
            byte[] source = new byte[lanes + 7];
            for (int i = 0; i < source.length; i++) source[i] = (byte) (i * 19);
            var loaded = support.fromByteSequence(species, new ArrayByteSequence(source), 3);
            byte[] target = new byte[lanes + 7];
            Arrays.fill(target, (byte) -99);
            support.intoByteSequence(loaded, new ArrayByteSequence(target), 2);
            for (int i = 0; i < target.length; i++)
                assertEquals(i >= 2 && i < lanes + 2 ? source[i + 1] : (byte) -99, target[i]);
            Arrays.fill(target, (byte) -99);
            int valid = lanes - 1;
            support.intoByteSequence(loaded, new ArrayByteSequence(target), 2, species.indexInRange(0, valid));
            for (int i = 0; i < target.length; i++)
                assertEquals(i >= 2 && i < valid + 2 ? source[i + 1] : (byte) -99, target[i]);
        }
    }

    @Test
    void partialTablesMatchScalarDistancesAndPreserveOtherCodebooks() {
        for (int size : new int[]{1, 2, 4, 8, 16, 17, 33, 768, 1536, 3072}) {
            int clusters = 7, index = 2, queryOffset = 3;
            float[] centroids = new float[clusters * size], query = new float[size + queryOffset];
            for (int i = 0; i < centroids.length; i++) centroids[i] = (i % 13 - 6) / 16f;
            for (int i = 0; i < query.length; i++) query[i] = (i % 17 - 8) / 32f;
            for (var metric : new VectorSimilarityFunction[]{VectorSimilarityFunction.DOT_PRODUCT,
                    VectorSimilarityFunction.EUCLIDEAN}) {
                float[] table = new float[clusters * 4];
                Arrays.fill(table, -99f);
                support.calculatePartialSums(new ArrayVectorFloat(centroids), index, size, clusters,
                        new ArrayVectorFloat(query), queryOffset, metric, new ArrayVectorFloat(table));
                for (int row = 0; row < table.length; row++) {
                    double expected = -99;
                    if (row >= index * clusters && row < (index + 1) * clusters) {
                        expected = 0;
                        for (int j = 0; j < size; j++) {
                            double a = centroids[(row - index * clusters) * size + j], b = query[queryOffset + j];
                            expected += metric == VectorSimilarityFunction.DOT_PRODUCT ? a * b : (a - b) * (a - b);
                        }
                    }
                    assertEquals(expected, table[row], 0.0);
                }
            }
        }
        assertThrows(UnsupportedOperationException.class, () -> support.calculatePartialSums(
                new ArrayVectorFloat(new float[]{1}), 0, 1, 1, new ArrayVectorFloat(new float[]{1}),
                0, VectorSimilarityFunction.COSINE, new ArrayVectorFloat(1)));
    }

    @Test
    void listSumReturnsIndependentStorageAndRetainsValidation() {
        for (int length : new int[]{0, 1, 2, 3, 8, 17, 768, 3072}) {
            float[] a = new float[length], b = new float[length];
            for (int i = 0; i < length; i++) { a[i] = i / 16f; b[i] = -i / 32f; }
            var av = new ArrayVectorFloat(a.clone());
            var bv = new ArrayVectorFloat(b.clone());
            var result = support.sum(List.of(av, bv, av));
            for (int i = 0; i < length; i++) assertEquals(a[i] + b[i] + a[i], result.get(i));
            if (length > 0) result.set(0, 100);
            assertArrayEquals(a, av.get());
            assertArrayEquals(b, bv.get());
        }
        assertThrows(IllegalArgumentException.class, () -> support.sum((List<io.github.jbellis.jvector.vector.types.VectorFloat<?>>) null));
        assertThrows(IllegalArgumentException.class, () -> support.sum(List.of()));
        assertThrows(IllegalArgumentException.class, () -> support.cosine(new ArrayVectorFloat(1), new ArrayVectorFloat(2)));
    }
}
