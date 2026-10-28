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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ASHNeighborhoodScorer;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.FusedASHLayout;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import static org.junit.Assert.*;

/** Recall ranking is checked independently of the heap and excludes packed padding. */
public class TestASHNeighborhoodRecall {
    @Test
    public void ranksTheCompleteBaseSetAndExcludesPadding() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> base = new ArrayList<>();
        var random = new Random(9294);
        for (int i = 0; i < 11; i++) {
            float[] vector = new float[64];
            for (int d = 0; d < vector.length; d++) {
                vector[d] = random.nextFloat() - .5f;
            }
            base.add(types.createFloatVector(vector));
        }
        var ash = AsymmetricHashing.initialize(new ListRandomAccessVectorValues(base, 64), AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 33 * 2, 1, 2);
        ASHVectors encoded = (ASHVectors) ash.encodeAll(new ListRandomAccessVectorValues(base, 64));
        byte[][] packed = new byte[2][FusedASHLayout.featureSize(8, 33, 2, 8)];
        for (int i = 0; i < base.size(); i++) {
            FusedASHLayout.packForFused(packed[i / 8], 0, i % 8, encoded.get(i), ash, 8);
        }
        // Maliciously attractive padding must never displace a real result.
        for (int lane = 3; lane < 8; lane++) {
            int offset = FusedASHLayout.offsetOffset(0, 33, 2, 8) + lane * 2;
            packed[1][offset] = 0x7b;
            packed[1][offset + 1] = (byte) 0xff; // Largest finite positive binary16.
        }
        var fixture = new ASHNeighborhoodFixture(ash, List.of(base.get(0)), packed, 8, 8, 11);
        float[] all = new float[16];
        float[] scratch = new float[8];
        var scorer = new ASHNeighborhoodScorer(ash, base.get(0), 8, 8);
        for (int i = 0; i < packed.length; i++) {
            scorer.score(packed[i], scratch);
            System.arraycopy(scratch, 0, all, i * 8, 8);
        }
        Integer[] sorted = new Integer[11];
        Arrays.setAll(sorted, i -> i);
        Arrays.sort(sorted, Comparator.<Integer>comparingDouble(i -> all[i]).reversed().thenComparingInt(i -> i));
        int[] expected = Arrays.stream(sorted).limit(10).mapToInt(Integer::intValue).toArray();
        var result = ASHNeighborhoodRecall.scan(fixture, 0, expected);
        assertArrayEquals(expected, result.ids);
        assertEquals(10, result.hits);
        for (int id : result.ids) {
            assertTrue(id < fixture.baseCount);
        }
    }

    @Test
    public void tiesPreferSmallerOrdinalsRegardlessOfOfferOrder() {
        var top = new ASHNeighborhoodRecall.TopK();
        for (int id = 24; id >= 0; id--) {
            top.offer(.5f, id);
        }
        var ranked = top.ranked();
        assertEquals(10, ranked.length);
        for (int i = 0; i < ranked.length; i++) {
            assertEquals(i, ranked[i].ordinal);
        }
    }

    @Test
    public void validatesLittleEndianGroundTruthAndRejectsBadRows() throws Exception {
        var path = Files.createTempFile("ash-recall-", ".ivecs");
        try {
            ByteBuffer bytes = ByteBuffer.allocate(11 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            bytes.putInt(10);
            for (int id = 0; id < 10; id++) {
                bytes.putInt(id);
            }
            Files.write(path, bytes.array());
            assertArrayEquals(
                    new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ASHNeighborhoodRecall.readGroundTruth(path, 1, 11)[0]);
            bytes.putInt(10 * Integer.BYTES, 11);
            Files.write(path, bytes.array());
            assertThrows(IOException.class, () -> ASHNeighborhoodRecall.readGroundTruth(path, 1, 11));
            bytes.putInt(10 * Integer.BYTES, 0);
            Files.write(path, bytes.array());
            assertThrows(IOException.class, () -> ASHNeighborhoodRecall.readGroundTruth(path, 1, 11));
            Files.write(path, Arrays.copyOf(bytes.array(), 20));
            assertThrows(IOException.class, () -> ASHNeighborhoodRecall.readGroundTruth(path, 1, 11));
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
