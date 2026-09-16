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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/** The blocked PQ scan (provider path and native kernel when present) against a plain reference. */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPqScanKernel extends RandomizedTest {

    private static void reference(byte[] blocks, int blockCount, int m, byte[] lut, short[] out) {
        for (int b = 0; b < blockCount; b++) {
            for (int i = 0; i < 64; i++) {
                int sum = 0;
                for (int s = 0; s < m; s++) {
                    sum += lut[s * 256 + (blocks[(b * m + s) * 64 + i] & 0xFF)] & 0xFF;
                }
                out[b * 64 + i] = (short) sum;
            }
        }
    }

    @Test
    public void testScanMatchesReference() {
        Random rnd = new Random(7);
        for (int m : new int[]{4, 16, 48, 96}) {
            int blockCount = 5;
            byte[] blocks = new byte[blockCount * m * 64];
            rnd.nextBytes(blocks);
            byte[] lut = new byte[m * 256];
            rnd.nextBytes(lut);
            short[] expected = new short[blockCount * 64];
            reference(blocks, blockCount, m, lut, expected);

            short[] provider = new short[blockCount * 64];
            VectorUtil.pqScanBlockedU8(blocks, blockCount, m, lut, provider);
            short[] kernel = new short[blockCount * 64];
            PqScanKernel.scan(blocks, blockCount, m, lut, kernel);
            for (int i = 0; i < expected.length; i++) {
                assertEquals("provider path, m=" + m + " code " + i, expected[i], provider[i]);
                assertEquals("kernel (native=" + PqScanKernel.nativeAvailable() + "), m=" + m + " code " + i, expected[i], kernel[i]);
            }
        }
    }
}
