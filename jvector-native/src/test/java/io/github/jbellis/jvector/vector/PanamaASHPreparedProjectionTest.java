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

import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** The Panama prepared-query projection dot matches the scalar decoder, wherever the code sits in its array. */
class PanamaASHPreparedProjectionTest {
    @Test void preparedDotMatchesScalarDecoder() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshProjectionScoring()) return;
        Random random = new Random(7177);
        for (int bits : new int[] {2, 4}) {
            int perByte = 8 / bits;
            for (int dims : new int[] {1, 2, 3, 5, 7, 15, 16, 17, 31, 63, 64, 65, 127, 190, 256, 380, 384, 512, 1536, 3073}) {
                int bytes = (dims + perByte - 1) / perByte;
                float[] query = new float[dims];
                for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
                float[] prepared = panama.ashPrepareProjectionQuery(query, dims, bits);
                // Earlier JDKs keep the gather kernel.
                assertEquals(Runtime.version().feature() >= 25, prepared != null);
                if (prepared == null) continue;
                // Codes in the middle of a page and ending exactly at its end (the tail-copy path).
                for (int lead : new int[] {0, 3}) {
                    for (int trail : new int[] {0, 40}) {
                        byte[] page = new byte[lead + bytes + trail];
                        random.nextBytes(page);
                        double expected = 0, magnitude = 0;
                        for (int i = 0; i < dims; i++) {
                            int field = (page[lead + i / perByte] & 255) >>> ((i % perByte) * bits);
                            double m = (field & ((1 << (bits - 1)) - 1)) + .5;
                            double v = (field & (1 << (bits - 1))) == 0 ? -m : m;
                            expected += query[i] * v;
                            magnitude += Math.abs(query[i] * v);
                        }
                        assertEquals(expected, panama.ashProjectionDotPrepared(prepared, page, lead, dims, bits),
                                1e-6 + magnitude * 2e-6, "bits " + bits + " dims " + dims + " lead " + lead + " trail " + trail);
                    }
                }
            }
        }
    }
}
