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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/** Packs canonical ASH codes into the existing fused neighborhood layout. */
public final class FusedASHBlockPacker {
    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final ThreadLocal<long[]> WORK = ThreadLocal.withInitial(() -> new long[8]);

    private FusedASHBlockPacker() {}

    /**
     * Packs up to {@code blockSize} vectors from {@code vectors[first]} into a
     * zero-initialized block. Supported block sizes are 8, 16 and 32.
     * Header encoding and lane order are identical to packQuantizedVector.
     */
    public static void pack(byte[] destination, int offset,
                            AsymmetricHashing.QuantizedVector[] vectors, int first, int count,
                            int dimension, int bits, int blockSize) {
        if (bits == 1) {
            for (int lane = 0; lane < count; lane++) {
                FusedASHLayout.packQuantizedVector(destination, offset, lane,
                        vectors[first + lane], dimension, bits, blockSize);
            }
            return;
        }
        int groups = FusedASHLayout.codeGroups(dimension, bits);
        int fullBytes = groups / 2;
        for (int lane = 0; lane < count; lane++) {
            byte[] code = vectors[first + lane].extraBits;
            if (code == null || code.length < (groups + 1) / 2) {
                throw new IllegalArgumentException("Missing or truncated ASH code");
            }
        }

        long[] work = WORK.get();
        int column = 0;
        // Transpose an 8-neighbor by 8-byte tile using three swap stages.
        // Explicit little-endian loads/stores preserve the format on either endianness.
        for (; column + 8 <= fullBytes; column += 8) {
            for (int laneBase = 0; laneBase < blockSize; laneBase += 8) {
                for (int i = 0; i < 8; i++) {
                    work[i] = laneBase + i < count
                            ? (long) LONGS.get(vectors[first + laneBase + i].extraBits, column)
                            : 0;
                }
                for (int step = 1; step <= 4; step *= 2) {
                    int shift = step * 8;
                    long mask = step == 1 ? 0x00ff00ff00ff00ffL
                            : step == 2 ? 0x0000ffff0000ffffL : 0x00000000ffffffffL;
                    for (int tile = 0; tile < 8; tile += 2 * step) {
                        for (int j = 0; j < step; j++) {
                            long a = work[tile + j];
                            long b = work[tile + j + step];
                            work[tile + j] = (a & mask) | ((b & mask) << shift);
                            work[tile + j + step] = ((a >>> shift) & mask) | (b & ~mask);
                        }
                    }
                }
                for (int i = 0; i < 8; i++) {
                    LONGS.set(destination, offset + (column + i) * blockSize + laneBase, work[i]);
                }
            }
        }
        for (; column < fullBytes; column++) {
            for (int lane = 0; lane < count; lane++) {
                destination[offset + column * blockSize + lane] = vectors[first + lane].extraBits[column];
            }
        }
        // An odd group count leaves a high padding nibble; preserve it.
        if ((groups & 1) != 0) {
            for (int lane = 0; lane < count; lane++) {
                int at = offset + fullBytes * blockSize + lane;
                destination[at] = (byte) ((destination[at] & 0xf0)
                        | (vectors[first + lane].extraBits[fullBytes] & 15));
            }
        }
        for (int lane = 0; lane < count; lane++) {
            var vector = vectors[first + lane];
            FusedASHLayout.writeLaneHeader(destination, offset, lane, dimension, bits, blockSize,
                    vector.scale, vector.offset, vector.landmark);
        }
    }
}
