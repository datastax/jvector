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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Entry point for the blocked PQ scan: 64 codes per block, codes subspace-major within the block,
 * an 8-bit per-subspace table, 16-bit sums out. Uses the native Highway kernel directly when the
 * native library is available (independent of the vectorization provider in use, so the rest of
 * the compactor can stay on the Panama path), otherwise a plain Java loop.
 */
final class PqScanKernel {
    private static final Logger log = LoggerFactory.getLogger(PqScanKernel.class);
    private static final MethodHandle NATIVE = lookupNative();

    private PqScanKernel() {}

    private static MethodHandle lookupNative() {
        try {
            Class<?> ops = Class.forName("io.github.jbellis.jvector.vector.cnative.NativeScanOps");
            MethodHandle mh = MethodHandles.publicLookup().findStatic(ops, "scanArrays",
                    MethodType.methodType(void.class, byte[].class, int.class, int.class, byte[].class, short[].class));
            // probe once so a missing library or symbol fails here, not in the hot loop
            byte[] blocks = new byte[64 * 2];
            byte[] lut = new byte[2 * 256];
            short[] out = new short[64];
            mh.invokeExact(blocks, 1, 2, lut, out);
            log.info("Cell join: native blocked PQ scan kernel available");
            return mh;
        } catch (Throwable t) {
            log.info("Cell join: native blocked PQ scan kernel unavailable ({}); using the Java loop", t.toString());
            return null;
        }
    }

    static boolean nativeAvailable() {
        return NATIVE != null;
    }

    static void scan(byte[] blocks, int blockCount, int subspaceCount, byte[] lut, short[] out) {
        if (NATIVE != null) {
            try {
                NATIVE.invokeExact(blocks, blockCount, subspaceCount, lut, out);
                return;
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }
        scanJava(blocks, blockCount, subspaceCount, lut, out);
    }

    static void scanJava(byte[] blocks, int blockCount, int subspaceCount, byte[] lut, short[] out) {
        int[] acc = new int[64];
        for (int b = 0; b < blockCount; b++) {
            int blk = b * subspaceCount * 64;
            java.util.Arrays.fill(acc, 0);
            for (int m = 0; m < subspaceCount; m++) {
                int row = blk + m * 64;
                int l = m * 256;
                for (int i = 0; i < 64; i++) {
                    acc[i] += lut[l + (blocks[row + i] & 0xFF)] & 0xFF;
                }
            }
            int o = b * 64;
            for (int i = 0; i < 64; i++) {
                out[o + i] = (short) acc[i];
            }
        }
    }
}
