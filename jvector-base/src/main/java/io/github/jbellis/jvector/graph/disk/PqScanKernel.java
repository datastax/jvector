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

import io.github.jbellis.jvector.vector.VectorUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * EXPERIMENT: entry point for the blocked PQ scan. Uses the native Highway kernel directly when
 * the native library is available (independent of the vectorization provider in use, so the rest
 * of the compactor can stay on the Panama path), otherwise the provider's implementation.
 */
final class PqScanKernel {
    private static final Logger log = LoggerFactory.getLogger(PqScanKernel.class);
    private static final boolean USE_NATIVE = Boolean.parseBoolean(System.getProperty("jvector.compaction.cellNativeScan", "true"));
    private static final MethodHandle NATIVE = USE_NATIVE ? lookupNative() : null;

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
            log.info("Cell join: native blocked PQ scan kernel unavailable ({}); using the provider path", t.toString());
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
        VectorUtil.pqScanBlockedU8(blocks, blockCount, subspaceCount, lut, out);
    }
}
