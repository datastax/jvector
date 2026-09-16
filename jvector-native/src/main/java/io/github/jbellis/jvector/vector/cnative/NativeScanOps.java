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

package io.github.jbellis.jvector.vector.cnative;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

/** Hand-written binding for the blocked PQ scan kernel (not part of the jextract output). */
public final class NativeScanOps {
    private NativeScanOps() {}

    private static final FunctionDescriptor DESC = FunctionDescriptor.ofVoid(
            NativeSimdOps.C_POINTER,   // blocks
            NativeSimdOps.C_LONG,      // blockCount (size_t)
            NativeSimdOps.C_INT,       // subspaceCount
            NativeSimdOps.C_POINTER,   // lut
            NativeSimdOps.C_POINTER);  // out
    private static final MethodHandle HANDLE;
    static {
        // The provider loads the library in its constructor; a direct caller under another
        // provider has to load it here before the symbol can resolve.
        if (!LibraryLoader.loadJvector()) {
            throw new UnsatisfiedLinkError("libjvector not found");
        }
        HANDLE = Linker.nativeLinker().downcallHandle(
                NativeSimdOps.findOrThrow("pq_scan_blocked_u8"), DESC, Linker.Option.critical(true));
    }

    /** Array entry point for callers outside the provider machinery (heap segments, critical downcall). */
    public static void scanArrays(byte[] blocks, int blockCount, int subspaceCount, byte[] lut, short[] out) {
        pq_scan_blocked_u8(MemorySegment.ofArray(blocks), blockCount, subspaceCount, MemorySegment.ofArray(lut), MemorySegment.ofArray(out));
    }

    public static void pq_scan_blocked_u8(MemorySegment blocks, long blockCount, int subspaceCount, MemorySegment lut, MemorySegment out) {
        try {
            HANDLE.invokeExact(blocks, blockCount, subspaceCount, lut, out);
        } catch (Throwable t) {
            throw new AssertionError("should not reach here", t);
        }
    }
}
