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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Checked optional Highway bindings for exact symmetric ASH integer arithmetic.
 * Graph pair comparisons use one bounded critical call. Complete interleaved
 * ranges use one LUT call; larger legal inputs use ordinary FFM with temporary
 * native buffers. Neither kernel allocates, retains pointers, blocks or calls Java.
 * Missing symbols preserve the provider's existing Java implementations.
 */
final class NativeASHSymmetricKernels {
    private static final int MAX_CRITICAL_WORK = 1 << 16;
    private static final boolean TRACE = Boolean.getBoolean("jextract.trace.downcalls");
    private static final Binding PAIR = new Binding("ash_symmetric_dot",
            FunctionDescriptor.of(JAVA_FLOAT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final Binding LUT = new Binding("ash_symmetric_lut_i16",
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS));

    private NativeASHSymmetricKernels() {}

    static boolean hasPairKernel() { return PAIR.available(); }
    static boolean hasLutKernel() { return LUT.available(); }

    static float dot(long[] aBits, byte[] aCodes, long[] bBits, byte[] bCodes, int dimensions, int bits) {
        if (dimensions < 0 || (bits != 1 && bits != 2 && bits != 4))
            throw new IllegalArgumentException("Symmetric ASH requires nonnegative dimensions and bits in {1,2,4}");
        if (!PAIR.available()) throw new UnsupportedOperationException("Native symmetric pair kernel unavailable");
        MemorySegment a, b;
        long bytes;
        if (bits == 1) {
            long words = (dimensions + 63L) / 64;
            if (words > aBits.length || words > bBits.length) throw new IndexOutOfBoundsException("ASH bit arrays");
            a = MemorySegment.ofArray(aBits); b = MemorySegment.ofArray(bBits); bytes = words * Long.BYTES;
        } else {
            bytes = ((long) dimensions * bits + 7) / 8;
            if (bytes > aCodes.length || bytes > bCodes.length) throw new IndexOutOfBoundsException("ASH code arrays");
            a = MemorySegment.ofArray(aCodes); b = MemorySegment.ofArray(bCodes);
        }
        if (dimensions <= MAX_CRITICAL_WORK) return invokePair(PAIR.critical, a, b, dimensions, bits);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment na = arena.allocate(bytes, Long.BYTES), nb = arena.allocate(bytes, Long.BYTES);
            MemorySegment.copy(a, 0, na, 0, bytes); MemorySegment.copy(b, 0, nb, 0, bytes);
            return invokePair(PAIR.ordinary, na, nb, dimensions, bits);
        }
    }

    static void scoreLut(byte[] codes, int groups, int stride, int lane, int count,
                         short[] lut, float[] out, int outOffset) {
        Objects.requireNonNull(codes, "codes"); Objects.requireNonNull(lut, "lut"); Objects.requireNonNull(out, "out");
        if (!LUT.available()) throw new UnsupportedOperationException("Native symmetric LUT kernel unavailable");
        if (groups < 0 || stride <= 0) throw new IllegalArgumentException("ASH groups/stride");
        Objects.checkFromIndexSize(lane, count, stride);
        Objects.checkFromIndexSize(outOffset, count, out.length);
        long codeBytes = ((groups + 1L) / 2) * stride, lutBytes = (long) groups * 32 * Short.BYTES;
        if (codeBytes > codes.length || lutBytes / Short.BYTES > lut.length)
            throw new IndexOutOfBoundsException("ASH LUT/code arrays");
        if (count == 0) return;
        MemorySegment source = MemorySegment.ofArray(codes), table = MemorySegment.ofArray(lut);
        MemorySegment result = MemorySegment.ofArray(out).asSlice((long) outOffset * Float.BYTES);
        // Charge padded SIMD lanes as well as active lanes (native width is capped at 16).
        long vectorSlots = ((count + 15L) / 16) * 16;
        if ((long) groups * vectorSlots <= MAX_CRITICAL_WORK) {
            invokeLut(LUT.critical, source, table, groups, stride, lane, count, result);
        } else {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nc = arena.allocate(Math.max(1, codeBytes));
                MemorySegment nl = arena.allocate(Math.max(Short.BYTES, lutBytes), Short.BYTES);
                MemorySegment no = arena.allocate((long) count * Float.BYTES, Float.BYTES);
                MemorySegment.copy(source, 0, nc, 0, codeBytes); MemorySegment.copy(table, 0, nl, 0, lutBytes);
                invokeLut(LUT.ordinary, nc, nl, groups, stride, lane, count, no);
                MemorySegment.copy(no, 0, result, 0, (long) count * Float.BYTES);
            }
        }
    }

    private static float invokePair(MethodHandle handle, MemorySegment a, MemorySegment b, int dimensions, int bits) {
        if (TRACE) System.out.printf("%s(%d, %d)%n", PAIR.symbol, dimensions, bits);
        try {
            return (float) handle.invokeExact(a, b, dimensions, bits);
        } catch (RuntimeException | Error error) { throw error;
        } catch (Throwable error) { throw new AssertionError("Unexpected exception from " + PAIR.symbol, error); }
    }

    private static void invokeLut(MethodHandle handle, MemorySegment codes, MemorySegment lut,
                                   int groups, int stride, int lane, int count, MemorySegment out) {
        if (TRACE) System.out.printf("%s(%d, %d)%n", LUT.symbol, groups, count);
        try {
            handle.invokeExact(codes, lut, groups, stride, lane, count, out);
        } catch (RuntimeException | Error error) { throw error;
        } catch (Throwable error) { throw new AssertionError("Unexpected exception from " + LUT.symbol, error); }
    }

    private static final class Binding {
        final String symbol;
        final MethodHandle critical;
        final MethodHandle ordinary;

        Binding(String symbol, FunctionDescriptor descriptor) {
            this.symbol = symbol;
            MemorySegment address = SymbolLookup.loaderLookup().find(symbol).orElse(null);
            critical = address == null ? null : Linker.nativeLinker().downcallHandle(
                    address, descriptor, Linker.Option.critical(true));
            ordinary = address == null ? null : Linker.nativeLinker().downcallHandle(address, descriptor);
        }

        boolean available() { return critical != null; }
    }
}
