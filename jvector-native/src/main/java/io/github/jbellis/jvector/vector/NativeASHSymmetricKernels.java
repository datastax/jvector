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
 * <p>
 * JIT notes. The pair kernel runs billions of times per graph build, so:
 * <ul>
 * <li>Downcall handles are {@code static final} fields. HotSpot trusts only static
 * finals (not instance finals) as constants, so {@code invokeExact} links straight to
 * the downcall stub and inlines it. A non-constant handle goes through the generic
 * invoker ({@code Invokers.checkCustomized}) and the heap segments escape into it.</li>
 * <li>Each heap {@link MemorySegment} is created in the same small method that passes
 * it to {@code invokeExact}, so once the stub inlines, escape analysis can
 * scalar-replace the segment and its session instead of allocating them per call.</li>
 * <li>Tracing, exception construction and the large-input {@link Arena} path live in
 * separate cold methods, keeping hot-method bytecode small enough to inline.</li>
 * </ul>
 */
final class NativeASHSymmetricKernels {
    private static final int MAX_CRITICAL_WORK = 1 << 16;
    private static final boolean TRACE = Boolean.getBoolean("jextract.trace.downcalls");
    private static final String PAIR_SYMBOL = "ash_symmetric_dot";
    private static final String LUT_SYMBOL = "ash_symmetric_lut_i16";
    private static final FunctionDescriptor PAIR_DESCRIPTOR =
            FunctionDescriptor.of(JAVA_FLOAT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT);
    private static final FunctionDescriptor LUT_DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS);
    private static final MemorySegment PAIR_ADDRESS = find(PAIR_SYMBOL);
    private static final MemorySegment LUT_ADDRESS = find(LUT_SYMBOL);
    // Must stay static final: see the class comment.
    private static final MethodHandle PAIR_CRITICAL = downcall(PAIR_ADDRESS, PAIR_DESCRIPTOR, true);
    private static final MethodHandle PAIR_ORDINARY = downcall(PAIR_ADDRESS, PAIR_DESCRIPTOR, false);
    private static final MethodHandle LUT_CRITICAL = downcall(LUT_ADDRESS, LUT_DESCRIPTOR, true);
    private static final MethodHandle LUT_ORDINARY = downcall(LUT_ADDRESS, LUT_DESCRIPTOR, false);

    private NativeASHSymmetricKernels() {}

    static boolean hasPairKernel() { return PAIR_CRITICAL != null; }
    static boolean hasLutKernel() { return LUT_CRITICAL != null; }

    static float dot(long[] aBits, byte[] aCodes, long[] bBits, byte[] bCodes, int dimensions, int bits) {
        if (dimensions < 0 || (bits != 1 && bits != 2 && bits != 4)) throw badPairShape();
        if (PAIR_CRITICAL == null) throw pairUnavailable();
        if (bits == 1) {
            long words = (dimensions + 63L) / 64;
            if (words > aBits.length || words > bBits.length) throw outOfBounds("ASH bit arrays");
            if (TRACE) trace(PAIR_SYMBOL, dimensions, bits);
            return dimensions <= MAX_CRITICAL_WORK
                    ? pairCritical(aBits, bBits, dimensions, bits)
                    : pairOrdinary(MemorySegment.ofArray(aBits), MemorySegment.ofArray(bBits),
                                   words * Long.BYTES, dimensions, bits);
        }
        long bytes = ((long) dimensions * bits + 7) / 8;
        if (bytes > aCodes.length || bytes > bCodes.length) throw outOfBounds("ASH code arrays");
        if (TRACE) trace(PAIR_SYMBOL, dimensions, bits);
        return dimensions <= MAX_CRITICAL_WORK
                ? pairCritical(aCodes, bCodes, dimensions, bits)
                : pairOrdinary(MemorySegment.ofArray(aCodes), MemorySegment.ofArray(bCodes), bytes, dimensions, bits);
    }

    // Hot path: segments are created and consumed here so they don't escape after inlining.
    private static float pairCritical(byte[] a, byte[] b, int dimensions, int bits) {
        try {
            return (float) PAIR_CRITICAL.invokeExact(MemorySegment.ofArray(a), MemorySegment.ofArray(b), dimensions, bits);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(PAIR_SYMBOL, t);
        }
    }

    private static float pairCritical(long[] a, long[] b, int dimensions, int bits) {
        try {
            return (float) PAIR_CRITICAL.invokeExact(MemorySegment.ofArray(a), MemorySegment.ofArray(b), dimensions, bits);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(PAIR_SYMBOL, t);
        }
    }

    // Cold path: inputs too large for a bounded critical call are copied to native memory.
    private static float pairOrdinary(MemorySegment a, MemorySegment b, long bytes, int dimensions, int bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment na = arena.allocate(bytes, Long.BYTES), nb = arena.allocate(bytes, Long.BYTES);
            MemorySegment.copy(a, 0, na, 0, bytes); MemorySegment.copy(b, 0, nb, 0, bytes);
            return (float) PAIR_ORDINARY.invokeExact(na, nb, dimensions, bits);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(PAIR_SYMBOL, t);
        }
    }

    static void scoreLut(byte[] codes, int groups, int stride, int lane, int count,
                         short[] lut, float[] out, int outOffset) {
        Objects.requireNonNull(codes, "codes"); Objects.requireNonNull(lut, "lut"); Objects.requireNonNull(out, "out");
        if (LUT_CRITICAL == null) throw lutUnavailable();
        if (groups < 0 || stride <= 0) throw new IllegalArgumentException("ASH groups/stride");
        Objects.checkFromIndexSize(lane, count, stride);
        Objects.checkFromIndexSize(outOffset, count, out.length);
        long codeBytes = ((groups + 1L) / 2) * stride, lutBytes = (long) groups * 32 * Short.BYTES;
        if (codeBytes > codes.length || lutBytes / Short.BYTES > lut.length) throw outOfBounds("ASH LUT/code arrays");
        if (count == 0) return;
        if (TRACE) trace(LUT_SYMBOL, groups, count);
        // Charge padded SIMD lanes as well as active lanes (native width is capped at 16).
        long vectorSlots = ((count + 15L) / 16) * 16;
        if ((long) groups * vectorSlots <= MAX_CRITICAL_WORK) {
            lutCritical(codes, lut, out, outOffset, groups, stride, lane, count);
        } else {
            lutOrdinary(codes, lut, out, outOffset, groups, stride, lane, count, codeBytes, lutBytes);
        }
    }

    // Hot path: as with pairCritical, segments stay local to the invokeExact call.
    private static void lutCritical(byte[] codes, short[] lut, float[] out, int outOffset,
                                    int groups, int stride, int lane, int count) {
        try {
            LUT_CRITICAL.invokeExact(MemorySegment.ofArray(codes), MemorySegment.ofArray(lut), groups, stride, lane, count,
                                     MemorySegment.ofArray(out).asSlice((long) outOffset * Float.BYTES));
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(LUT_SYMBOL, t);
        }
    }

    private static void lutOrdinary(byte[] codes, short[] lut, float[] out, int outOffset,
                                    int groups, int stride, int lane, int count, long codeBytes, long lutBytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment nc = arena.allocate(Math.max(1, codeBytes));
            MemorySegment nl = arena.allocate(Math.max(Short.BYTES, lutBytes), Short.BYTES);
            MemorySegment no = arena.allocate((long) count * Float.BYTES, Float.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(codes), 0, nc, 0, codeBytes);
            MemorySegment.copy(MemorySegment.ofArray(lut), 0, nl, 0, lutBytes);
            LUT_ORDINARY.invokeExact(nc, nl, groups, stride, lane, count, no);
            MemorySegment.copy(no, 0, MemorySegment.ofArray(out), (long) outOffset * Float.BYTES, (long) count * Float.BYTES);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(LUT_SYMBOL, t);
        }
    }

    // ---- cold helpers (kept out of line so hot methods stay small) ----

    private static MemorySegment find(String symbol) {
        return SymbolLookup.loaderLookup().find(symbol).orElse(null);
    }

    private static MethodHandle downcall(MemorySegment address, FunctionDescriptor descriptor, boolean critical) {
        if (address == null) return null;
        return critical
                ? Linker.nativeLinker().downcallHandle(address, descriptor, Linker.Option.critical(true))
                : Linker.nativeLinker().downcallHandle(address, descriptor);
    }

    private static void trace(String symbol, int x, int y) {
        System.out.printf("%s(%d, %d)%n", symbol, x, y);
    }

    private static IllegalArgumentException badPairShape() {
        return new IllegalArgumentException("Symmetric ASH requires nonnegative dimensions and bits in {1,2,4}");
    }

    private static UnsupportedOperationException pairUnavailable() {
        return new UnsupportedOperationException("Native symmetric pair kernel unavailable");
    }

    private static UnsupportedOperationException lutUnavailable() {
        return new UnsupportedOperationException("Native symmetric LUT kernel unavailable");
    }

    private static IndexOutOfBoundsException outOfBounds(String what) {
        return new IndexOutOfBoundsException(what);
    }

    private static AssertionError unexpected(String symbol, Throwable t) {
        return new AssertionError("Unexpected exception from " + symbol, t);
    }
}
