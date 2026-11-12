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
    private static final String ROWS_SYMBOL = "ash_symmetric_dot_rows";
    private static final FunctionDescriptor PAIR_DESCRIPTOR =
            FunctionDescriptor.of(JAVA_FLOAT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT);
    private static final FunctionDescriptor LUT_DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS);
    private static final FunctionDescriptor ROWS_DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS);
    private static final MemorySegment PAIR_ADDRESS = find(PAIR_SYMBOL);
    private static final MemorySegment LUT_ADDRESS = find(LUT_SYMBOL);
    // Must stay static final: see the class comment.
    private static final MethodHandle PAIR_CRITICAL = downcall(PAIR_ADDRESS, PAIR_DESCRIPTOR, true);
    private static final MethodHandle PAIR_ORDINARY = downcall(PAIR_ADDRESS, PAIR_DESCRIPTOR, false);
    private static final MethodHandle LUT_CRITICAL = downcall(LUT_ADDRESS, LUT_DESCRIPTOR, true);
    private static final MethodHandle LUT_ORDINARY = downcall(LUT_ADDRESS, LUT_DESCRIPTOR, false);
    private static final MethodHandle ROWS_CRITICAL = downcall(find(ROWS_SYMBOL), ROWS_DESCRIPTOR, true);

    private NativeASHSymmetricKernels() {}

    static boolean hasPairKernel() { return PAIR_CRITICAL != null; }
    static boolean hasLutKernel() { return LUT_CRITICAL != null; }
    static boolean hasRowsKernel() { return ROWS_CRITICAL != null; }

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

    /** 2/4-bit codes at offsets in larger arrays (flat code pages). Same arithmetic as {@link #dot}. */
    static float dotAt(byte[] aCodes, int aOffset, byte[] bCodes, int bOffset, int dimensions, int bits) {
        if (dimensions < 0 || (bits != 2 && bits != 4)) throw badOffsetShape();
        if (PAIR_CRITICAL == null) throw pairUnavailable();
        long bytes = ((long) dimensions * bits + 7) / 8;
        if (aOffset < 0 || bOffset < 0 || aOffset + bytes > aCodes.length || bOffset + bytes > bCodes.length)
            throw outOfBounds("ASH code array ranges");
        if (TRACE) trace(PAIR_SYMBOL, dimensions, bits);
        return dimensions <= MAX_CRITICAL_WORK
                ? pairCriticalAt(aCodes, aOffset, bCodes, bOffset, dimensions, bits)
                : pairOrdinary(MemorySegment.ofArray(aCodes).asSlice(aOffset, bytes),
                               MemorySegment.ofArray(bCodes).asSlice(bOffset, bytes), bytes, dimensions, bits);
    }

    /** 1-bit sign words at offsets in larger arrays (code pages). Same arithmetic as {@link #dot}. */
    static float binaryDotAt(long[] aBits, int aOffset, long[] bBits, int bOffset, int dimensions) {
        if (dimensions < 0) throw badPairShape();
        if (PAIR_CRITICAL == null) throw pairUnavailable();
        long words = (dimensions + 63L) / 64;
        if (aOffset < 0 || bOffset < 0 || aOffset + words > aBits.length || bOffset + words > bBits.length)
            throw outOfBounds("ASH bit array ranges");
        if (TRACE) trace(PAIR_SYMBOL, dimensions, 1);
        return dimensions <= MAX_CRITICAL_WORK
                ? pairCriticalAt(aBits, aOffset, bBits, bOffset, dimensions)
                : pairOrdinary(MemorySegment.ofArray(aBits).asSlice((long) aOffset * Long.BYTES, words * Long.BYTES),
                               MemorySegment.ofArray(bBits).asSlice((long) bOffset * Long.BYTES, words * Long.BYTES),
                               words * Long.BYTES, dimensions, 1);
    }

    /**
     * 2/4-bit query code against {@code count} codes {@code stride} bytes apart: {@code out[i]} equals
     * {@link #dotAt} of the query and row {@code i}. Each critical call scores at most
     * {@code MAX_CRITICAL_WORK} dimensions in total.
     */
    static void dotRows(byte[] query, int queryOffset, byte[] rows, int stride, int count,
                        int dimensions, int bits, float[] out) {
        if (dimensions < 0 || count < 0 || (bits != 2 && bits != 4)) throw badOffsetShape();
        if (ROWS_CRITICAL == null) throw rowsUnavailable();
        long bytes = ((long) dimensions * bits + 7) / 8;
        if (stride < bytes || queryOffset < 0 || queryOffset + bytes > query.length || count > out.length
                || (count > 0 && (long) (count - 1) * stride + bytes > rows.length))
            throw outOfBounds("ASH code row ranges");
        if (count == 0) return;
        if (TRACE) trace(ROWS_SYMBOL, dimensions, count);
        if (dimensions > MAX_CRITICAL_WORK) {
            for (int i = 0; i < count; i++) out[i] = dotAt(query, queryOffset, rows, i * stride, dimensions, bits);
            return;
        }
        int perCall = Math.max(1, MAX_CRITICAL_WORK / Math.max(1, dimensions));
        for (int first = 0; first < count; first += perCall) {
            rowsCritical(query, queryOffset, rows, first * stride, stride, Math.min(perCall, count - first),
                         dimensions, bits, out, first);
        }
    }

    // Hot path: as in pairCritical, the sliced segments stay local to invokeExact.
    private static void rowsCritical(byte[] query, int queryOffset, byte[] rows, int rowsOffset, int stride,
                                     int count, int dimensions, int bits, float[] out, int outOffset) {
        try {
            ROWS_CRITICAL.invokeExact(MemorySegment.ofArray(query).asSlice(queryOffset),
                                      MemorySegment.ofArray(rows).asSlice(rowsOffset), stride, count, dimensions, bits,
                                      MemorySegment.ofArray(out).asSlice((long) outOffset * Float.BYTES));
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(ROWS_SYMBOL, t);
        }
    }

    private static float pairCriticalAt(long[] a, int aOffset, long[] b, int bOffset, int dimensions) {
        try {
            return (float) PAIR_CRITICAL.invokeExact(MemorySegment.ofArray(a).asSlice((long) aOffset * Long.BYTES),
                                                     MemorySegment.ofArray(b).asSlice((long) bOffset * Long.BYTES), dimensions, 1);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(PAIR_SYMBOL, t);
        }
    }

    // Hot path for dotAt: as in pairCritical, the sliced segments stay local to invokeExact.
    private static float pairCriticalAt(byte[] a, int aOffset, byte[] b, int bOffset, int dimensions, int bits) {
        try {
            return (float) PAIR_CRITICAL.invokeExact(MemorySegment.ofArray(a).asSlice(aOffset),
                                                     MemorySegment.ofArray(b).asSlice(bOffset), dimensions, bits);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw unexpected(PAIR_SYMBOL, t);
        }
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

    private static IllegalArgumentException badOffsetShape() {
        return new IllegalArgumentException("Offset symmetric ASH requires nonnegative dimensions and bits in {2,4}");
    }

    private static UnsupportedOperationException pairUnavailable() {
        return new UnsupportedOperationException("Native symmetric pair kernel unavailable");
    }

    private static UnsupportedOperationException rowsUnavailable() {
        return new UnsupportedOperationException("Native symmetric rows kernel unavailable");
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
