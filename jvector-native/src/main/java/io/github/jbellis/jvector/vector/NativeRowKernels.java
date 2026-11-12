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

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Checked FFM access to the native row-gather FP32 similarity kernels, which score one query against many rows of a
 * contiguous base in one call.
 *
 * <p>Calls are critical downcalls that read the Java heap directly, so neither the query, the base, the row indexes
 * nor the output is copied. Work per call is bounded; longer batches are split into several calls so the JVM can
 * reach safepoints between them. The kernels neither retain pointers nor call Java.</p>
 *
 * <p>The symbols are optional for compatibility with libraries predating these kernels; when one is absent the
 * caller falls back to scoring rows one at a time.</p>
 */
final class NativeRowKernels {
    // Bound critical-call work independently of CPU model and clock speed: query-row floats read per call. A graph
    // neighborhood of a few dozen rows at a few hundred dimensions is one call.
    static final int MAX_CRITICAL_FLOATS = 1 << 16;

    private static final FunctionDescriptor DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS);
    private static final MethodHandle DOT_PRODUCT = bind("dot_product_rows_f32");
    private static final MethodHandle COSINE = bind("cosine_rows_f32");
    private static final MethodHandle EUCLIDEAN = bind("euclidean_rows_f32");
    private static final FunctionDescriptor PAGED_DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS);
    private static final MethodHandle DOT_PRODUCT_PAGED = bind("dot_product_paged_rows_f32", PAGED_DESCRIPTOR);
    private static final MethodHandle COSINE_PAGED = bind("cosine_paged_rows_f32", PAGED_DESCRIPTOR);
    private static final MethodHandle EUCLIDEAN_PAGED = bind("euclidean_paged_rows_f32", PAGED_DESCRIPTOR);

    private NativeRowKernels() {}

    private static MethodHandle bind(String symbol) {
        return bind(symbol, DESCRIPTOR);
    }

    private static MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        return SymbolLookup.loaderLookup().find(symbol)
                .map(address -> Linker.nativeLinker().downcallHandle(address, descriptor, Linker.Option.critical(true)))
                .orElse(null);
    }

    /**
     * The native addresses of {@code pages}, for the paged kernels, or null when a paged kernel is missing or a page
     * is not native memory (a heap page can move, so its address cannot be kept).
     */
    static long[] pageTable(VectorFloat<?>[] pages) {
        if (DOT_PRODUCT_PAGED == null || COSINE_PAGED == null || EUCLIDEAN_PAGED == null) return null;
        long[] addresses = new long[pages.length];
        for (int p = 0; p < pages.length; p++) {
            if (!(pages[p] instanceof MemorySegmentVectorFloat page)) return null;
            var segment = page.get();
            if (!segment.isNative()) return null;
            addresses[p] = segment.address();
        }
        return addresses;
    }

    /**
     * Scores {@code query} against row {@code rowInPage[i]} of page {@code pageOf[i]} for each {@code i < count}, in
     * one call per bounded chunk. {@code table} must be {@code pageTable(pages)}. Every page index and row range is
     * checked against {@code pages} before any call, and the pages stay reachable until the calls return.
     */
    static void scorePaged(VectorSimilarityKernel kernel, VectorFloat<?> query, VectorFloat<?>[] pages, long[] table,
                           int[] pageOf, int[] rowInPage, int stride, int count, int length, float[] out) {
        var handle = kernel == VectorSimilarityKernel.DOT_PRODUCT ? DOT_PRODUCT_PAGED
                : kernel == VectorSimilarityKernel.COSINE ? COSINE_PAGED : EUCLIDEAN_PAGED;
        if (table.length != pages.length) throw new IllegalArgumentException("page table does not match the pages");
        if (query.length() < length || stride < length || length < 0) throw new IllegalArgumentException("bad row shape");
        java.util.Objects.checkFromIndexSize(0, count, Math.min(pageOf.length, Math.min(rowInPage.length, out.length)));
        for (int i = 0; i < count; i++) {
            int page = pageOf[i];
            java.util.Objects.checkIndex(page, pages.length);
            long end = (long) rowInPage[i] * stride + length;
            if (rowInPage[i] < 0 || end > pages[page].length()) throw new IndexOutOfBoundsException("row outside its page");
        }
        var querySegment = ((MemorySegmentVectorFloat) query).get();
        var tableSegment = MemorySegment.ofArray(table);
        int rowsPerCall = Math.max(1, MAX_CRITICAL_FLOATS / Math.max(1, length));
        try {
            for (int start = 0; start < count; start += rowsPerCall) {
                int n = Math.min(rowsPerCall, count - start);
                invokePaged(handle, querySegment, tableSegment,
                        MemorySegment.ofArray(pageOf).asSlice((long) start * Integer.BYTES),
                        MemorySegment.ofArray(rowInPage).asSlice((long) start * Integer.BYTES),
                        stride, n, length, MemorySegment.ofArray(out).asSlice((long) start * Float.BYTES));
            }
        } finally {
            // The table holds raw addresses; keep the pages, and so their memory, alive until the calls finish.
            java.lang.ref.Reference.reachabilityFence(pages);
        }
    }

    /** Which similarity kernel a paged call uses. */
    enum VectorSimilarityKernel { DOT_PRODUCT, COSINE, EUCLIDEAN }

    private static void invokePaged(MethodHandle kernel, MemorySegment query, MemorySegment pages, MemorySegment pageOf,
                                    MemorySegment rowInPage, int stride, int count, int length, MemorySegment out) {
        try {
            kernel.invokeExact(query, pages, pageOf, rowInPage, (long) stride, count, (long) length, out);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected paged row-scoring invocation failure", failure);
        }
    }

    /** Scores with the native dot-product kernel; returns false, scoring nothing, if the library lacks it. */
    static boolean dotProductRows(VectorFloat<?> query, VectorFloat<?> base, int stride, int[] rows, int count, int length, float[] out) {
        return score(DOT_PRODUCT, query, base, stride, rows, count, length, out);
    }

    /** Scores with the native cosine kernel; returns false, scoring nothing, if the library lacks it. */
    static boolean cosineRows(VectorFloat<?> query, VectorFloat<?> base, int stride, int[] rows, int count, int length, float[] out) {
        return score(COSINE, query, base, stride, rows, count, length, out);
    }

    /** Scores with the native squared-L2 kernel; returns false, scoring nothing, if the library lacks it. */
    static boolean squareDistanceRows(VectorFloat<?> query, VectorFloat<?> base, int stride, int[] rows, int count, int length, float[] out) {
        return score(EUCLIDEAN, query, base, stride, rows, count, length, out);
    }

    /** Validates every pointer the kernel will read or write before any call, then scores in bounded calls. */
    private static boolean score(MethodHandle kernel, VectorFloat<?> query, VectorFloat<?> base, int stride,
                                 int[] rows, int count, int length, float[] out) {
        if (kernel == null) return false;
        VectorUtilSupport.checkRows(query, base, stride, rows, count, length, out);
        var querySegment = ((MemorySegmentVectorFloat) query).get();
        var baseSegment = ((MemorySegmentVectorFloat) base).get();
        int rowsPerCall = Math.max(1, MAX_CRITICAL_FLOATS / Math.max(1, length));
        if (count <= rowsPerCall) {
            // These heap segments are valid only for this synchronous, short leaf call.
            invoke(kernel, querySegment, baseSegment, stride, MemorySegment.ofArray(rows), count, length, MemorySegment.ofArray(out));
            return true;
        }
        var rowSegment = MemorySegment.ofArray(rows);
        var outSegment = MemorySegment.ofArray(out);
        for (int start = 0; start < count; start += rowsPerCall) {
            int n = Math.min(rowsPerCall, count - start);
            invoke(kernel, querySegment, baseSegment, stride, rowSegment.asSlice((long) start * Integer.BYTES),
                    n, length, outSegment.asSlice((long) start * Float.BYTES));
        }
        return true;
    }

    private static void invoke(MethodHandle kernel, MemorySegment query, MemorySegment base, int stride,
                               MemorySegment rows, int count, int length, MemorySegment out) {
        try {
            kernel.invokeExact(query, base, (long) stride, rows, count, (long) length, out);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected row-scoring invocation failure", failure);
        }
    }
}
