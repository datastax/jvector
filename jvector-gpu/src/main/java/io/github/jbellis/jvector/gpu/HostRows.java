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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Vectors as "rows": one contiguous block of native memory holding {@code n x dim} float32 values, row after row.
 * This is the layout cuVS and the GPU code read. Plain Java: no GPU code.
 */
final class HostRows {

    /** C's {@code malloc} and {@code free}, for {@link #allocate}. */
    private static final MethodHandle MALLOC;
    private static final MethodHandle FREE;

    static {
        Linker linker = Linker.nativeLinker();
        MALLOC = linker.downcallHandle(linker.defaultLookup().find("malloc").orElseThrow(), FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        FREE = linker.downcallHandle(linker.defaultLookup().find("free").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    }

    private HostRows() {
    }

    /**
     * Native memory that is freed when {@code arena} closes, without the zero-filling of {@link Arena#allocate}:
     * the callers overwrite every byte, in parallel, and zero-filling gigabytes on one thread would cost about as
     * much as the copy itself (1.2 s for 6 GB).
     */
    static MemorySegment allocate(long bytes, Arena arena) {
        if (bytes == 0) {
            return arena.allocate(0);
        }
        try {
            MemorySegment memory = (MemorySegment) MALLOC.invokeExact(bytes);
            if (memory.equals(MemorySegment.NULL)) {
                throw new OutOfMemoryError("malloc of " + bytes + " bytes failed");
            }
            return memory.reinterpret(bytes, arena, HostRows::free);
        } catch (Error | RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    private static void free(MemorySegment memory) {
        try {
            FREE.invokeExact(memory);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /** Copies all vectors into rows, scaled to unit length if {@code normalise}. */
    static MemorySegment copy(RandomAccessVectorValues vectors, Arena arena, boolean normalise) {
        int n = vectors.size();
        int dim = vectors.dimension();
        MemorySegment rows = allocate((long) n * dim * Float.BYTES, arena);
        var supplier = vectors.threadLocalSupplier();
        ThreadLocal<float[]> buffers = ThreadLocal.withInitial(() -> new float[dim]);
        IntStream.range(0, n).parallel().forEach(i -> {
            VectorFloat<?> v = supplier.get().getVector(i);
            if (!normalise) {
                copyRow(v, rows, i, dim);
                return;
            }
            float[] row = buffers.get();
            read(v, row, dim);
            float scale = (float) (1.0 / Math.sqrt(VectorUtil.dotProduct(v, v)));
            for (int d = 0; d < dim; d++) {
                row[d] *= scale;
            }
            MemorySegment.copy(row, 0, rows, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
        });
        return rows;
    }

    /** Copies the rows of the given nodes, in that order, into new rows. */
    static MemorySegment gather(MemorySegment rows, int[] nodes, int dim, Arena arena) {
        long rowBytes = (long) dim * Float.BYTES;
        MemorySegment gathered = allocate(nodes.length * rowBytes, arena);
        IntStream.range(0, nodes.length).parallel()
                .forEach(x -> MemorySegment.copy(rows, nodes[x] * rowBytes, gathered, x * rowBytes, rowBytes));
        return gathered;
    }

    /** Copies one vector into row {@code i}, as one block copy when the vector's storage allows. */
    static void copyRow(VectorFloat<?> v, MemorySegment rows, long i, int dim) {
        long at = i * dim * Float.BYTES;
        Object storage = v.get();
        if (storage instanceof float[] array) {
            MemorySegment.copy(array, v.offset(0), rows, ValueLayout.JAVA_FLOAT, at, dim);
        } else if (storage instanceof MemorySegment segment) {
            MemorySegment.copy(segment, (long) v.offset(0) * Float.BYTES, rows, at, (long) dim * Float.BYTES);
        } else {
            for (int d = 0; d < dim; d++) {
                rows.set(ValueLayout.JAVA_FLOAT, at + (long) d * Float.BYTES, v.get(d));
            }
        }
    }

    /** Copies one vector into {@code row}. */
    private static void read(VectorFloat<?> v, float[] row, int dim) {
        Object storage = v.get();
        if (storage instanceof float[] array) {
            System.arraycopy(array, v.offset(0), row, 0, dim);
        } else if (storage instanceof MemorySegment segment) {
            MemorySegment.copy(segment, ValueLayout.JAVA_FLOAT, (long) v.offset(0) * Float.BYTES, row, 0, dim);
        } else {
            for (int d = 0; d < dim; d++) {
                row[d] = v.get(d);
            }
        }
    }
}
