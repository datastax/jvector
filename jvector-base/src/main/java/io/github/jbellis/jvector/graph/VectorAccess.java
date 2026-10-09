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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** Source-independent vector access with the existing point-access path as a fallback. */
public final class VectorAccess {
    private VectorAccess() {}

    @FunctionalInterface
    public interface VectorConsumer {
        /** The vector is borrowed for this call; copy it before retaining it. */
        void accept(int position, VectorFloat<?> vector);
    }

    /**
     * Open [startInclusive, endExclusive), including for sources that only implement
     * point access. The end ordinal is excluded.
     * Algorithms should use this utility to retain the ordinary-source fallback.
     */
    public static VectorCursor openRange(RandomAccessVectorValues source, int startInclusive, int endExclusive) {
        Objects.checkFromToIndex(startInclusive, endExclusive, source.size());
        if (source instanceof BatchedVectorValues)
            return ((BatchedVectorValues) source).openRange(startInclusive, endExclusive);
        return pointCursor(source, startInclusive, endExclusive - startInclusive, null);
    }

    /** Open a selection in request order, including duplicates. */
    public static VectorCursor openSelection(RandomAccessVectorValues source, int[] ordinals, int offset, int count) {
        Objects.checkFromIndexSize(offset, count, ordinals.length);
        for (int i = offset; i < offset + count; i++) Objects.checkIndex(ordinals[i], source.size());
        if (source instanceof BatchedVectorValues)
            return ((BatchedVectorValues) source).openSelection(ordinals, offset, count);
        return pointCursor(source, offset, count, ordinals);
    }

    private static VectorCursor pointCursor(RandomAccessVectorValues source, int start, int count, int[] ordinals) {
        RandomAccessVectorValues values = source.copy();
        return new VectorCursor() {
            int position = -1;
            VectorFloat<?> current;
            boolean closed;
            public boolean next() {
                if (closed) throw new IllegalStateException("Cursor is closed");
                if (position == count) return false;
                if (++position == count) { current = null; return false; }
                current = values.getVector(ordinals == null ? start + position : ordinals[start + position]);
                return true;
            }
            private void checkPosition() {
                if (closed || position < 0 || position >= count) throw new IllegalStateException("No current vector");
            }
            public int ordinal() { checkPosition(); return ordinals == null ? start + position : ordinals[start + position]; }
            public VectorFloat<?> vector() { checkPosition(); return current; }
            public boolean isValueShared() { return values.isValueShared(); }
            public void close() { closed = true; current = null; }
        };
    }

    /**
     * Materialize a selection, retaining its order and duplicates. Only capable sources
     * use storage-order scheduling; the sample policy and returned training order stay unchanged.
     * Retained vectors are algorithm working memory, separate from the source's I/O budget.
     */
    public static List<VectorFloat<?>> copySelected(RandomAccessVectorValues source, int[] ordinals, ForkJoinPool executor) {
        if (!(source instanceof BatchedVectorValues)) {
            var local = source.threadLocalSupplier();
            return executor.submit(() -> IntStream.of(ordinals).parallel().mapToObj(ordinal -> {
                var values = local.get();
                var v = values.getVector(ordinal);
                return v != null && values.isValueShared() ? v.copy() : v;
            }).collect(Collectors.<VectorFloat<?>>toList())).join();
        }
        // Pack the original position so sorting the reads cannot reorder the training set.
        long[] order = new long[ordinals.length];
        for (int i = 0; i < ordinals.length; i++) {
            Objects.checkIndex(ordinals[i], source.size());
            order[i] = ((long) ordinals[i] << 32) | (i & 0xffffffffL);
        }
        Arrays.sort(order);
        int[] sorted = new int[order.length];
        for (int i = 0; i < order.length; i++) sorted[i] = (int) (order[i] >>> 32);
        VectorFloat<?>[] result = new VectorFloat<?>[ordinals.length];
        int partitions = Math.min(ordinals.length, executor.getParallelism());
        executor.submit(() -> IntStream.range(0, partitions).parallel().forEach(part -> {
            int start = (int) ((long) part * sorted.length / partitions);
            int end = (int) ((long) (part + 1) * sorted.length / partitions);
            try (var cursor = openSelection(source, sorted, start, end - start)) {
                int i = start;
                while (cursor.next()) {
                    var v = cursor.vector();
                    result[(int) order[i++]] = v != null && cursor.isValueShared() ? v.copy() : v;
                }
                if (i != end) throw new IllegalStateException("Incomplete vector selection");
            }
        })).join();
        return Arrays.asList(result);
    }

    /**
     * Consume vectors in parallel, with contiguous request partitions for capable sources.
     * The callback position is the output position, not the mapped input ordinal.
     */
    public static void forEach(RandomAccessVectorValues source, int count, IntUnaryOperator mapping,
                               ForkJoinPool executor, VectorConsumer consumer) {
        if (count < 0) throw new IllegalArgumentException("Negative vector count");
        if (!(source instanceof BatchedVectorValues)) {
            var local = source.threadLocalSupplier();
            executor.submit(() -> IntStream.range(0, count).parallel().forEach(i ->
                    consumer.accept(i, local.get().getVector(mapping.applyAsInt(i))))).join();
            return;
        }
        int partitions = Math.min(count, executor.getParallelism());
        executor.submit(() -> IntStream.range(0, partitions).parallel().forEach(part -> {
            int start = (int) ((long) part * count / partitions);
            int end = (int) ((long) (part + 1) * count / partitions);
            int[] ordinals = IntStream.range(start, end).map(mapping).toArray();
            try (var cursor = openSelection(source, ordinals, 0, ordinals.length)) {
                int i = start;
                while (cursor.next()) consumer.accept(i++, cursor.vector());
                if (i != end) throw new IllegalStateException("Incomplete vector range");
            }
        })).join();
    }
}
