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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.VectorCursor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Worker-local bounded input ranges; a borrowed vector is consumed before advancing. */
final class FinalWriteVectors implements AutoCloseable {
    private final RandomAccessVectorValues source;
    private final ConcurrentLinkedQueue<Range> ranges =
            new ConcurrentLinkedQueue<>();
    private final ThreadLocal<Range> local = ThreadLocal.withInitial(() -> {
        var range = new Range();
        ranges.add(range);
        return range;
    });

    FinalWriteVectors(RandomAccessVectorValues source) { this.source = source; }

    VectorFloat<?> get(int ordinal) {
        Objects.checkIndex(ordinal, source.size());
        var range = local.get();
        if (range.cursor == null || ordinal != range.ordinal) {
            if (range.cursor == null || ordinal != range.ordinal + 1 || ordinal == range.end) {
                range.close();
                range.end = (int) Math.min(source.size(), (long) ordinal + 64);
                range.cursor = VectorAccess.openRange(source, ordinal, range.end);
            }
            if (!range.cursor.next() || range.cursor.ordinal() != ordinal)
                throw new IllegalStateException("Incomplete final-write input range");
            range.ordinal = ordinal;
        }
        return range.cursor.vector();
    }

    @Override public void close() {
        // writer.write joins its workers before cursors and their read-ahead are released.
        RuntimeException failure = null;
        try {
            for (var range : ranges) {
                try { range.close(); }
                catch (RuntimeException e) {
                    if (failure == null) failure = e;
                    else if (failure != e) failure.addSuppressed(e);
                }
            }
        } finally { ranges.clear(); local.remove(); }
        if (failure != null) throw failure;
    }

    private static final class Range {
        VectorCursor cursor;
        int ordinal = -1, end;
        void close() {
            if (cursor != null) {
                try { cursor.close(); }
                finally { cursor = null; ordinal = -1; }
            }
        }
    }
}
