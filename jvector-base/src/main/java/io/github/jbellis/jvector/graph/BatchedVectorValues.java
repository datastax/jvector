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

/**
 * Optional range and selection access for a RandomAccessVectorValues source. Algorithms
 * describe the vectors they need; the source owns batching, I/O concurrency and read-ahead.
 * Implementations must share their I/O memory limit across all copies and cursors.
 */
public interface BatchedVectorValues extends RandomAccessVectorValues {
    /**
     * Open a cursor for [startInclusive, endExclusive), using an exclusive end. Invalid ranges
     * fail before scheduling I/O; equal endpoints produce an empty cursor. Opening
     * may start asynchronous reads, but must not materialize the entire range.
     * Close the cursor with try-with-resources before closing its source.
     */
    VectorCursor openRange(int startInclusive, int endExclusive);

    /**
     * Open a cursor for the specified slice, preserving request order and duplicates.
     * The caller must not modify the ordinal array until the cursor closes. Invalid
     * ordinals fail before scheduling I/O. Memory is bounded independently of count.
     */
    VectorCursor openSelection(int[] ordinals, int offset, int count);
}
