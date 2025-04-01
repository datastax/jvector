/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.annotations.VisibleForTesting;

/**
 * A min heap that stores longs; a primitive priority queue that like all priority queues maintains
 * a partial ordering of its elements such that the leastbo element can always be found in constant
 * time. Push()'s and pop()'s require log(size). {@link #push(long)} may either grow the heap or
 * replace the worst element, depending on the subclass implementation.
 * <p>
 * The heap is a min heap, meaning that the top element is the lowest value.
 */
public abstract class AbstractLongHeap {

    protected long[] heap;
    protected int size = 0;

    /**
     * Create an empty heap with the configured initial size.
     *
     * @param initialSize the initial size of the heap
     */
    public AbstractLongHeap(int initialSize) {
        final int heapSize;
        if (initialSize < 1 || initialSize >= ArrayUtil.MAX_ARRAY_LENGTH) {
            // Throw exception to prevent confusing OOME:
            throw new IllegalArgumentException(
                    "initialSize must be > 0 and < " + (ArrayUtil.MAX_ARRAY_LENGTH - 1) + "; got: " + initialSize);
        }
        // NOTE: we add +1 because all access to heap is 1-based not 0-based.  heap[0] is unused.
        heapSize = initialSize + 1;
        this.heap = new long[heapSize];
    }

    /**
     * Adds a value to an LongHeap in log(size) time.
     *
     * @return true if the new value was added. (A fixed-size heap will not add the new value
     * if it is full, and the new value is worse than the existing ones.)
     */
    public abstract boolean push(long element);

    protected long add(long element) {
        size++;
        if (size == heap.length) {
            heap = ArrayUtil.grow(heap, (size * 3 + 1) / 2);
        }
        heap[size] = element;
        upHeap(size);
        return heap[1];
    }

    /**
     * Returns the least element of the LongHeap in constant time. It is up to the caller to verify
     * that the heap is not empty; no checking is done, and if no elements have been added, 0 is
     * returned.
     */
    public final long top() {
        return heap[1];
    }

    /**
     * Removes and returns the least element of the PriorityQueue in log(size) time.
     *
     * @throws IllegalStateException if the LongHeap is empty.
     */
    public final long pop() {
        if (size > 0) {
            long result = heap[1]; // save first value
            heap[1] = heap[size]; // move last to first
            size--;
            downHeap(1); // adjust heap
            return result;
        } else {
            throw new IllegalStateException("The heap is empty");
        }
    }

    /** Returns the number of elements currently stored in the PriorityQueue. */
    public final int size() {
        return size;
    }

    /** Removes all entries from the PriorityQueue. */
    public final void clear() {
        size = 0;
    }

    protected void upHeap(int origPos) {
        int i = origPos;
        long value = heap[i]; // save bottom value
        int j = i >>> 1;
        while (j > 0 && value < heap[j]) {
            heap[i] = heap[j]; // shift parents down
            i = j;
            j = j >>> 1;
        }
        heap[i] = value; // install saved value
    }

    protected void downHeap(int i) {
        long value = heap[i]; // save top value
        int j = i << 1; // find smaller child
        int k = j + 1;
        if (k <= size && heap[k] < heap[j]) {
            j = k;
        }
        while (j <= size && heap[j] < value) {
            heap[i] = heap[j]; // shift up child
            i = j;
            j = i << 1;
            k = j + 1;
            if (k <= size && heap[k] < heap[j]) {
                j = k;
            }
        }
        heap[i] = value; // install saved value
    }

    /**
     * Return the element at the ith location in the heap array. Use for iterating over elements when
     * the order doesn't matter. Note that the valid arguments range from [1, size].
     */
    public long get(int i) {
        return heap[i];
    }

    @VisibleForTesting
    long[] getHeapArray() {
        return heap;
    }

    /**
     * Copies the contents and current size from `other`.  Does NOT copy subclass field like BLH's maxSize
     */
    public void copyFrom(AbstractLongHeap other)
    {
        if (this.heap.length < other.size) {
            this.heap = new long[other.heap.length];
        }
        System.arraycopy(other.heap, 0, this.heap, 0, other.size);
        this.size = other.size;
    }
}
