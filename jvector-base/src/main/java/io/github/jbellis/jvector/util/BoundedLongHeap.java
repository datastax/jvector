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
import java.util.PrimitiveIterator;

/**
 * An AbstractLongHeap with an adjustable maximum size.
 */
public class BoundedLongHeap extends AbstractLongHeap {

    private int maxSize;

    /**
     * Create an empty Heap of the configured initial size.
     *
     * @param maxSize the maximum size of the heap
     */
    public BoundedLongHeap(int maxSize) {
        this(maxSize, maxSize);
    }

    public BoundedLongHeap(int initialSize, int maxSize) {
        super(initialSize);
        this.maxSize = maxSize;
    }

    public void setMaxSize(int maxSize) {
        if (size > maxSize) {
            throw new IllegalArgumentException("Cannot set maxSize smaller than current size");
        }
        this.maxSize = maxSize;
    }

    @Override
    public boolean push(long value) {
        if (size >= maxSize) {
            if (value < heap[1]) {
                return false;
            }
            updateTop(value);
            return true;
        }
        add(value);
        return true;
    }

    @Override
    public void pushAll(PrimitiveIterator.OfLong elements, int elementsSize)
    {
        if (elementsSize + size >= maxSize) {
            throw new IllegalArgumentException("Cannot add more elements than maxSize");
        }
        addAll(elements, elementsSize);
    }

    /**
     * Replace the top of the heap with {@code newTop}, and enforce the heap invariant.
     * Should be called when the top value changes.
     * Still log(n) worst case, but it's at least twice as fast to
     *
     * <pre class="prettyprint">
     * pq.updateTop(value);
     * </pre>
     * <p>
     * instead of
     *
     * <pre class="prettyprint">
     * pq.pop();
     * pq.push(value);
     * </pre>
     * <p>
     * Calling this method on an empty BoundedLongHeap has no visible effect.
     *
     * @param value the new element that is less than the current top.
     * @return the new 'top' element after shuffling the heap.
     */
    @VisibleForTesting
    long updateTop(long value) {
        heap[1] = value;
        downHeap(1);
        return heap[1];
    }
}
