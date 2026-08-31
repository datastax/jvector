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
package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.NodesIterator;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.IntStream;

/**
 * A map (but not a Map) of int -> T where the int keys are dense-ish and start at zero,
 * but the size of the map is not known in advance.  This provides fast, concurrent
 * updates and minimizes contention when the map is resized.
 * <p>
 * "Dense-ish" means that space is allocated for all keys from 0 to the highest key, but
 * it is valid to have gaps in the keys.  The value associated with "gap" keys is null.
 */
public class DenseIntMap<T> implements IntMap<T> {
    // locking strategy:
    // - writelock to resize the array
    // - readlock to update the array with put or remove
    // - no lock to read the array, volatile is enough
    private final ReadWriteLock rwl = new ReentrantReadWriteLock();
    private volatile AtomicReferenceArray<T> objects;
    private final AtomicInteger size;

    public DenseIntMap(int initialCapacity) {
        objects = new AtomicReferenceArray<>(initialCapacity);
        size = new AtomicInteger();
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }

        ensureCapacity(key);
        rwl.readLock().lock();
        try {
            var success = objects.compareAndSet(key, existing, value);
            var isInsert = success && existing == null;
            if (isInsert) {
                size.incrementAndGet();
            }
            return success;
        } finally {
            rwl.readLock().unlock();
        }
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T get(int key) {
        if (key >= objects.length()) {
            return null;
        }

        return objects.get(key);
    }

    private void ensureCapacity(int node) {
        if (node < objects.length()) {
            return;
        }

        rwl.writeLock().lock();
        try {
            var oldArray = objects;
            if (node >= oldArray.length()) {
                int newSize = ArrayUtil.oversize(node + 1, RamUsageEstimator.NUM_BYTES_OBJECT_REF);
                var newArray = new AtomicReferenceArray<T>(newSize);
                for (int i = 0; i < oldArray.length(); i++) {
                    newArray.set(i, oldArray.get(i));
                }
                objects = newArray;
            }
        } finally {
            rwl.writeLock().unlock();
        }
    }

    @Override
    public T remove(int key) {
        if (key >= objects.length()) {
            return null;
        }
        var old = objects.get(key);
        if (old == null) {
            return null;
        }

        rwl.readLock().lock();
        try {
            if (objects.compareAndSet(key, old, null)) {
                size.decrementAndGet();
                return old;
            } else {
                return null;
            }
        } finally {
            rwl.readLock().unlock();
        }
    }

    @Override
    public boolean containsKey(int key) {
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        var ref = objects;
        for (int i = 0; i < ref.length(); i++) {
            var value = get(i);
            if (value != null) {
                consumer.consume(i, value);
            }
        }
    }
}
