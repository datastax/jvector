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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * A growable map of dense-ish int keys to values, like {@link DenseIntMap}, whose updates take no lock.
 * <p>
 * Values live in fixed-size pages. A page, once installed, is never copied or replaced, so a slot
 * has one location for the life of the map. Growing installs new pages and, rarely, replaces the
 * spine (the array of page references) with a larger copy that refers to the same page objects.
 * Because no slot ever moves, a compare-and-set can never be lost to a concurrent resize, which is
 * the race {@link DenseIntMap}'s read-write lock exists to prevent. {@link #compareAndPut} is
 * therefore a single slot CAS, and readers take no locks.
 * <p>
 * Publication: pages are installed with a volatile store into the spine, and a replaced spine is
 * published through a volatile field after it holds every existing page. A thread that observes a
 * value written into a page therefore also observes that page through the spine.
 * <p>
 * As with {@link DenseIntMap}, space is allocated for all keys from 0 to the highest page in use,
 * and gap keys map to null.
 */
public final class PagedIntMap<T> implements IntMap<T> {
    static final int PAGE_BITS = 12;
    static final int PAGE_SIZE = 1 << PAGE_BITS;
    private static final int PAGE_MASK = PAGE_SIZE - 1;

    /** Spine of pages. Entries are installed once, under {@link #growLock}, and never change. */
    private volatile AtomicReferenceArray<AtomicReferenceArray<T>> spine;
    private final Object growLock = new Object();
    private final AtomicInteger size = new AtomicInteger();

    public PagedIntMap(int initialCapacity) {
        if (initialCapacity < 0) throw new IllegalArgumentException("initialCapacity must be nonnegative");
        int pages = Math.max(1, (int) ((initialCapacity + (long) PAGE_SIZE - 1) >>> PAGE_BITS));
        var s = new AtomicReferenceArray<AtomicReferenceArray<T>>(pages);
        for (int p = 0; p < pages; p++) s.set(p, new AtomicReferenceArray<>(PAGE_SIZE));
        spine = s;
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }
        if (key < 0) throw new IndexOutOfBoundsException("negative key " + key);
        boolean success = pageForWrite(key).compareAndSet(key & PAGE_MASK, existing, value);
        if (success && existing == null) size.incrementAndGet();
        return success;
    }

    @Override
    public int size() {
        return size.get();
    }

    @Override
    public T get(int key) {
        var page = pageOrNull(key);
        return page == null ? null : page.get(key & PAGE_MASK);
    }

    @Override
    public T remove(int key) {
        var page = pageOrNull(key);
        if (page == null) return null;
        int slot = key & PAGE_MASK;
        T old = page.get(slot);
        if (old == null) return null;
        if (page.compareAndSet(slot, old, null)) {
            size.decrementAndGet();
            return old;
        }
        return null;
    }

    @Override
    public boolean containsKey(int key) {
        return get(key) != null;
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        var s = spine;
        for (int p = 0; p < s.length(); p++) {
            var page = s.get(p);
            if (page == null) continue;
            int base = p << PAGE_BITS;
            for (int i = 0; i < PAGE_SIZE; i++) {
                T value = page.get(i);
                if (value != null) consumer.consume(base + i, value);
            }
        }
    }

    private AtomicReferenceArray<T> pageOrNull(int key) {
        if (key < 0) throw new IndexOutOfBoundsException("negative key " + key);
        int p = key >>> PAGE_BITS;
        var s = spine;
        return p < s.length() ? s.get(p) : null;
    }

    private AtomicReferenceArray<T> pageForWrite(int key) {
        int p = key >>> PAGE_BITS;
        var s = spine;
        if (p < s.length()) {
            var page = s.get(p);
            if (page != null) return page;
        }
        return installPage(p);
    }

    /** Slow path, once per new page: install it, growing the spine geometrically if needed. */
    private AtomicReferenceArray<T> installPage(int p) {
        synchronized (growLock) {
            var s = spine;
            if (p >= s.length()) {
                // Keys up to Integer.MAX_VALUE need (MAX_VALUE >>> PAGE_BITS) + 1 pages.
                int newLength = (int) Math.min((Integer.MAX_VALUE >>> PAGE_BITS) + 1L,
                        Math.max(p + 1L, s.length() + (s.length() >> 1) + 1L));
                var grown = new AtomicReferenceArray<AtomicReferenceArray<T>>(newLength);
                for (int i = 0; i < s.length(); i++) grown.set(i, s.get(i));
                spine = grown;  // volatile publish after copying every existing page
                s = grown;
            }
            var page = s.get(p);
            if (page == null) {
                page = new AtomicReferenceArray<>(PAGE_SIZE);
                s.set(p, page);
            }
            return page;
        }
    }
}
