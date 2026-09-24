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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/// A thread-safe, bounded, least-recently-used cache of heap-resident vectors in front of any
/// {@link RandomAccessVectorValues}, organised in *grains* of `grainSize` consecutive ordinals.
///
/// A read of ordinal `n` serves grain `n / grainSize` from the cache, loading it on a miss through
/// `origin.range(start, end).copy()` so the origin may be value-shared (a memory-mapped file, say).
/// At most `maxGrains` loaded grains are kept; when a load would exceed that, the grain with the
/// oldest last use is dropped. The bound is soft only while more than `maxGrains` grains are being
/// loaded concurrently, since a grain still loading is never evicted.
///
/// Vectors handed out are independent heap objects owned by their grain. They stay valid after the
/// grain is evicted (the caller's reference keeps them alive), so this reader is not value-shared
/// and {@link #copy()} returns `this`: one cache is meant to be shared by every thread.
///
/// Loads of distinct grains proceed in parallel; concurrent misses on the same grain wait for the
/// first loader. A failed load is retried by the next reader rather than poisoning the grain.
public final class LruGrainRandomAccessVectorValues implements RandomAccessVectorValues {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final RandomAccessVectorValues origin;
    private final int size;
    private final int dimension;
    private final int grainSize;
    private final int maxGrains;
    private final int grainCount;

    private final Map<Integer, Grain> grains = new ConcurrentHashMap<>();
    private final Object evictionLock = new Object();
    private final AtomicLong clock = new AtomicLong();
    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder evictions = new LongAdder();

    private static final class Grain {
        final CountDownLatch loaded = new CountDownLatch(1);
        volatile VectorFloat<?>[] vectors;
        volatile long lastUsed;
    }

    /// @param origin    the vectors to cache; read through ranged views, so it may be value-shared
    /// @param grainSize consecutive ordinals per grain; must be positive
    /// @param maxGrains loaded grains to keep; must be positive
    public LruGrainRandomAccessVectorValues(RandomAccessVectorValues origin, int grainSize, int maxGrains) {
        if (grainSize <= 0) {
            throw new IllegalArgumentException("grainSize must be positive, got " + grainSize);
        }
        if (maxGrains <= 0) {
            throw new IllegalArgumentException("maxGrains must be positive, got " + maxGrains);
        }
        this.origin = origin;
        this.size = origin.size();
        this.dimension = origin.dimension();
        this.grainSize = grainSize;
        this.maxGrains = maxGrains;
        this.grainCount = (int) (((long) size + grainSize - 1) / grainSize);
    }

    /// @return consecutive ordinals per grain
    public int grainSize() {
        return grainSize;
    }

    /// @return the maximum number of loaded grains kept resident
    public int maxGrains() {
        return maxGrains;
    }

    /// @return the number of grains the origin divides into
    public int grainCount() {
        return grainCount;
    }

    /// @return grains currently in the cache, loading ones included
    public int residentGrains() {
        return grains.size();
    }

    /// @return reads served from an already-loaded grain
    public long hits() {
        return hits.sum();
    }

    /// @return grain loads performed
    public long misses() {
        return misses.sum();
    }

    /// @return grains dropped to stay within {@link #maxGrains()}
    public long evictions() {
        return evictions.sum();
    }

    /// @return a one-line summary of the cache counters, for logs
    public String stats() {
        return String.format("grains=%d/%d resident=%d hits=%d misses=%d evictions=%d",
                Math.min(grainCount, maxGrains), grainCount, grains.size(), hits(), misses(), evictions());
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        int idx = Objects.checkIndex(nodeId, size);
        int g = idx / grainSize;
        Grain grain = grain(g);
        return grain.vectors[idx - g * grainSize];
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        destinationVector.copyFrom(getVector(node), 0, offset, dimension);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return this;
    }

    private Grain grain(int g) {
        while (true) {
            Grain grain = grains.get(g);
            if (grain == null) {
                Grain fresh = new Grain();
                grain = grains.putIfAbsent(g, fresh);
                if (grain == null) {
                    return load(g, fresh);
                }
            }
            if (grain.vectors == null) {
                awaitLoaded(grain);
                if (grain.vectors == null) {
                    continue; // the loader failed and withdrew the grain; try again
                }
            }
            hits.increment();
            grain.lastUsed = clock.incrementAndGet();
            return grain;
        }
    }

    private Grain load(int g, Grain grain) {
        try {
            grain.lastUsed = clock.incrementAndGet();
            evictIfNeeded(grain);
            int start = g * grainSize;
            int end = Math.min(size, start + grainSize);
            RandomAccessVectorValues slice = origin.range(start, end).copy();
            VectorFloat<?>[] vectors = new VectorFloat<?>[end - start];
            for (int i = 0; i < vectors.length; i++) {
                VectorFloat<?> v = vts.createFloatVector(dimension);
                slice.getVectorInto(i, v, 0);
                vectors[i] = v;
            }
            grain.vectors = vectors;
            misses.increment();
            grain.lastUsed = clock.incrementAndGet();
            return grain;
        } catch (RuntimeException | Error e) {
            grains.remove(g, grain);
            throw e;
        } finally {
            grain.loaded.countDown();
        }
    }

    private static void awaitLoaded(Grain grain) {
        try {
            grain.loaded.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for a grain to load", e);
        }
    }

    private void evictIfNeeded(Grain incoming) {
        if (grains.size() <= maxGrains) {
            return;
        }
        synchronized (evictionLock) {
            while (grains.size() > maxGrains) {
                Integer victimKey = null;
                Grain victim = null;
                for (var e : grains.entrySet()) {
                    Grain candidate = e.getValue();
                    if (candidate == incoming || candidate.vectors == null) {
                        continue; // never evict a grain that is still loading
                    }
                    if (victim == null || candidate.lastUsed < victim.lastUsed) {
                        victim = candidate;
                        victimKey = e.getKey();
                    }
                }
                if (victim == null) {
                    return; // everything resident is mid-load; the bound is exceeded until those finish
                }
                if (grains.remove(victimKey, victim)) {
                    evictions.increment();
                }
            }
        }
    }
}
