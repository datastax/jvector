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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.example.util.LruGrainRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/// A {@link DataSetWrapper} that serves base vectors through a bounded, grain-resolution LRU cache
/// ({@link LruGrainRandomAccessVectorValues}) in front of the origin's reader.
///
/// This is the wrapper for datasets larger than memory: the origin (typically the loader's
/// memory-mapped fvecs reader) stays on disk, and only the most recently used grains of
/// `grainSize` consecutive vectors are held in heap memory, up to `capacityBytes` in total.
///
/// Registered in {@link DataSets#wrapperProviders} under the name {@value #WRAPPER_NAME}, with the
/// options {@value #GRAIN_OPTION} and {@value #CAPACITY_MB_OPTION}. An option that is not given uses
/// a default, each overridable with a system property:
///
/// - `{@value #GRAIN_PROPERTY}` — vectors per grain, default {@value #DEFAULT_GRAIN_SIZE}
/// - `{@value #CAPACITY_MB_PROPERTY}` — cache capacity in MiB, default one quarter of the maximum heap
///
/// Programmatic users pick their own numbers with {@link #provider(int, long)}.
public final class LruCachedDataSet implements DataSetWrapper {
    private static final Logger logger = LoggerFactory.getLogger(LruCachedDataSet.class);

    /// Symbolic wrapper name, as used in `dataset(lru)` and `wrappers: [lru]`.
    public static final String WRAPPER_NAME = "lru";

    /// System property naming the vectors per grain for the symbolic wrapper.
    public static final String GRAIN_PROPERTY = "jvector.dataset.lru.grain";

    /// System property naming the cache capacity in MiB for the symbolic wrapper.
    public static final String CAPACITY_MB_PROPERTY = "jvector.dataset.lru.capacityMb";

    /// Vectors per grain when {@value #GRAIN_PROPERTY} is not set.
    public static final int DEFAULT_GRAIN_SIZE = 1024;

    /// Provider using the property-driven defaults; idempotent for already-wrapped datasets.
    public static final DataSetWrapper.Provider PROVIDER = LruCachedDataSet::of;

    /// Option key for vectors per grain, e.g. `lru[grain=4096]` or `lru: { grain: 4096 }`.
    public static final String GRAIN_OPTION = "grain";

    /// Option key for cache capacity in MiB, e.g. `lru[capacityMb=512]` or `lru: { capacityMb: 512 }`.
    public static final String CAPACITY_MB_OPTION = "capacityMb";

    /// Registry entry for {@value #WRAPPER_NAME}: accepts {@value #GRAIN_OPTION} and
    /// {@value #CAPACITY_MB_OPTION}, each falling back to the property-driven default when absent.
    public static final DataSetWrapper.Factory FACTORY = LruCachedDataSet::provider;

    /// Builds a provider from wrapper options.
    ///
    /// @param options {@value #GRAIN_OPTION} (positive integer) and/or {@value #CAPACITY_MB_OPTION} (positive integer)
    /// @return a provider with those settings, defaults filling any absent option
    /// @throws IllegalArgumentException on an unknown key or a non-positive or non-numeric value
    public static DataSetWrapper.Provider provider(Map<String, String> options) {
        int grainSize = defaultGrainSize();
        long capacityBytes = defaultCapacityBytes();
        for (var e : options.entrySet()) {
            switch (e.getKey()) {
                case GRAIN_OPTION:
                    grainSize = (int) positive(e.getKey(), e.getValue(), Integer.MAX_VALUE);
                    break;
                case CAPACITY_MB_OPTION:
                    capacityBytes = positive(e.getKey(), e.getValue(), Long.MAX_VALUE / (1024 * 1024)) * 1024 * 1024;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown option '" + e.getKey() + "' for dataset wrapper '" + WRAPPER_NAME
                            + "'; known options: " + GRAIN_OPTION + ", " + CAPACITY_MB_OPTION);
            }
        }
        if (options.isEmpty()) {
            return PROVIDER;
        }
        return provider(grainSize, capacityBytes);
    }

    private static long positive(String key, String value, long max) {
        long parsed;
        try {
            parsed = Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Option '" + key + "' of dataset wrapper '" + WRAPPER_NAME + "' must be an integer, got '" + value + "'");
        }
        if (parsed <= 0 || parsed > max) {
            throw new IllegalArgumentException("Option '" + key + "' of dataset wrapper '" + WRAPPER_NAME + "' must be between 1 and " + max + ", got " + parsed);
        }
        return parsed;
    }

    private final DataSet origin;
    private final LruGrainRandomAccessVectorValues baseRavv;

    /// Returns `origin` itself if it is already an {@link LruCachedDataSet}, otherwise a new cache over it
    /// sized by {@link #defaultGrainSize()} and {@link #defaultCapacityBytes()}.
    ///
    /// @param origin the dataset whose base vectors should be served through a bounded cache
    /// @return an LRU-cached view of `origin`
    public static DataSetWrapper of(DataSet origin) {
        if (origin instanceof LruCachedDataSet) {
            return (LruCachedDataSet) origin;
        }
        return new LruCachedDataSet(origin, defaultGrainSize(), defaultCapacityBytes());
    }

    /// @param grainSize     vectors per grain
    /// @param capacityBytes heap bytes of vector data to keep resident
    /// @return a provider building caches with exactly these parameters
    public static DataSetWrapper.Provider provider(int grainSize, long capacityBytes) {
        return origin -> new LruCachedDataSet(origin, grainSize, capacityBytes);
    }

    /// @return `{@value #GRAIN_PROPERTY}` or {@value #DEFAULT_GRAIN_SIZE}
    public static int defaultGrainSize() {
        return Integer.getInteger(GRAIN_PROPERTY, DEFAULT_GRAIN_SIZE);
    }

    /// @return `{@value #CAPACITY_MB_PROPERTY}` in bytes, or one quarter of the maximum heap
    public static long defaultCapacityBytes() {
        Long mb = Long.getLong(CAPACITY_MB_PROPERTY);
        return mb != null ? mb * 1024 * 1024 : Runtime.getRuntime().maxMemory() / 4;
    }

    /// Creates a cache of `capacityBytes` worth of `grainSize`-vector grains over `origin`.
    ///
    /// @param origin        the dataset to wrap
    /// @param grainSize     vectors per grain; must be positive
    /// @param capacityBytes heap bytes of vector data to keep resident; at least one grain is always kept
    public LruCachedDataSet(DataSet origin, int grainSize, long capacityBytes) {
        this.origin = origin;
        RandomAccessVectorValues source = origin.getBaseRavv();
        long grainBytes = (long) grainSize * source.dimension() * Float.BYTES;
        int maxGrains = (int) Math.max(1, Math.min(Integer.MAX_VALUE, capacityBytes / grainBytes));
        this.baseRavv = new LruGrainRandomAccessVectorValues(source, grainSize, maxGrains);
        logger.info("LRU cache over '{}': {} vectors in grains of {} ({} MB each), keeping at most {} of {} grains",
                origin.getName(), source.size(), grainSize, String.format("%.1f", grainBytes / (1024.0 * 1024.0)),
                maxGrains, baseRavv.grainCount());
    }

    @Override
    public DataSet getOrigin() {
        return origin;
    }

    @Override
    public LruGrainRandomAccessVectorValues getBaseRavv() {
        return baseRavv;
    }

    @Override
    public int getDimension() {
        return baseRavv.dimension();
    }
}
