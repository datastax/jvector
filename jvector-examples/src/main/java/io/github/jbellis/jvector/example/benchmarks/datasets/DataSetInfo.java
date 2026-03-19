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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import java.util.Optional;
import java.util.function.Supplier;

/// A lightweight, lazy handle that separates *identifying* a dataset from *loading* its data.
///
/// Metadata such as the dataset name and similarity function are available immediately
/// without any I/O, while the expensive work of reading vectors, deduplicating, scrubbing
/// zero vectors, and normalizing is deferred until the first call to {@link #getDataSet()}.
///
/// This design allows callers to enumerate or filter available datasets cheaply, and
/// ensures that the full load-and-scrub pipeline runs at most once per handle thanks to
/// thread-safe caching.
///
/// Instances are created by {@link DataSetLoader} implementations; callers obtain them
/// through {@link DataSets#loadDataSet(String)}.
///
/// ### Typical usage
/// ```java
/// DataSetInfo info = DataSets.loadDataSet("ada002-100k").orElseThrow();
///
/// // Cheap — no vectors loaded yet
/// System.out.println(info.getName());
/// System.out.println(info.similarityFunction());
///
/// // First call triggers full load; subsequent calls return the cached DataSet
/// DataSet ds = info.getDataSet();
/// ```
///
/// @see DataSet
/// @see DataSetLoader
/// @see DataSets
public class DataSetInfo implements DataSetProperties {
    private final Supplier<DataSet> loader;
    private final DataSetProperties baseProperties;
    private volatile DataSet cached;

    /// Creates a new dataset info handle.
    ///
    /// The supplied {@code loader} will not be invoked until {@link #getDataSet()} is called.
    /// It should perform the full load-and-scrub pipeline (read vectors, remove duplicates /
    /// zero vectors, filter queries, normalize) and return a ready-to-use {@link DataSet}.
    ///
    /// @param baseProperties     the dataset properties (name, similarity function, etc.)
    /// @param loader             a supplier that performs the deferred load; invoked at most once
    public DataSetInfo(DataSetProperties baseProperties, Supplier<DataSet> loader) {
        this.baseProperties = baseProperties;
        this.loader = loader;
    }

    /**
     * @inheritDoc
     */
    @Override
    public Optional<VectorSimilarityFunction> similarityFunction() {
        return baseProperties.similarityFunction();
    }

    /**
     * @inheritDoc
     */
    @Override
    public int numVectors() {
        return this.baseProperties.numVectors();
    }

    /**
     * @inheritDoc
     */
    @Override
    public String getName() {
        return baseProperties.getName();
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean isNormalized() {
        return baseProperties.isNormalized();
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean isZeroVectorFree() {
        return baseProperties.isZeroVectorFree();
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean isDuplicateVectorFree() {
        return baseProperties.isDuplicateVectorFree();
    }

    /// Returns the fully loaded and scrubbed {@link DataSet}.
    ///
    /// On the first invocation this triggers the deferred load pipeline, which may involve
    /// reading large vector files from disk, deduplication, zero-vector removal, and
    /// normalization. The result is cached so that subsequent calls return immediately.
    ///
    /// This method is thread-safe: concurrent callers will block until the first load
    /// completes, after which all callers share the same cached instance.
    ///
    /// @return the ready-to-use {@link DataSet}
    public DataSet getDataSet() {
        if (cached == null) {
            synchronized (this) {
                if (cached == null) {
                    cached = loader.get();
                }
            }
        }
        return cached;
    }
}
