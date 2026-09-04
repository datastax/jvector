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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Optional;

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
    private final DataSetFiles dsFiles;
    private final DataSetProperties baseProperties;
    private volatile InMemoryDataSet cached;

    /// Creates a new dataset info handle.
    ///
    /// The dataset will not be loaded until {@link #getDataSet()} is called for the first time.
    ///
    /// @param baseProperties     the dataset properties (name, similarity function, etc.)
    /// @param dataSetFiles       the bundle of base/query/gt paths required to load the dataset
    public DataSetInfo(DataSetProperties baseProperties, DataSetFiles dataSetFiles) {
        this.baseProperties = baseProperties;
        this.dsFiles = dataSetFiles;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<VectorSimilarityFunction> similarityFunction() {
        return baseProperties.similarityFunction();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int numVectors() {
        return this.baseProperties.numVectors();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getName() {
        return baseProperties.getName();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isNormalized() {
        return baseProperties.isNormalized();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isZeroVectorFree() {
        return baseProperties.isZeroVectorFree();
    }

    /**
     * {@inheritDoc}
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
    public InMemoryDataSet getDataSet() {
        if (cached == null) {
            synchronized (this) {
                if (cached == null) {
                    var baseVectors = SiftLoader.readFvecs(dsFiles.getBaseFvecsPath().toString());
                    var queryVectors = SiftLoader.readFvecs(dsFiles.getQueryFvecsPath().toString());
                    var gtVectors = SiftLoader.readIvecs(dsFiles.getGtIvecsPath().toString());
                    cached = DataSetUtils.processDataSet(baseProperties.getName(), baseProperties, baseVectors, queryVectors, gtVectors);
                }
            }
        }
        return cached;
    }

    /// Returns a {@link DataSet} whose index vectors are backed by memory-mapped files.
    /// The query vectors and ground truth are still loaded into memory.
    ///
    /// The output is not cached, so multiple invocoations will cause duplicate loads
    /// and create duplicate maps.
    ///
    /// @return the ready-to-use {@link DataSet}
    public RavvDataSet getMappedDataSet() {
        if (baseProperties.loadBehavior() == LoadBehavior.LEGACY_SCRUB) {
            throw new UnsupportedOperationException(
                "Can't map a dataset that wants to be scrubbed, load the full dataset with getDataSet()"
            );
        }
        var maybeVsf = baseProperties.similarityFunction();
        if (maybeVsf.isEmpty()) {
            throw new RuntimeException("Need a similarity function to create dataset");
        }
        try {
            return new RavvDataSet(
                baseProperties.getName(),
                maybeVsf.get(),
                FvecRavv.of(dsFiles.getBaseFvecsPath()),
                SiftLoader.readFvecs(dsFiles.getQueryFvecsPath().toString()),
                SiftLoader.readIvecs(dsFiles.getGtIvecsPath().toString()));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
