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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;
import java.util.Map;

/// A {@link DataSet} layered over an origin dataset.
///
/// Wrappers change *how* a dataset's vectors are held or served without changing *what* they
/// are: every accessor delegates to {@link #getOrigin()} unless the wrapper overrides it. The
/// built-in wrappers re-home the base vectors: fully in heap memory ({@link InMemoryCachedDataSet}),
/// in a memory-mapped file ({@link MMapCachedDataSet}), or behind a bounded grain-resolution LRU
/// cache for datasets larger than memory ({@link LruCachedDataSet}).
///
/// Wrappers are applied by {@link DataSets} through a {@link Provider}, either passed explicitly
/// or resolved by name from {@link DataSets#wrapperProviders} when a dataset is named with a
/// symbolic wrapper list such as `cohere-english-v3-100k(mmap)` (see {@link DataSetSpec}).
///
/// @see DataSets
/// @see DataSetSpec
public interface DataSetWrapper extends DataSet {

    /// @return the dataset this wrapper is layered over
    DataSet getOrigin();

    @Override
    default int getDimension() {
        return getOrigin().getDimension();
    }

    @Override
    default RandomAccessVectorValues getBaseRavv() {
        return getOrigin().getBaseRavv();
    }

    @Override
    default String getName() {
        return getOrigin().getName();
    }

    @Override
    default VectorSimilarityFunction getSimilarityFunction() {
        return getOrigin().getSimilarityFunction();
    }

    @Override
    default List<VectorFloat<?>> getQueryVectors() {
        return getOrigin().getQueryVectors();
    }

    @Override
    default List<? extends List<Integer>> getGroundTruth() {
        return getOrigin().getGroundTruth();
    }

    /// Creates a {@link DataSetWrapper} over an origin dataset.
    ///
    /// Providers should be idempotent for their own wrapper type: wrapping an already-wrapped
    /// dataset of the same kind returns it unchanged rather than stacking a redundant layer.
    @FunctionalInterface
    interface Provider {
        /// @param origin the dataset to wrap
        /// @return the wrapped dataset
        DataSetWrapper wrap(DataSet origin);
    }

    /// Builds a {@link Provider} from the options written on a {@link DataSetSpec.WrapperSpec}.
    /// This is what {@link DataSets#wrapperProviders} registers under each symbolic wrapper name.
    @FunctionalInterface
    interface Factory {
        /// @param options the wrapper's options as written, possibly empty
        /// @return a provider configured with those options
        /// @throws IllegalArgumentException if an option is unknown or malformed
        Provider provider(Map<String, String> options);

        /// Wraps a provider that takes no options; any option given is rejected by name.
        ///
        /// @param wrapperName the symbolic name, for the error message
        /// @param provider    the provider to return
        /// @return a factory that yields `provider` when given no options
        static Factory optionless(String wrapperName, Provider provider) {
            return options -> {
                if (!options.isEmpty()) {
                    throw new IllegalArgumentException("Dataset wrapper '" + wrapperName + "' takes no options, got " + options.keySet());
                }
                return provider;
            };
        }
    }
}
