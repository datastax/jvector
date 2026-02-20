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

import java.util.function.Supplier;

/// A lightweight handle that identifies a dataset without eagerly loading its data.
/// The name and similarity function are available immediately, while the full
/// {@link DataSet} is loaded lazily on the first call to {@link #getDataSet()}.
public class DataSetInfo {
    private final String name;
    private final VectorSimilarityFunction similarityFunction;
    private final Supplier<DataSet> loader;
    private volatile DataSet cached;

    /// Creates a new dataset info handle.
    ///
    /// @param name the dataset name
    /// @param similarityFunction the similarity function used by this dataset
    /// @param loader a supplier that loads the full dataset on demand
    public DataSetInfo(String name, VectorSimilarityFunction similarityFunction, Supplier<DataSet> loader) {
        this.name = name;
        this.similarityFunction = similarityFunction;
        this.loader = loader;
    }

    /// Returns the dataset name, available without loading data.
    public String getName() {
        return name;
    }

    /// Returns the similarity function, available without loading data.
    public VectorSimilarityFunction getSimilarityFunction() {
        return similarityFunction;
    }

    /// Returns the full {@link DataSet}, loading it on first access and caching for subsequent calls.
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
