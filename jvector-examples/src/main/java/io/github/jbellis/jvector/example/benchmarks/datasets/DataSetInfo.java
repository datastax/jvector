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

public interface DataSetInfo extends DataSetProperties {
    /// Loads and returns a {@link InMemoryDataSet} corresponding to the underlying source.
    ///
    /// This method may incur an IO penalty based on the size of the dataset and it's source.
    /// Implementations are not required to cache the dataset or ensure thread-safety.
    ///
    /// @return the ready-to-use {@link InMemoryDataSet}
    public InMemoryDataSet getDataSet();
}
