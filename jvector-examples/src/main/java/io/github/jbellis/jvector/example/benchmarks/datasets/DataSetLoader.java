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

import java.util.Optional;

/**
 * A DataSet Loader, which makes dataset sources modular and configurable without breaking existing callers.
 */
public interface DataSetLoader {
    /**
     * Looks up a dataset by name and returns a lightweight {@link DataSetInfo} handle.
     *
     * <p>The returned handle provides the dataset name and similarity function immediately,
     * without loading vector data into memory. The full {@link DataSet} (vectors, ground truth,
     * etc.) is loaded lazily on the first call to {@link DataSetInfo#getDataSet()}.
     *
     * <p>Implementations <em>MUST NOT</em> throw exceptions related to the presence or absence of a
     * requested dataset. Instead, {@link Optional} should be used. Other errors should still be indicated with
     * exceptions as usual, including any errors downloading or preparing a dataset which has been found.
     * Implementors should reliably return from this method, avoiding any {@link System#exit(int)} or similar calls.
     *
     * <p>Implementations may perform file downloads or other preparation work before returning the handle,
     * but should defer the expensive parsing and scrubbing of vector data to the {@link DataSetInfo} supplier.
     *
     * <HR/>
     *
     * <p>Implementations are encouraged to include logging at debug level for diagnostics, such as when datasets are
     * not found, and info level for when datasets are found and loaded. This can assist users troubleshooting
     * diverse data sources.
     *
     * @param dataSetName the logical dataset name (not a filename; do not include extensions like {@code .hdf5})
     * @return a {@link DataSetInfo} handle for the dataset, if found
     */
    Optional<DataSetInfo> loadDataSet(String dataSetName);
}
