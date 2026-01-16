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
     * Implementations of this method <EM>MUST NOT</EM> throw exceptions related to the presence or absence of a
     * requested dataset. Instead, {@link Optional} should be used. Other errors should still be indicated with
     * exceptions as usual, including any errors loading a dataset which has been found. Implementors should reliably
     * return from this method, avoiding any {@link System#exit(int)} or similar calls.
     *
     * <HR/>
     *
     * Implementations are encouraged to include logging at debug level for diagnostics, such as when datasets are
     * not found, and info level for when datasets are found and loaded. This can assist users troubleshooting
     * diverse data sources.
     *
     * @param dataSetName
     * @return a {@link DataSet}, if found
     */
    Optional<DataSet> loadDataSet(String dataSetName);
}
