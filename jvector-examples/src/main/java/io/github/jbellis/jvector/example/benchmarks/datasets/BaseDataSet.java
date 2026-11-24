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

import java.io.IOException;
import java.util.List;

/**
 * Common base interface for all vector datasets, regardless of element type (float or byte).
 */
public interface BaseDataSet extends AutoCloseable {

    /** Release file-backed input resources after all benchmark consumers stop. */
    @Override
    default void close() throws IOException {}

    /**
     * Get dimensions of the vectors in this dataset.
     * @return the dimensionality
     */
    int getDimension();

    /** True only for the harness's explicit bounded-input mode. */
    default boolean useDatasetPrefetching() { return false; }

    /**
     * The symbolic name of this dataset, used for dataset selection and result labeling.
     * @return the dataset name
     */
    String getName();

    /**
     * The ground truth as a list.
     * Each major index corresponds to the query index.
     * Each minor index within represents the corresponding base vector ordinal.
     * @return the ground truth list
     */
    List<? extends List<Integer>> getGroundTruth();

    /**
     * The number of base vectors in this dataset.
     * @return the base vector count
     */
    int size();

    /**
     * The number of query vectors available. Works for all dataset types.
     * @return the query vector count
     */
    default int getQueryCount() {
        return getGroundTruth().size();
    }
}
