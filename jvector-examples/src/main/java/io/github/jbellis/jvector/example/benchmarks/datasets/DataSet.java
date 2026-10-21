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

/**
 * A dataset backed by single-precision floating point vectors (default dataset type in JVector).
 */
public interface DataSet extends BaseDataSet {

    /**
     * Get a random-access view of base vectors.
     * @return base vectors
     */
    RandomAccessVectorValues getBaseRavv();

    /**
     * The similarity function originally used to build this dataset, and the one that should be used for testing
     * during indexing and traversal.
     * @return the similarity function
     */
    VectorSimilarityFunction getSimilarityFunction();

    /**
     * The base vectors as a list.
     * @return a list of base vectors
     */
    List<VectorFloat<?>> getBaseVectors();

    /**
     * The query vectors as a list.
     * Each major index corresponds to the self-same index from {@link #getGroundTruth()}.
     * Ideally, the query vectors are disjoint with respect to the base vectors to improve testing integrity.
     * @return a list of query vectors
     */
    List<VectorFloat<?>> getQueryVectors();

    @Override
    default int size() {
        return getBaseVectors().size();
    }

    @Override
    default int getQueryCount() {
        return getQueryVectors().size();
    }
}
