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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.*;

/**
 * This provides a uniform way to access vector test data, regardless of where it comes from or how it is implemented.
 */
public interface DataSet {

    /**
     * Get dimensions of the vectors in this dataset.
     * @return the dimensionality
     */
    int getDimension();

    /**
     * Get a random-access view of base vectors.
     * @return base vectors
     */
    RandomAccessVectorValues getBaseRavv();

    /**
     * The symbolic name of this dataset, used for dataset selection and result labeling.
     * @return the dataset name
     */
    String getName();

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

    /**
     * The ground truth as a list.
     * Each major index corresponds to the self-same index from {@link #getQueryVectors()}.
     * Each minor index within represents the corresponding ordinal from {@link #getBaseVectors()} and {@link #getBaseRavv()}.
     * @return a list of query vectors.
     */
    List<? extends List<Integer>> getGroundTruth();
}
