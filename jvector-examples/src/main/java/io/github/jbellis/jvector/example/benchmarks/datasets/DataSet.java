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

import java.util.List;

/**
 * Uniform access to vector test data, regardless of element type ({@code VectorFloat<?>} for
 * float32 datasets, {@code ByteSequence<?>} for int8 datasets).
 *
 * <p>Type-specific accessors (RAVV, similarity function) live on the concrete subclasses
 * {@link FloatDataSet} and {@link ByteDataSet} rather than here.
 *
 * @param <V> the vector element type
 */
public interface DataSet<V> {

    /**
     * The symbolic name of this dataset, used for dataset selection and result labeling.
     */
    String getName();

    /**
     * Dimensionality of the vectors in this dataset.
     */
    int getDimension();

    /**
     * Base vectors as a list.
     */
    List<V> getBaseVectors();

    /**
     * Query vectors as a list.
     * Each index corresponds to the same index in {@link #getGroundTruth()}.
     */
    List<V> getQueryVectors();

    /**
     * Ground truth as a list of neighbor-ordinal lists.
     * Each major index corresponds to the same index in {@link #getQueryVectors()}.
     * Each minor index is an ordinal into {@link #getBaseVectors()}.
     */
    List<? extends List<Integer>> getGroundTruth();
}
