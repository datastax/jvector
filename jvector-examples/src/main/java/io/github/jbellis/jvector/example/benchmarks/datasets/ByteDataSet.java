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

import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.util.List;

/**
 * A dataset backed by native signed int8 (byte) vectors.
 */
public interface ByteDataSet extends BaseDataSet {

    /**
     * Get a random-access view of base byte vectors.
     * @return base byte vectors
     */
    RandomAccessByteVectorValues getBaseRavv();

    /**
     * The similarity function used for byte vector comparison.
     * @return the byte similarity function
     */
    ByteVectorSimilarityFunction getSimilarityFunction();

    /**
     * The base byte vectors as a list.
     * @return a list of base byte vectors
     */
    List<ByteSequence<?>> getBaseVectors();

    /**
     * The query byte vectors as a list.
     * @return a list of query byte vectors
     */
    List<ByteSequence<?>> getQueryVectors();

    @Override
    default int size() {
        return getBaseVectors().size();
    }

    @Override
    default int getQueryCount() {
        return getQueryVectors().size();
    }
}
