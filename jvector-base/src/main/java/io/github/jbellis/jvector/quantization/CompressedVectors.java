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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;

/** Interface for compressed vectors. */
public interface CompressedVectors extends Accountable {
    /**
     * Write the compressed vectors to the given DataOutput
     * @param out the DataOutput to write to
     * @param version the serialization version.  versions 2 and 3 are supported
     * @throws IOException if an error occurs
     */
    void write(DataOutput out, int version) throws IOException;

    /**
     * Write the compressed vectors to the given DataOutput at the current serialization version
     * @param out the DataOutput to write to
     * @throws IOException if an error occurs
     */
    default void write(DataOutput out) throws IOException {
        write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    /**
     * Gets the original size of each vector, in bytes, before compression.
     * @return the original size of each vector, in bytes, before compression
     */
    int getOriginalSize();

    /**
     * Gets the compressed size of each vector, in bytes.
     * @return the compressed size of each vector, in bytes
     */
    int getCompressedSize();

    /**
     * Gets the compressor used by this instance.
     * @return the compressor used by this instance
     */
    VectorCompressor<?> getCompressor();

    /**
     * precomputes partial scores for the given query with every centroid; suitable for most searches
     * @param q the query vector
     * @param similarityFunction the similarity function
     * @return the approximate score function
     */
    ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /**
     * no precomputation; suitable when just a handful of score computations are performed
     * @param nodeId the node id
     * @param similarityFunction the similarity function
     * @return the approximate score function
     */
    ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int nodeId, VectorSimilarityFunction similarityFunction);

    /**
     * no precomputation; suitable when just a handful of score computations are performed
     * @param q the query vector
     * @param similarityFunction the similarity function
     * @return the approximate score function
     */
    ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /**
     * Use precomputedScoreFunctionFor instead.
     * @param q the query vector
     * @param similarityFunction the similarity function
     * @return the approximate score function
     */
    @Deprecated
    default ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return precomputedScoreFunctionFor(q, similarityFunction);
    }

    /**
     * the number of vectors
     * @return the number of vectors
     */
    int count();
}
