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

/**
 * Represents a collection of vectors that have been compressed using a {@link VectorCompressor}.
 * <p>
 * This interface provides methods for serialization, size information, and creating score functions
 * for similarity comparisons. Compressed vectors trade precision for reduced memory footprint,
 * enabling efficient approximate nearest neighbor search.
 */
public interface CompressedVectors extends Accountable {
    /**
     * Writes the compressed vectors to the given DataOutput using the specified serialization version.
     * @param out the DataOutput to write to
     * @param version the serialization version; versions 2 and 3 are supported
     * @throws IOException if an I/O error occurs during writing
     */
    void write(DataOutput out, int version) throws IOException;

    /**
     * Writes the compressed vectors to the given DataOutput at the current serialization version.
     * @param out the DataOutput to write to
     * @throws IOException if an I/O error occurs during writing
     */
    default void write(DataOutput out) throws IOException {
        write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    /**
     * Returns the original size of each vector in bytes, before compression.
     * @return the original size of each vector, in bytes
     */
    int getOriginalSize();

    /**
     * Returns the compressed size of each vector in bytes.
     * @return the compressed size of each vector, in bytes
     */
    int getCompressedSize();

    /**
     * Returns the compressor used by this instance.
     * @return the compressor used by this instance
     */
    VectorCompressor<?> getCompressor();

    /**
     * Creates an approximate score function with precomputed partial scores for the query vector
     * against every centroid. This is suitable for most search operations where precomputation
     * cost can be amortized across many score comparisons.
     * @param q the query vector
     * @param similarityFunction the similarity function to use for scoring
     * @return an approximate score function with precomputed scores
     */
    ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /**
     * Creates an approximate score function without precomputation, suitable for diversity checks
     * where only a handful of score computations are performed per node.
     * @param nodeId the node ID to compute scores against
     * @param similarityFunction the similarity function to use for scoring
     * @return an approximate score function without precomputation
     */
    ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int nodeId, VectorSimilarityFunction similarityFunction);

    /**
     * Creates an approximate score function without precomputation, suitable when only a small number
     * of score computations are performed.
     * @param q the query vector
     * @param similarityFunction the similarity function to use for scoring
     * @return an approximate score function without precomputation
     */
    ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /**
     * Creates an approximate score function for the given query vector.
     * @param q the query vector
     * @param similarityFunction the similarity function to use for scoring
     * @return an approximate score function with precomputed scores
     * @deprecated use {@link #precomputedScoreFunctionFor(VectorFloat, VectorSimilarityFunction)} instead
     */
    @Deprecated
    default ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return precomputedScoreFunctionFor(q, similarityFunction);
    }

    /**
     * Returns the number of compressed vectors in this collection.
     * @return the number of vectors
     */
    int count();
}
