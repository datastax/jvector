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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Container for ASH-compressed vectors with a minimal scoring interface.
 *
 * <p>
 * This class intentionally implements only the functionality required for:
 * <ul>
 *   <li>Encoding benchmarks</li>
 *   <li>Approximate distance microbenchmarks</li>
 *   <li>Correctness validation against full-precision distances</li>
 * </ul>
 *
 * <p>
 * TODO (future work):
 * <ul>
 *   <li>Implement landmark-aware ASH scoring using residualNorm and dotWithLandmark</li>
 *   <li>Add a full ASH ScoreFunction aligned with graph search semantics</li>
 *   <li>Support similarityToNeighbor(...) where appropriate</li>
 *   <li>Integrate ASH as a Feature for on-disk graph indexes</li>
 *   <li>Define reranking strategy (likely via full vectors)</li>
 *   <li>Support multi-landmark ASH (this is intentionally deferred)</li>
 * </ul>
 *
 * <p>
 * NOTE:
 * This class avoids PQ/NVQ feature plumbing and graph integration by design.
 * Those concerns belong to the scoring plane and will be addressed after
 * baseline performance is established.
 */
public class ASHVectors implements CompressedVectors {

    final AsymmetricHashing ash;
    final AsymmetricHashing.QuantizedVector[] compressedVectors;

    /**
     * Initialize ASHVectors with an array of ASH-compressed vectors.
     *
     * <p>
     * The array is treated as immutable after construction.
     */
    public ASHVectors(AsymmetricHashing ash,
                      AsymmetricHashing.QuantizedVector[] compressedVectors) {
        this.ash = ash;
        this.compressedVectors = compressedVectors;
    }

    @Override
    public int count() {
        return compressedVectors.length;
    }

    /**
     * Serialize the compressor followed by the compressed vectors.
     *
     * <p>
     * NOTE:
     * This format is intentionally simple and versioned only at the compressor level.
     * Additional per-vector metadata (e.g., multiple landmarks) must be versioned
     * carefully when introduced.
     */
    @Override
    public void write(DataOutput out, int version) throws IOException {
        // Write ASH compressor first
        ash.write(out, version);

        // Write vector count
        out.writeInt(compressedVectors.length);

        // Write vectors
        for (var v : compressedVectors) {
            v.write(out, ash.quantizedDim);
        }
    }

    public static ASHVectors load(RandomAccessReader in) throws IOException {
        var ash = AsymmetricHashing.load(in);

        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }

        var compressedVectors = new AsymmetricHashing.QuantizedVector[size];
        for (int i = 0; i < size; i++) {
            compressedVectors[i] =
                    AsymmetricHashing.QuantizedVector.load(in, ash.quantizedDim);
        }

        return new ASHVectors(ash, compressedVectors);
    }

    /**
     * Return an approximate score function for the given query.
     *
     * <p>
     * Baseline implementation:
     * <ul>
     *   <li>Encode query once using ASH</li>
     *   <li>Score candidates using pure Hamming distance</li>
     * </ul>
     *
     * <p>
     * TODO:
     * <ul>
     *   <li>Replace pure Hamming with full ASH scoring</li>
     *   <li>Incorporate landmark scalars (dotWithLandmark, residualNorm)</li>
     *   <li>Respect VectorSimilarityFunction semantics where applicable</li>
     * </ul>
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {

        // Encode query once
        var qv = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim);
        ash.encodeTo(query, qv);
        final long[] queryBits = qv.binaryVector;

        return node -> {
            var v = compressedVectors[node];

            // TODO:
            //  - Replace with proper ASH scoring.
            //  - Consider affine scaling or normalization.
            int hamming = VectorUtil.hammingDistance(queryBits, v.binaryVector);

            // Higher similarity is better
            return -hamming;
        };
    }

    /**
     * For ASH, precomputed and non-precomputed scoring are currently identical.
     *
     * <p>
     * TODO:
     *  - Evaluate caching projected queries or other query-dependent state.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
    }

    /**
     * Diversity-aware scoring is not supported for ASH at this stage.
     *
     * <p>
     * TODO:
     *  - Define whether diversity should be measured via:
     *      * Hamming distance
     *      * Landmark-aware distance
     *      * Full-vector reranking
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(
            int node1,
            VectorSimilarityFunction similarityFunction) {
        throw new UnsupportedOperationException("ASH diversity scoring not implemented");
    }

    public AsymmetricHashing.QuantizedVector get(int ordinal) {
        return compressedVectors[ordinal];
    }

    @Override
    public int getOriginalSize() {
        return ash.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return ash.compressedVectorSize();
    }

    @Override
    public AsymmetricHashing getCompressor() {
        return ash;
    }

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long vectorsArraySize =
                AH_BYTES + (long) REF_BYTES * compressedVectors.length;
        long vectorsDataSize =
                (long) ash.compressedVectorSize() * compressedVectors.length;

        return ash.ramBytesUsed() + vectorsArraySize + vectorsDataSize;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ASHVectors that = (ASHVectors) o;
        return Objects.equals(ash, that.ash)
                && Arrays.equals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(ash);
        result = 31 * result + Arrays.hashCode(compressedVectors);
        return result;
    }

    @Override
    public String toString() {
        return "ASHVectors{count=" + compressedVectors.length +
                ", ash=" + ash + '}';
    }
}
