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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ImmutablePQVectors extends PQVectors {
    private final int vectorCount;
    private final Map<VectorSimilarityFunction, VectorFloat<?>> codebookPartialSumsMap;

    /**
     * Construct an immutable PQVectors instance with the given ProductQuantization and compressed data chunks.
     * @param pq the ProductQuantization to use
     * @param compressedDataChunks the compressed data chunks
     * @param vectorCount the number of vectors
     * @param vectorsPerChunk the number of vectors per chunk
     */
    public ImmutablePQVectors(ProductQuantization pq, ByteSequence<?>[] compressedDataChunks, int vectorCount, int vectorsPerChunk) {
        super(pq);
        this.compressedDataChunks = compressedDataChunks;
        this.vectorCount = vectorCount;
        this.vectorsPerChunk = vectorsPerChunk;
        this.codebookPartialSumsMap = new ConcurrentHashMap<>();
    }

    @Override
    protected int validChunkCount() {
        return compressedDataChunks.length;
    }

    @Override
    public int count() {
        return vectorCount;
    }

    private VectorFloat<?> getOrCreateCodebookPartialSums(VectorSimilarityFunction vsf) {
        return codebookPartialSumsMap.computeIfAbsent(vsf, pq::createCodebookPartialSums);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        final int subspaceCount = pq.getSubspaceCount();
        var node1Chunk = getChunk(node1);
        var node1Offset = getOffsetInChunk(node1);
        int clusterCount = pq.getClusterCount();

        VectorFloat<?> codebookPartialSums = getOrCreateCodebookPartialSums(similarityFunction);

        switch (similarityFunction) {
            case DOT_PRODUCT:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = VectorUtil.assembleAndSumPQ(codebookPartialSums, subspaceCount, node1Chunk, node1Offset, node2Chunk, node2Offset, clusterCount);
                    // scale to [0, 1]
                    return (1 + sum) / 2;
                };
            case COSINE:
                float norm1 = VectorUtil.assembleAndSumPQ(codebookPartialSums, subspaceCount, node1Chunk, node1Offset, node1Chunk, node1Offset, clusterCount);
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float sum = VectorUtil.assembleAndSumPQ(codebookPartialSums, subspaceCount, node1Chunk, node1Offset, node2Chunk, node2Offset, clusterCount);
                    float norm2 = VectorUtil.assembleAndSumPQ(codebookPartialSums, subspaceCount, node2Chunk, node2Offset, node2Chunk, node2Offset, clusterCount);
                    float cosine = sum / (float) Math.sqrt(norm1 * norm2);
                    // scale to [0, 1]
                    return (1 + cosine) / 2;
                };
            case EUCLIDEAN:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = VectorUtil.assembleAndSumPQ(codebookPartialSums, subspaceCount, node1Chunk, node1Offset, node2Chunk, node2Offset, clusterCount);

                    // scale to [0, 1]
                    return 1 / (1 + sum);
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }
}
