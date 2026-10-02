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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.util.Objects;

/**
 * A remapped view over an existing {@link PQVectors} instance.
 * Ordinals are mapped from new ordinals (0 &lt;= newOrdinal &lt; outputCount) to old ordinals
 * in the source {@link PQVectors} instance using the provided {@link OrdinalMapper}.
 * Omitted ordinals or ordinals mapped out of bounds return zero-filled byte sequences.
 */
public class RemappedPQVectors extends PQVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int STREAMING_CHUNK_VECTORS = 1024;

    private final PQVectors source;
    private final int outputCount;
    private final OrdinalMapper mapper;
    private final ByteSequence<?> zeroSequence;

    public RemappedPQVectors(PQVectors source, int outputCount, OrdinalMapper mapper) {
        super(source.pq);
        if (outputCount < 0) {
            throw new IllegalArgumentException("Invalid outputCount " + outputCount);
        }
        this.source = Objects.requireNonNull(source);
        this.outputCount = outputCount;
        this.mapper = Objects.requireNonNull(mapper);
        this.zeroSequence = vectorTypeSupport.createByteSequence(pq.getSubspaceCount());
        this.zeroSequence.zero();
    }

    @Override
    public int count() {
        return outputCount;
    }

    @Override
    protected int validChunkCount() {
        return 0;
    }

    @Override
    public ByteSequence<?> get(int ordinal) {
        if (ordinal < 0 || ordinal >= outputCount) {
            throw new IndexOutOfBoundsException("Ordinal " + ordinal + " out of bounds for count " + outputCount);
        }
        int oldOrdinal = mapper.newToOld(ordinal);
        if (oldOrdinal >= 0 && oldOrdinal < source.count()) {
            return source.get(oldOrdinal);
        }
        return zeroSequence;
    }

    @Override
    public void write(IndexWriter out, int version) throws IOException {
        // pq codebooks
        pq.write(out, version);

        // compressed vectors
        out.writeInt(outputCount);
        out.writeInt(pq.getSubspaceCount());

        int M = pq.getSubspaceCount();
        int chunkCapacity = STREAMING_CHUNK_VECTORS * M;
        ByteSequence<?> chunkBuffer = vectorTypeSupport.createByteSequence(chunkCapacity);

        int ordinal = 0;
        while (ordinal < outputCount) {
            int vectorsInChunk = Math.min(STREAMING_CHUNK_VECTORS, outputCount - ordinal);
            int bytesInChunk = vectorsInChunk * M;

            for (int i = 0; i < vectorsInChunk; i++) {
                int newOrdinal = ordinal + i;
                int oldOrdinal = mapper.newToOld(newOrdinal);
                int destOffset = i * M;

                if (oldOrdinal >= 0 && oldOrdinal < source.count()) {
                    // Use get() rather than getChunk/getOffsetInChunk so that nested
                    // RemappedPQVectors sources (which have uninitialized vectorsPerChunk)
                    // are handled correctly via their overridden get().
                    ByteSequence<?> srcSeq = source.get(oldOrdinal);
                    for (int m = 0; m < M; m++) {
                        chunkBuffer.set(destOffset + m, srcSeq.get(m));
                    }
                } else {
                    for (int m = 0; m < M; m++) {
                        chunkBuffer.set(destOffset + m, (byte) 0);
                    }
                }
            }

            if (bytesInChunk == chunkCapacity) {
                vectorTypeSupport.writeByteSequence(out, chunkBuffer);
            } else {
                ByteSequence<?> tailBuffer = vectorTypeSupport.createByteSequence(bytesInChunk);
                for (int b = 0; b < bytesInChunk; b++) {
                    tailBuffer.set(b, chunkBuffer.get(b));
                }
                vectorTypeSupport.writeByteSequence(out, tailBuffer);
            }

            ordinal += vectorsInChunk;
        }
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        var sourceScoreFunction = source.precomputedScoreFunctionFor(q, similarityFunction);
        return (newOrdinal) -> {
            int oldOrdinal = mapper.newToOld(newOrdinal);
            if (oldOrdinal >= 0 && oldOrdinal < source.count()) {
                return sourceScoreFunction.similarityTo(oldOrdinal);
            }
            return 0.0f;
        };
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        var sourceScoreFunction = source.scoreFunctionFor(q, similarityFunction);
        return (newOrdinal) -> {
            int oldOrdinal = mapper.newToOld(newOrdinal);
            if (oldOrdinal >= 0 && oldOrdinal < source.count()) {
                return sourceScoreFunction.similarityTo(oldOrdinal);
            }
            return 0.0f;
        };
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        int oldNode1 = mapper.newToOld(node1);
        if (oldNode1 < 0 || oldNode1 >= source.count()) {
            return (node2) -> 0.0f;
        }
        var sourceDiversityFunction = source.diversityFunctionFor(oldNode1, similarityFunction);
        return (node2) -> {
            int oldNode2 = mapper.newToOld(node2);
            if (oldNode2 >= 0 && oldNode2 < source.count()) {
                return sourceDiversityFunction.similarityTo(oldNode2);
            }
            return 0.0f;
        };
    }

    @Override
    public long ramBytesUsed() {
        return source.ramBytesUsed() + zeroSequence.length();
    }
}
