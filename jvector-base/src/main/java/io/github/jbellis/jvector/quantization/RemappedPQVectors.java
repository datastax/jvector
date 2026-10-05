/*
 * Copyright IBM Corp.
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
 * A lazy remapped view over an existing {@link PQVectors} instance.
 * Ordinals are translated from new ordinals to old ordinals in the source
 * {@link PQVectors} instance using the provided {@link OrdinalMapper}.
 * No vectors are copied; this is a zero-allocation wrapper.
 */
public class RemappedPQVectors extends PQVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final PQVectors source;
    private final OrdinalMapper mapper;

    /**
     * Creates a remapped view of {@code source} where ordinals are translated
     * through {@code mapper}.
     *
     * @param source the underlying PQVectors whose compressed data is reused
     * @param mapper maps new (view) ordinals to old (source) ordinals
     */
    public RemappedPQVectors(PQVectors source, OrdinalMapper mapper) {
        super(source.getCompressor());
        this.source = Objects.requireNonNull(source, "source");
        this.mapper = Objects.requireNonNull(mapper, "mapper");
    }

    @Override
    public int count() {
        return mapper.maxOrdinal() + 1;
    }

    @Override
    protected int validChunkCount() {
        throw new UnsupportedOperationException("RemappedPQVectors is a lazy view and does not manage chunks directly");
    }

    @Override
    public ByteSequence<?> get(int ordinal) {
        if (ordinal < 0 || ordinal >= count())
            throw new IndexOutOfBoundsException("Ordinal " + ordinal + " out of bounds for vector count " + count());
        int oldOrdinal = mapper.newToOld(ordinal);
        if (oldOrdinal < 0 || oldOrdinal >= source.count())
            throw new IndexOutOfBoundsException("Mapped ordinal " + oldOrdinal + " out of bounds for source count " + source.count());
        return source.get(oldOrdinal);
    }

    @Override
    public void write(IndexWriter out, int version) throws IOException {
        ProductQuantization pq = getCompressor();

        // pq codebooks
        pq.write(out, version);

        // compressed vectors
        int totalCount = count();
        out.writeInt(totalCount);
        int subspaceCount = pq.getSubspaceCount();
        out.writeInt(subspaceCount);
        for (int i = 0; i < totalCount; i++) {
            vectorTypeSupport.writeByteSequence(out, get(i));
        }
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        ScoreFunction.ApproximateScoreFunction sourceScoreFunction = source.precomputedScoreFunctionFor(q, similarityFunction);
        int viewCount = count();
        int srcCount = source.count();
        return newOrdinal -> {
            if (newOrdinal < 0 || newOrdinal >= viewCount)
                throw new IndexOutOfBoundsException("Ordinal " + newOrdinal + " out of bounds for vector count " + viewCount);
            int oldOrdinal = mapper.newToOld(newOrdinal);
            if (oldOrdinal < 0 || oldOrdinal >= srcCount)
                throw new IndexOutOfBoundsException("Mapped ordinal " + oldOrdinal + " out of bounds for source count " + srcCount);
            return sourceScoreFunction.similarityTo(oldOrdinal);
        };
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        ScoreFunction.ApproximateScoreFunction sourceScoreFunction = source.scoreFunctionFor(q, similarityFunction);
        int viewCount = count();
        int srcCount = source.count();
        return newOrdinal -> {
            if (newOrdinal < 0 || newOrdinal >= viewCount)
                throw new IndexOutOfBoundsException("Ordinal " + newOrdinal + " out of bounds for vector count " + viewCount);
            int oldOrdinal = mapper.newToOld(newOrdinal);
            if (oldOrdinal < 0 || oldOrdinal >= srcCount)
                throw new IndexOutOfBoundsException("Mapped ordinal " + oldOrdinal + " out of bounds for source count " + srcCount);
            return sourceScoreFunction.similarityTo(oldOrdinal);
        };
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        if (node1 < 0 || node1 >= count())
            throw new IndexOutOfBoundsException("Ordinal " + node1 + " out of bounds for vector count " + count());
        int oldNode1 = mapper.newToOld(node1);
        if (oldNode1 < 0 || oldNode1 >= source.count())
            throw new IndexOutOfBoundsException("Mapped ordinal " + oldNode1 + " out of bounds for source count " + source.count());
        ScoreFunction.ApproximateScoreFunction sourceDiversityFunction = source.diversityFunctionFor(oldNode1, similarityFunction);
        int viewCount = count();
        int srcCount = source.count();
        return node2 -> {
            if (node2 < 0 || node2 >= viewCount)
                throw new IndexOutOfBoundsException("Ordinal " + node2 + " out of bounds for vector count " + viewCount);
            int oldNode2 = mapper.newToOld(node2);
            if (oldNode2 < 0 || oldNode2 >= srcCount)
                throw new IndexOutOfBoundsException("Mapped ordinal " + oldNode2 + " out of bounds for source count " + srcCount);
            return sourceDiversityFunction.similarityTo(oldNode2);
        };
    }

    @Override
    public long ramBytesUsed() {
        return source.ramBytesUsed();
    }
}
