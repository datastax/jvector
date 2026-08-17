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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;

/**
 * Compressed vector store produced by {@link ScalarQuantizer}.
 * Each vector is stored as a signed int8 {@link ByteSequence}.
 * Scoring uses {@link ByteVectorSimilarityFunction} for approximate distance comparisons.
 */
public class SQVectors implements CompressedVectors {
    private final ScalarQuantizer sq;
    private final ByteSequence<?>[] vectors;

    public SQVectors(ScalarQuantizer sq, ByteSequence<?>[] vectors) {
        this.sq = sq;
        this.vectors = vectors;
    }

    // ── CompressedVectors ────────────────────────────────────────────────────

    @Override
    public void write(IndexWriter out, int version) throws IOException {
        sq.write(out, version);
        out.writeInt(vectors.length);
        for (ByteSequence<?> v : vectors) {
            for (int i = 0; i < v.length(); i++) {
                out.writeByte(v.get(i));
            }
        }
    }

    public static SQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        ScalarQuantizer sq = ScalarQuantizer.load(in);
        int count = in.readInt();
        int dim = sq.compressedVectorSize();
        ByteSequence<?>[] vectors = new ByteSequence[count];
        byte[] buf = new byte[dim];
        for (int i = 0; i < count; i++) {
            in.readFully(buf);
            var bs = io.github.jbellis.jvector.vector.VectorizationProvider
                    .getInstance().getVectorTypeSupport().createByteSequence(dim);
            for (int d = 0; d < dim; d++) bs.set(d, buf[d]);
            vectors[i] = bs;
        }
        return new SQVectors(sq, vectors);
    }

    @Override
    public int getOriginalSize() {
        return sq.compressedVectorSize() * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return sq.compressedVectorSize();
    }

    @Override
    public ScalarQuantizer getCompressor() {
        return sq;
    }

    @Override
    public int count() {
        return vectors.length;
    }

    @Override
    public long ramBytesUsed() {
        if (vectors.length == 0) return 0;
        return (long) vectors.length * RamUsageEstimator.sizeOf(new byte[sq.compressedVectorSize()]);
    }

    // ── scoring ──────────────────────────────────────────────────────────────

    /**
     * Encodes the query with the same {@link ScalarQuantizer} and returns a byte×byte
     * approximate score function over the stored int8 vectors.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(
            VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(q, similarityFunction);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(
            VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        ByteVectorSimilarityFunction bvsf = toByteSimFunc(similarityFunction);
        ByteSequence<?> qBytes = sq.encode(q);
        return node -> bvsf.compare(qBytes, vectors[node]);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(
            int node1, VectorSimilarityFunction similarityFunction) {
        ByteVectorSimilarityFunction bvsf = toByteSimFunc(similarityFunction);
        ByteSequence<?> v1 = vectors[node1];
        return node2 -> bvsf.compare(v1, vectors[node2]);
    }

    /** Returns the raw int8 vector for the given node ordinal. */
    public ByteSequence<?> get(int node) {
        return vectors[node];
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static ByteVectorSimilarityFunction toByteSimFunc(VectorSimilarityFunction vsf) {
        switch (vsf) {
            case EUCLIDEAN:   return ByteVectorSimilarityFunction.EUCLIDEAN;
            case DOT_PRODUCT: return ByteVectorSimilarityFunction.DOT_PRODUCT;
            case COSINE:      return ByteVectorSimilarityFunction.COSINE;
            default: throw new IllegalArgumentException("No ByteVectorSimilarityFunction for " + vsf);
        }
    }

    @Override
    public String toString() {
        return "SQVectors{count=" + vectors.length + ", sq=" + sq + '}';
    }
}
