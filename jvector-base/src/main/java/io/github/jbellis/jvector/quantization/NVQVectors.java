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
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

public class NVQVectors implements CompressedVectors {
    final NVQuantization nvq;
    final NVQScorer scorer;
    final NVQuantization.QuantizedVector[] compressedVectors;

    /**
     * Initialize the NVQVectors with an initial array of vectors.  This array may be
     * mutated, but caller is responsible for thread safety issues when doing so.
     */
    public NVQVectors(NVQuantization nvq, NVQuantization.QuantizedVector[] compressedVectors) {
        this.nvq = nvq;
        this.scorer = new NVQScorer(nvq);
        this.compressedVectors = compressedVectors;
    }

    @Override
    public int count() {
        return compressedVectors.length;
    }

    @Override
    public void write(DataOutput out, int version) throws IOException
    {
        // serializing NVQ at the given version
        nvq.write(out, version);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        for (var v : compressedVectors) {
            v.write(out);
        }
    }

    public static NVQVectors load(RandomAccessReader in) throws IOException {
        var nvq = NVQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        NVQuantization.QuantizedVector[] compressedVectors = new NVQuantization.QuantizedVector[size];

        for (int i = 0; i < size; i++) {
            compressedVectors[i] = NVQuantization.QuantizedVector.load(in);
        }

        return new NVQVectors(nvq, compressedVectors);
    }

    public static NVQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        return load(in);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NVQVectors that = (NVQVectors) o;
        if (!Objects.equals(nvq, that.nvq)) return false;
        return Arrays.deepEquals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nvq, Arrays.hashCode(compressedVectors));
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        var function = scorer.scoreFunctionFor(query, similarityFunction);
        return node2 -> function.similarityTo(compressedVectors[node2]);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        throw new UnsupportedOperationException();
    }

    public NVQuantization.QuantizedVector get(int ordinal) {
        return compressedVectors[ordinal];
    }

    public NVQuantization getNVQuantization() {
        return nvq;
    }

    @Override
    public int getOriginalSize() {
        return nvq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return nvq.compressedVectorSize();
    }

    @Override
    public NVQuantization getCompressor() {
        return nvq;
    }

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long codebooksSize = nvq.ramBytesUsed();
        long listSize = (long) REF_BYTES * (1 + compressedVectors.length);
        long dataSize = (long) (OH_BYTES + AH_BYTES + nvq.compressedVectorSize()) * compressedVectors.length;
        return codebooksSize + listSize + dataSize;
    }

    @Override
    public String toString() {
        return "NVQVectors{" +
                "NVQ=" + nvq +
                ", count=" + compressedVectors.length +
                '}';
    }
}
