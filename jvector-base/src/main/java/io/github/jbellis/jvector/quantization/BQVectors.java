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

public abstract class BQVectors implements CompressedVectors {
    protected final BinaryQuantization bq;
    protected long[][] compressedVectors;

    protected BQVectors(BinaryQuantization bq) {
        this.bq = bq;
    }

    @Override
    public void write(DataOutput out, int version) throws IOException {
        // BQ centering data
        bq.write(out, version);

        // compressed vectors
        out.writeInt(count());
        if (count() <= 0) {
            return;
        }
        out.writeInt(compressedVectors[0].length);
        for (int i = 0; i < count(); i++) {
            var v = compressedVectors[i];
            for (long l : v) {
                out.writeLong(l);
            }
        }
    }

    public static BQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);

        // BQ
        var bq = BinaryQuantization.load(in);

        // check validity of compressed vectors header
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        var compressedVectors = new long[size][];
        if (size == 0) {
            return new ImmutableBQVectors(bq, compressedVectors);
        }
        int compressedLength = in.readInt();
        if (compressedLength < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedLength);
        }

        // read the compressed vectors
        for (int i = 0; i < size; i++)
        {
            long[] vector = new long[compressedLength];
            in.readFully(vector);
            compressedVectors[i] = vector;
        }

        return new ImmutableBQVectors(bq, compressedVectors);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(q, similarityFunction);
    }

    /**
     * Note that `similarityFunction` is ignored, you always get Hamming distance similarity with BQ, which
     * is a useful approximation for cosine distance and not really anything else.
     */
    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        var qBQ = compressedVectors[node1];
        return node2 -> {
            var vBQ = compressedVectors[node2];
            return similarityBetween(qBQ, vBQ);
        };
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        var qBQ = bq.encode(q);
        return node2 -> {
            var vBQ = compressedVectors[node2];
            return similarityBetween(qBQ, vBQ);
        };
    }

    public float similarityBetween(long[] encoded1, long[] encoded2) {
        return 1 - (float) VectorUtil.hammingDistance(encoded1, encoded2) / bq.getOriginalDimension();
    }

    public long[] get(int i) {
        return compressedVectors[i];
    }

    @Override
    public int getOriginalSize() {
        return bq.getOriginalDimension() * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return bq.compressedVectorSize();
    }

    @Override
    public BinaryQuantization getCompressor() {
        return bq;
    }

    @Override
    public long ramBytesUsed() {
        long[] compressedVector = compressedVectors[0];
        if (compressedVector == null) {
            return 0;
        }
        return count() * RamUsageEstimator.sizeOf(compressedVector);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BQVectors bqVectors = (BQVectors) o;
        return Objects.equals(bq, bqVectors.bq) && Arrays.deepEquals(compressedVectors, bqVectors.compressedVectors);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(bq);
        result = 31 * result + Arrays.deepHashCode(compressedVectors);
        return result;
    }

    @Override
    public String toString() {
        return "BQVectors{" +
               "bq=" + bq +
               ", count=" + count() +
               '}';
    }
}
