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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

class PanamaVectorUtilSupport implements VectorUtilSupport {
    protected final SimdOps simdOps;

    PanamaVectorUtilSupport(SimdOps simdOps) {
        this.simdOps = simdOps;
    }

    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return simdOps.dotProduct(a, b);
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return simdOps.cosineSimilarity(v1, v2);
    }

    @Override
    public float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return simdOps.cosineSimilarity(a, aoffset, b, boffset, length);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, VectorFloat<?> b) {
        return simdOps.squareDistance(a, b);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return simdOps.squareDistance( a, aoffset,  b, boffset, length);
    }

    @Override
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return simdOps.dotProduct(a, aoffset, b, boffset, length);
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        return simdOps.sum(vectors);
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return simdOps.sum( vector);
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {
        simdOps.scale( vector, multiplier);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        simdOps.addInPlace(v1, v2);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {
        simdOps.addInPlace(v1, value);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        simdOps.subInPlace( v1,  v2);
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {
        simdOps.subInPlace( vector, value);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Vectors must be the same length");
        }
        return simdOps.sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, float value) {
        return simdOps.sub(a, 0, value, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        return simdOps.sub( a, aOffset,  b, bOffset, length);
    }

    @Override
    public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        simdOps.minInPlace(v1, v2);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return simdOps.assembleAndSum(data, dataBase, baseOffsets, 0, baseOffsets.length());
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        return simdOps.assembleAndSum(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
    }

    @Override
    public int hammingDistance(long[] v1, long[] v2) {
        return simdOps.hammingDistance(v1, v2);
    }

    @Override
    public float max(VectorFloat<?> vector) {
        return simdOps.max( vector);
    }

    @Override
    public float min(VectorFloat<?> vector) {
        return simdOps.min( vector);
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    partialSums.set(codebookBase + i, dotProduct(codebook, i * size, query, queryOffset, size));
                    break;
                case EUCLIDEAN:
                    partialSums.set(codebookBase + i, squareDistance(codebook, i * size, query, queryOffset, size));
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBest) {
        float best = vsf == VectorSimilarityFunction.EUCLIDEAN ? Float.MAX_VALUE : -Float.MAX_VALUE;
        float val;
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    val = dotProduct(codebook, i * size, query, queryOffset, size);
                    partialSums.set(codebookBase + i, val);
                    best = Math.max(best, val);
                    break;
                case EUCLIDEAN:
                    val = squareDistance(codebook, i * size, query, queryOffset, size);
                    partialSums.set(codebookBase + i, val);
                    best = Math.min(best, val);
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
        partialBest.set(codebookIndex, best);
    }

    @Override
    public void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBases, ByteSequence<?> quantizedPartials) {
        simdOps.quantizePartials(delta, partials,  partialBases, (ArrayByteSequence) quantizedPartials);
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(),  clusterCount, partialSums, aMagnitude, bMagnitude);
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude)
    {
        return simdOps.pqDecodedCosineSimilarity(encoded, encodedOffset, encodedLength, clusterCount,  partialSums,  aMagnitude, bMagnitude);
    }

    @Override
    public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float alpha, float x0, float minValue, float maxValue) {
        return simdOps.nvqDotProduct8bit(
                 vector, (ArrayByteSequence) bytes,
                alpha, x0, minValue, maxValue);
    }

    @Override
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float alpha, float x0, float minValue, float maxValue) {
        return simdOps.nvqSquareDistance8bit(
                 vector, (ArrayByteSequence) bytes,
                alpha, x0, minValue, maxValue);
    }

    @Override
    public float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float alpha, float x0, float minValue, float maxValue, VectorFloat<?> centroid) {
        return simdOps.nvqCosine8bit(
                 vector, (ArrayByteSequence) bytes,
                alpha, x0, minValue, maxValue,
                 centroid
        );
    }

    @Override
    public void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
        simdOps.nvqShuffleQueryInPlace8bit( vector);
    }

    @Override
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        simdOps.nvqQuantize8bit( vector, alpha, x0, minValue, maxValue,(ArrayByteSequence) destination);
    }

    @Override
    public float nvqLoss(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        return simdOps.nvqLoss( vector, alpha, x0, minValue, maxValue, nBits);
    }

    @Override
    public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        return simdOps.nvqUniformLoss( vector, minValue, maxValue, nBits);
    }
}

