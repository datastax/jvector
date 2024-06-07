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

import java.util.List;

final class PanamaVectorUtilSupport implements VectorUtilSupport {
    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, (ArrayVectorFloat)b);
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return SimdOps.cosineSimilarity((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.cosineSimilarity((ArrayVectorFloat)a, aoffset, (ArrayVectorFloat)b, boffset, length);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, VectorFloat<?> b) {
        return SimdOps.squareDistance((ArrayVectorFloat)a, (ArrayVectorFloat)b);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.squareDistance((ArrayVectorFloat) a, aoffset, (ArrayVectorFloat) b, boffset, length);
    }

    @Override
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, aoffset, (ArrayVectorFloat)b, boffset, length);
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        return SimdOps.sum(vectors);
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return SimdOps.sum((ArrayVectorFloat) vector);
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {
        SimdOps.scale((ArrayVectorFloat) vector, multiplier);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        SimdOps.addInPlace((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        SimdOps.subInPlace((ArrayVectorFloat) v1, (ArrayVectorFloat) v2);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Vectors must be the same length");
        }
        return SimdOps.sub((ArrayVectorFloat)a, 0, (ArrayVectorFloat)b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        return SimdOps.sub((ArrayVectorFloat) a, aOffset, (ArrayVectorFloat) b, bOffset, length);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        float sum = 0f;
        for (int i = 0; i < baseOffsets.length(); i++) {
            sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i)));
        }
        return sum;
    }

    @Override
    public int hammingDistance(long[] v1, long[] v2) {
        return SimdOps.hammingDistance(v1, v2);
    }

    @Override
    public float max(VectorFloat<?> vector) {
        return SimdOps.max((ArrayVectorFloat) vector);
    }

    @Override
    public float min(VectorFloat<?> vector) {
        return SimdOps.min((ArrayVectorFloat) vector);
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
        SimdOps.quantizePartials(delta, (ArrayVectorFloat) partials, (ArrayVectorFloat) partialBases, (ArrayByteSequence) quantizedPartials);
    }
}

