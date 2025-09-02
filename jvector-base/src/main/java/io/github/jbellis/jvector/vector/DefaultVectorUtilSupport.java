/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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

/**
 * A VectorUtilSupport implementation supported by JDK 11+. This implementation assumes the VectorFloat/ByteSequence
 * objects wrap an on-heap array of the corresponding type.
 */
final class DefaultVectorUtilSupport implements VectorUtilSupport {

  @Override
  public float dotProduct(VectorFloat<?> av, VectorFloat<?> bv) {
    float[] a = ((ArrayVectorFloat) av).get();
    float[] b = ((ArrayVectorFloat) bv).get();

    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += b[i] * a[i];
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
      res +=
          b[i + 8] * a[i + 8]
              + b[i + 9] * a[i + 9]
              + b[i + 10] * a[i + 10]
              + b[i + 11] * a[i + 11]
              + b[i + 12] * a[i + 12]
              + b[i + 13] * a[i + 13]
              + b[i + 14] * a[i + 14]
              + b[i + 15] * a[i + 15];
      res +=
          b[i + 16] * a[i + 16]
              + b[i + 17] * a[i + 17]
              + b[i + 18] * a[i + 18]
              + b[i + 19] * a[i + 19]
              + b[i + 20] * a[i + 20]
              + b[i + 21] * a[i + 21]
              + b[i + 22] * a[i + 22]
              + b[i + 23] * a[i + 23];
      res +=
          b[i + 24] * a[i + 24]
              + b[i + 25] * a[i + 25]
              + b[i + 26] * a[i + 26]
              + b[i + 27] * a[i + 27]
              + b[i + 28] * a[i + 28]
              + b[i + 29] * a[i + 29]
              + b[i + 30] * a[i + 30]
              + b[i + 31] * a[i + 31];
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
    }
    return res;
  }

  @Override
  public float dotProduct(VectorFloat<?> av, int aoffset, VectorFloat<?> bv, int boffset, int length)
  {
    float[] b = ((ArrayVectorFloat) bv).get();
    float[] a = ((ArrayVectorFloat) av).get();

    float sum = 0f;
    for (int i = 0; i < length; i++) {
      sum += a[aoffset + i] * b[boffset + i];
    }

    return sum;
  }

  @Override
  public float cosine(VectorFloat<?> av, VectorFloat<?> bv) {
    float[] a = ((ArrayVectorFloat) av).get();
    float[] b = ((ArrayVectorFloat) bv).get();

    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int dim = a.length;

    for (int i = 0; i < dim; i++) {
      float elem1 = a[i];
      float elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }

  @Override
  public float cosine(VectorFloat<?> av, int aoffset, VectorFloat<?> bv, int boffset, int length) {
    float[] a = ((ArrayVectorFloat) av).get();
    float[] b = ((ArrayVectorFloat) bv).get();
    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    for (int i = 0; i < length; i++) {
      float elem1 = a[aoffset + i];
      float elem2 = b[(boffset + i)];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }

  @Override
  public float squareDistance(VectorFloat<?> av, VectorFloat<?> bv) {
    float[] a = ((ArrayVectorFloat) av).get();
    float[] b = ((ArrayVectorFloat) bv).get();

    float squareSum = 0.0f;
    int dim = a.length;
    int i;
    for (i = 0; i + 8 <= dim; i += 8) {
      squareSum += squareDistanceUnrolled(a, b, i);
    }
    for (; i < dim; i++) {
      float diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  private static float squareDistanceUnrolled(float[] v1, float[] v2, int index) {
    float diff0 = v1[index + 0] - v2[index + 0];
    float diff1 = v1[index + 1] - v2[index + 1];
    float diff2 = v1[index + 2] - v2[index + 2];
    float diff3 = v1[index + 3] - v2[index + 3];
    float diff4 = v1[index + 4] - v2[index + 4];
    float diff5 = v1[index + 5] - v2[index + 5];
    float diff6 = v1[index + 6] - v2[index + 6];
    float diff7 = v1[index + 7] - v2[index + 7];
    return diff0 * diff0
        + diff1 * diff1
        + diff2 * diff2
        + diff3 * diff3
        + diff4 * diff4
        + diff5 * diff5
        + diff6 * diff6
        + diff7 * diff7;
  }

  @Override
  public float squareDistance(VectorFloat<?> av, int aoffset, VectorFloat<?> bv, int boffset, int length)
  {
    float[] a = ((ArrayVectorFloat) av).get();
    float[] b = ((ArrayVectorFloat) bv).get();

    float squareSum = 0f;
    for (int i = 0; i < length; i++) {
      float diff = a[aoffset + i] - b[boffset + i];
      squareSum += diff * diff;
    }

    return squareSum;
  }

  @Override
  public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {

    VectorFloat<?> sum = new ArrayVectorFloat(vectors.get(0).length());
    for (VectorFloat<?> vector : vectors) {
      for (int i = 0; i < vector.length(); i++) {
        sum.set(i, sum.get(i) + vector.get(i));
      }
    }
    return sum;
  }

  @Override
  public float sum(VectorFloat<?> vector) {
    float sum = 0;
    for (int i = 0; i < vector.length(); i++) {
      sum += vector.get(i);
    }

    return sum;
  }

  @Override
  public void scale(VectorFloat<?> vector, float multiplier) {
    for (int i = 0; i < vector.length(); i++) {
      vector.set(i, vector.get(i) * multiplier);
    }
  }

  @Override
  public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    for (int i = 0; i < v1.length(); i++) {
      v1.set(i, v1.get(i) + v2.get(i));
    }
  }

  /** Adds value to each element of v1, in place (v1 will be modified) */
  public void addInPlace(VectorFloat<?> v1, float value) {
    for (int i = 0; i < v1.length(); i++) {
      v1.set(i, v1.get(i) + value);
    }
  }

  @Override
  public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    for (int i = 0; i < v1.length(); i++) {
      v1.set(i, v1.get(i) - v2.get(i));
    }
  }

  @Override
  public void subInPlace(VectorFloat<?> vector, float value) {
    for (int i = 0; i < vector.length(); i++) {
      vector.set(i, vector.get(i) - value);
    }
  }

  @Override
  public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
    return sub(a, 0, b, 0, a.length());
  }

  @Override
  public VectorFloat<?> sub(VectorFloat<?> a, float value) {
    VectorFloat<?> result = new ArrayVectorFloat(a.length());
    for (int i = 0; i < a.length(); i++) {
      result.set(i, a.get(i) - value);
    }
    return result;
  }

  @Override
  public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    VectorFloat<?> result = new ArrayVectorFloat(length);
    for (int i = 0; i < length; i++) {
      result.set(i, a.get(aOffset + i) - b.get(bOffset + i));
    }
    return result;
  }

  @Override
  public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    for (int i = 0; i < v1.length(); i++) {
      v1.set(i, Math.min(v1.get(i), v2.get(i)));
    }
  }

  @Override
  public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
    return assembleAndSum(data, dataBase, baseOffsets, 0, baseOffsets.length());
  }

  @Override
  public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    float sum = 0f;
    for (int i = 0; i < baseOffsetsLength; i++) {
      sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));
    }
    return sum;
  }

  @Override
  public float assembleAndSumPQ(
          VectorFloat<?> codebookPartialSums,
          int subspaceCount,                  // = M
          ByteSequence<?> vector1Ordinals,
          int vector1OrdinalOffset,
          ByteSequence<?> vector2Ordinals,
          int vector2OrdinalOffset,
          int clusterCount
  ) {
      final int k          = clusterCount;
      final int blockSize  = k * (k + 1) / 2;
      float res = 0f;

      for (int i = 0; i < subspaceCount; i++) {
          int c1 = Byte.toUnsignedInt(vector1Ordinals.get(i + vector1OrdinalOffset));
          int c2 = Byte.toUnsignedInt(vector2Ordinals.get(i + vector2OrdinalOffset));
          int r  = Math.min(c1, c2);
          int c  = Math.max(c1, c2);

          int offsetRow  = r * k - (r * (r - 1) / 2);
          int idxInBlock = offsetRow + (c - r);
          int base       = i * blockSize;

          res += codebookPartialSums.get(base + idxInBlock);
      }

      return res;
  }

  @Override
  public int hammingDistance(long[] v1, long[] v2) {
    int hd = 0;
    for (int i = 0; i < v1.length; i++) {
      hd += Long.bitCount(v1[i] ^ v2[i]);
    }
    return hd;
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
    var codebookSize = partials.length() / partialBases.length();
    for (int i = 0; i < partialBases.length(); i++) {
      var localBest = partialBases.get(i);
      for (int j = 0; j < codebookSize; j++) {
        var val = partials.get(i * codebookSize + j);
        var quantized = (short) Math.min((val - localBest) / delta, 65535);
        quantizedPartials.setLittleEndianShort(i * codebookSize + j, quantized);
      }
    }
  }

  @Override
  public float max(VectorFloat<?> v) {
    float max = -Float.MAX_VALUE;
    for (int i = 0; i < v.length(); i++) {
      max = Math.max(max, v.get(i));
    }
    return max;
  }

  @Override
  public float min(VectorFloat<?> v) {
    float min = Float.MAX_VALUE;
    for (int i = 0; i < v.length(); i++) {
      min = Math.min(min, v.get(i));
    }
    return min;
  }

  @Override
  public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    var delta = maxValue - minValue;
    var scaledGrowthRate = growthRate / delta;
    var scaledMidpoint = midpoint * delta;
    var inverseScaledGrowthRate = 1 / scaledGrowthRate;
    var logisticBias = logisticFunctionNQT(minValue, scaledGrowthRate, scaledMidpoint);
    var logisticScale = (logisticFunctionNQT(maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / 255;

    float dotProd = 0;
    float value;
    for (int d = 0; d < bytes.length(); d++) {
      value = Byte.toUnsignedInt(bytes.get(d));
      value = scaledLogitFunctionNQT(value, inverseScaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);

      dotProd = Math.fma(vector.get(d), value, dotProd);
    }
    return dotProd;
  }

  @Override
  public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    var delta = maxValue - minValue;
    var scaledGrowthRate = growthRate / delta;
    var scaledMidpoint = midpoint * delta;
    var inverseScaledGrowthRate = 1 / scaledGrowthRate;
    var logisticBias = logisticFunctionNQT(minValue, scaledGrowthRate, scaledMidpoint);
    var logisticScale = (logisticFunctionNQT(maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / 255;

    float squareSum = 0;

    float value;

    for (int d = 0; d < bytes.length(); d++) {
      value = Byte.toUnsignedInt(bytes.get(d));
      value = scaledLogitFunctionNQT(value, inverseScaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);

      var temp = value - vector.get(d);
      squareSum = Math.fma(temp, temp, squareSum);
    }
    return squareSum;
  }

  @Override
  public float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    var delta = maxValue - minValue;
    var scaledGrowthRate = growthRate / delta;
    var scaledMidpoint = midpoint * delta;
    var inverseScaledGrowthRate = 1 / scaledGrowthRate;
    var logisticBias = logisticFunctionNQT(minValue, scaledGrowthRate, scaledMidpoint);
    var logisticScale = (logisticFunctionNQT(maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / 255;

    float sum = 0;
    float normDQ = 0;

    float elem2;

    for (int d = 0; d < bytes.length(); d++) {
      elem2 = Byte.toUnsignedInt(bytes.get(d));
      elem2 = scaledLogitFunctionNQT(elem2, inverseScaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);
      elem2 += centroid.get(d);

      sum = Math.fma(vector.get(d), elem2, sum);
      normDQ = Math.fma(elem2, elem2, normDQ);
    }
    return new float[]{sum, normDQ};
  }

  @Override
  public void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {}

  static float logisticFunctionNQT(float value, float alpha, float x0) {
    float temp = Math.fma(value, alpha, -alpha * x0);
    int p = Math.round(temp + 0.5f);
    int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));

    temp = Float.intBitsToFloat(m + (p << 23));  // temp = m * 2^p
    return temp / (temp + 1);
  }

  static float logitNQT(float value, float inverseAlpha, float x0) {
    float z = value / (1 - value);

    int temp = Float.floatToIntBits(z);
    int e = temp & 0x7f800000;
    float p = (float) ((e >> 23) - 128);
    float m = Float.intBitsToFloat((temp & 0x007fffff) + 0x3f800000);

    return Math.fma(m + p, inverseAlpha, x0);
  }

  static float scaledLogisticFunction(float value, float growthRate, float midpoint, float logisticScale, float logisticBias) {
    var y = logisticFunctionNQT(value, growthRate, midpoint);
    return (y - logisticBias) * (1 / logisticScale);
  }

  static float scaledLogitFunctionNQT(float value, float inverseGrowthRate, float midpoint, float logisticScale, float logisticBias) {
    var scaledValue = Math.fma(value, logisticScale, logisticBias);
    return logitNQT(scaledValue, inverseGrowthRate, midpoint);
  }

  @Override
  public void nvqQuantize8bit(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, ByteSequence<?> destination) {
    var delta = maxValue - minValue;
    var scaledGrowthRate = growthRate / delta;
    var scaledMidpoint = midpoint * delta;
    var logisticBias = logisticFunctionNQT(minValue, scaledGrowthRate, scaledMidpoint);
    var logisticScale = (logisticFunctionNQT(maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / 255;


    for (int d = 0; d < vector.length(); d++) {
      // Ensure the quantized value is within the 0 to constant range
      float value = vector.get(d);
      value = scaledLogisticFunction(value, scaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);
      int quantizedValue = Math.round(value);
      destination.set(d, (byte) quantizedValue);
    }
  }

  public float nvqLoss(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, int nBits) {
    float constant = (1 << nBits) - 1;

    var delta = maxValue - minValue;
    var scaledGrowthRate = growthRate / delta;
    var scaledMidpoint = midpoint * delta;

    var logisticBias = logisticFunctionNQT(minValue, scaledGrowthRate, scaledMidpoint);
    var logisticScale = (logisticFunctionNQT(maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / constant;
    var inverseScaledGrowthRate = 1 / scaledGrowthRate;

    float squaredSum = 0.f;
    float originalValue, reconstructedValue;
    for (int d = 0; d < vector.length(); d++) {
      originalValue = vector.get(d);

      reconstructedValue = scaledLogisticFunction(originalValue, scaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);
      reconstructedValue = Math.round(reconstructedValue);
      reconstructedValue = scaledLogitFunctionNQT(reconstructedValue, inverseScaledGrowthRate, scaledMidpoint, logisticScale, logisticBias);

      var diff = originalValue - reconstructedValue;
      squaredSum = Math.fma(diff, diff, squaredSum);
    }

    return squaredSum;
  }

  public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    float constant = (1 << nBits) - 1;

    float squaredSum = 0.f;
    float originalValue, reconstructedValue;
    for (int d = 0; d < vector.length(); d++) {
      originalValue = vector.get(d);

      reconstructedValue = (originalValue - minValue) / (maxValue - minValue);
      reconstructedValue = Math.round(constant * reconstructedValue) / constant;
      reconstructedValue = reconstructedValue * (maxValue - minValue) + minValue;

      var diff = originalValue - reconstructedValue;
      squaredSum = Math.fma(diff, diff, squaredSum);
    }

    return squaredSum;
  }

}
