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

import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Vector similarity function; used in search to return top K most similar vectors to a target
 * vector. This is a label describing the method used during indexing and searching of the vectors
 * in order to determine the nearest neighbors.
 */
public enum VectorSimilarityFunction {

  /** Euclidean distance */
  EUCLIDEAN {
    @Override
    public float compare(VectorFloat<?> v1, VectorFloat<?> v2) {
      return 1 / (1 + VectorUtil.squareL2Distance(v1, v2));
    }

    @Override
    public void compare(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
      VectorUtil.squareL2Distance(q, targets, count, results);
      for (int i = 0; i < count; i++) {
        results[i] = 1 / (1 + results[i]);
      }
    }
  },

  /**
   * Dot product. NOTE: this similarity is intended as an optimized way to perform cosine
   * similarity. In order to use it, all vectors must be normalized, including both document and
   * query vectors. Using dot product with vectors that are not normalized can result in errors or
   * poor search results. Floating point vectors must be normalized to be of unit length, while byte
   * vectors should simply all have the same norm.
   */
  DOT_PRODUCT {
    @Override
    public float compare(VectorFloat<?> v1, VectorFloat<?> v2) {
      return (1 + VectorUtil.dotProduct(v1, v2)) / 2;
    }

    @Override
    public void compare(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
      VectorUtil.dotProduct(q, targets, count, results);
      for (int i = 0; i < count; i++) {
        results[i] = (1 + results[i]) / 2;
      }
    }
  },

  /**
   * Cosine similarity. NOTE: the preferred way to perform cosine similarity is to normalize all
   * vectors to unit length, and instead use {@link VectorSimilarityFunction#DOT_PRODUCT}. You
   * should only use this function if you need to preserve the original vectors and cannot normalize
   * them in advance. The similarity score is normalised to assure it is positive.
   */
  COSINE {
    @Override
    public float compare(VectorFloat<?> v1, VectorFloat<?> v2) {
      return (1 + VectorUtil.cosine(v1, v2)) / 2;
    }

    @Override
    public void compare(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
      VectorUtil.cosine(q, targets, count, results);
      for (int i = 0; i < count; i++) {
        results[i] = (1 + results[i]) / 2;
      }
    }
  };

  /**
   * Calculates a similarity score between the two vectors with a specified function. Higher
   * similarity scores correspond to closer vectors.
   *
   * @param v1 a vector
   * @param v2 another vector, of the same dimension
   * @return the value of the similarity function applied to the two vectors
   */
  public abstract float compare(VectorFloat<?> v1, VectorFloat<?> v2);

  /**
   * Computes {@code compare(q, targets[i])} into {@code results[i]} for each {@code i < count}. Each result equals
   * the one {@link #compare(VectorFloat, VectorFloat)} returns for the same pair; scoring the targets together lets
   * the vector provider overlap their memory loads.
   *
   * @param q a vector
   * @param targets the vectors to compare with {@code q}, each of its dimension
   * @param count how many of {@code targets} to compare
   * @param results receives the scores, at least {@code count} long
   */
  public abstract void compare(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results);
}
