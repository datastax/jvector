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

/**
 * Vector similarity function for signed int8 (byte) vectors; parallel to
 * {@link VectorSimilarityFunction} but operating on {@link ByteSequence}.
 * <p>
 * Bytes are treated as signed int8 values (Java's {@code byte} is already signed, range −128..127).
 * Return-value conventions match {@link VectorSimilarityFunction}: higher is more similar.
 */
public enum ByteVectorSimilarityFunction {

    /**
     * Euclidean similarity normalised to {@code (0, 1]}.
     * Raw squared L2 is divided by {@code n * 255^2} (the maximum possible squared distance
     * between two signed int8 vectors) before the {@code 1 / (1 + x)} mapping, so the result
     * is always in (0, 1] regardless of dimension.
     */
    EUCLIDEAN {
        @Override
        public float compare(ByteSequence<?> v1, ByteSequence<?> v2) {
            float maxSquaredDist = v1.length() * (255.0f * 255.0f);
            return 1.0f / (1.0f + VectorUtil.squareL2Distance(v1, v2) / maxSquaredDist);
        }
    },

    /**
     * Dot product normalised to {@code [0, 1]}.
     * Raw int8 dot product is divided by {@code n * 127^2} (the maximum possible magnitude)
     * before applying the {@code (1 + x) / 2} mapping, so the result is always in [0, 1]
     * regardless of dimension or whether the vectors are unit-norm.
     * For already unit-norm int8 vectors (e.g. Cohere, OpenAI reduced-precision) prefer {@link #COSINE}.
     */
    DOT_PRODUCT {
        @Override
        public float compare(ByteSequence<?> v1, ByteSequence<?> v2) {
            float maxMagnitude = v1.length() * (127.0f * 127.0f);
            return (1.0f + VectorUtil.dotProduct(v1, v2) / maxMagnitude) / 2.0f;
        }
    },

    /** Cosine similarity: {@code (1 + cosine(v1, v2)) / 2} */
    COSINE {
        @Override
        public float compare(ByteSequence<?> v1, ByteSequence<?> v2) {
            return (1.0f + VectorUtil.cosine(v1, v2)) / 2.0f;
        }
    };

    /**
     * Calculates a similarity score between the two int8 vectors.
     * Higher values correspond to closer vectors.
     *
     * @param v1 a byte vector
     * @param v2 another byte vector, of the same dimension
     * @return the similarity score
     */
    public abstract float compare(ByteSequence<?> v1, ByteSequence<?> v2);
}
