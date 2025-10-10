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

/**
 * An immutable collection of binary quantized vectors.
 * Wraps pre-computed compressed vectors for efficient similarity comparisons.
 */
public class ImmutableBQVectors extends BQVectors {
    /**
     * Constructs an ImmutableBQVectors instance with pre-compressed vectors.
     *
     * @param bq the binary quantization scheme used to compress the vectors
     * @param compressedVectors array of compressed vector representations (one per vector)
     */
    public ImmutableBQVectors(BinaryQuantization bq, long[][] compressedVectors) {
        super(bq);
        this.compressedVectors = compressedVectors;
    }

    @Override
    public int count() {
        return compressedVectors.length;
    }
}
