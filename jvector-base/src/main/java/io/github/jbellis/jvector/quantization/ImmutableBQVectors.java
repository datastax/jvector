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
 * This class provides read-only access to a fixed set of compressed vectors.
 */
public class ImmutableBQVectors extends BQVectors {
    /**
     * Creates a new ImmutableBQVectors instance with the given quantization and compressed vectors.
     *
     * @param bq the binary quantization configuration
     * @param compressedVectors the array of compressed vector data
     */
    public ImmutableBQVectors(BinaryQuantization bq, long[][] compressedVectors) {
        super(bq);
        this.compressedVectors = compressedVectors;
    }

    /**
     * Returns the number of vectors in this collection.
     *
     * @return the count of compressed vectors
     */
    @Override
    public int count() {
        return compressedVectors.length;
    }
}
