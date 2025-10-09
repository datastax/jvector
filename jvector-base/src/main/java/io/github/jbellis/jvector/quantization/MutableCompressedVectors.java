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
 * Interface for mutable compressed vector storage that allows adding and modifying vectors.
 *
 * @param <T> the type of vectors to compress
 */
public interface MutableCompressedVectors<T> extends CompressedVectors {
    /**
     * Encode the given vector and set it at the given ordinal. Done without unnecessary copying.
     *
     * It's the caller's responsibility to ensure there are no "holes" in the ordinals that are
     * neither encoded nor set to zero.
     *
     * @param ordinal the ordinal to set
     * @param vector the vector to encode and set
     */
    void encodeAndSet(int ordinal, T vector);

    /**
     * Set the vector at the given ordinal to zero.
     *
     * It's the caller's responsibility to ensure there are no "holes" in the ordinals that are
     * neither encoded nor set to zero.
     *
     * @param ordinal the ordinal to set
     */
    void setZero(int ordinal);
}
