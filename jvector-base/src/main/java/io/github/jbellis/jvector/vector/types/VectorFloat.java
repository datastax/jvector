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

package io.github.jbellis.jvector.vector.types;

import io.github.jbellis.jvector.util.Accountable;

/**
 * Represents a vector of float values with a generic backing storage type.
 * <p>
 * This interface provides abstraction over different vector storage implementations,
 * allowing for optimized memory layouts and access patterns. The type parameter {@code T}
 * represents the underlying storage mechanism (e.g., float array, ByteBuffer, etc.).
 * @param <T> the type of the backing storage
 */
public interface VectorFloat<T> extends Accountable
{
    /**
     * Returns the entire vector backing storage.
     * @return the backing storage
     */
    T get();

    /**
     * Returns the length of the vector.
     * @return the number of elements in the vector
     */
    int length();

    /**
     * Returns the offset for the element at the specified index in the backing storage.
     * The default implementation returns the index itself.
     * @param i the logical index
     * @return the offset in the backing storage
     */
    default int offset(int i) {
        return i;
    }

    /**
     * Creates a copy of this vector.
     * @return a new VectorFloat instance with the same values
     */
    VectorFloat<T> copy();

    /**
     * Copies elements from another vector into this vector.
     * @param src the source vector to copy from
     * @param srcOffset the starting offset in the source vector
     * @param destOffset the starting offset in this vector
     * @param length the number of elements to copy
     */
    void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length);

    /**
     * Returns the float value at the specified index.
     * @param i the index
     * @return the float value at the index
     */
    float get(int i);

    /**
     * Sets the float value at the specified index.
     * @param i the index
     * @param value the value to set
     */
    void set(int i, float value);

    /**
     * Sets all elements in the vector to zero.
     */
    void zero();

    /**
     * Computes a hash code for this vector based on its non-zero elements.
     * @return the hash code
     */
    default int getHashCode() {
        int result = 1;
        for (int i = 0; i < length(); i++) {
            if (get(i) != 0) {
                result = 31 * result + Float.hashCode(get(i));
            }
        }
        return result;
    }
}
