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
 * A generic interface for float vector storage and manipulation, parameterized by backing storage type T.
 * @param <T> the type of the backing storage (e.g., float[] or ByteBuffer)
 */
public interface VectorFloat<T> extends Accountable
{
    /**
     * Gets the entire vector backing storage.
     * @return entire vector backing storage
     */
    T get();

    /**
     * Gets the length of the vector.
     * @return the length of the vector
     */
    int length();

    /**
     * Computes the physical offset in the backing storage for the logical vector index.
     * @param i the logical index in the vector
     * @return the physical offset in the backing storage
     */
    default int offset(int i) {
        return i;
    }

    /**
     * Creates a copy of this vector.
     * @return a copy of this vector
     */
    VectorFloat<T> copy();

    /**
     * Copy from another vector.
     * @param src the source vector
     * @param srcOffset the source offset
     * @param destOffset the destination offset
     * @param length the length to copy
     */
    void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length);

    /**
     * Get the value at the specified index.
     * @param i the index
     * @return the value at index i
     */
    float get(int i);

    /**
     * Set the value at the specified index.
     * @param i the index
     * @param value the value to set
     */
    void set(int i, float value);

    /** Set all values to zero. */
    void zero();

    /**
     * Computes a hash code for this vector based on its non-zero elements.
     * @return the hash code for this vector
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
