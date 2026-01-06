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

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public interface VectorRepresentation extends Accountable {
    /** Return the dimension of the returned vector values */
    int dimension();

    /** @return the original size of each vector, in bytes, before compression */
    default int getOriginalSize() {
        return dimension() * Float.BYTES;
    }

    /** @return the compressed size of each vector, in bytes */
    int getCompressedSize();

    /**
     * @return true if the representation is exact, i.e., a full-precision (float32, FP32) vector
     */
    boolean isExact();

    /**
     * @return a copy of the vector representation
     */
    VectorRepresentation copy();

    /**
     * @return this if the representation is exact, else a decoded version of the representation
     */
    VectorFloat<?> decode();

    interface Exact extends VectorRepresentation {
        default boolean isExact() {
            return true;
        }
    }

    interface Approximate extends VectorRepresentation {
        default boolean isExact() {
            return false;
        }
    }
}
