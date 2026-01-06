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

import io.github.jbellis.jvector.vector.VectorRepresentation;

public interface VectorFloat<T> extends VectorRepresentation {
    /**
     * @return entire vector backing storage
     */
    T get();

    int dimension();

    default int getCompressedSize() {
        return dimension() * Float.BYTES;
    }

    default int offset(int i) {
        return i;
    }

    float get(int i);

    void set(int i, float value);

    void zero();

    default int getHashCode() {
        int result = 1;
        for (int i = 0; i < dimension(); i++) {
            if (get(i) != 0) {
                result = 31 * result + Float.hashCode(get(i));
            }
        }
        return result;
    }
}
