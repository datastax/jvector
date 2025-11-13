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
import java.util.Objects;

public interface ByteSequence<T> extends Accountable
{
    /**
     * @return entire sequence backing storage
     */
    T get();

    int offset();

    int length();

    byte get(int i);

    void set(int i, byte value);

    void zero();

    void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int length);

    ByteSequence<T> copy();

    ByteSequence<T>  slice(int offset, int length);

    /**
     * Two ByteSequences are equal if they have the same length and the same bytes at each position.
     * @param o the other object to compare to
     * @return true if the two ByteSequences are equal
     */
    default boolean equalTo(Object o) {
        if (this == o) return true;
        if (!(o instanceof ByteSequence)) return false;
        ByteSequence<?> that = (ByteSequence<?>) o;
        if (length() != that.length()) return false;
        for (int i = 0; i < length(); i++) {
            if (get(i) != that.get(i)) return false;
        }
        return true;
    }

    /**
     * @return a hash code for this ByteSequence
     */
    default int getHashCode() {
        int result = 1;
        for (int i = 0; i < length(); i++) {
            if (get(i) != 0) {
                result = 31 * result + get(i);
            }
        }
        return result;
    }
}
