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

package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.NodesIterator;

import java.util.stream.IntStream;

public interface IntMap<T> {
    /**
     * @param key ordinal
     * @return true if successful, false if the current value != `existing`
     */
    boolean compareAndPut(int key, T existing, T value);

    /**
     * @return number of items that have been added
     */
    int size();

    /**
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    T get(int key);

    /**
     * @return the former value of the key, or null if it was not set
     */
    T remove(int key);

    /**
     * @return true iff the given key is set in the map
     */
    boolean containsKey(int key);

    /**
     * Iterates keys in ascending order and calls the consumer for each non-null key-value pair.
     */
    void forEach(IntBiConsumer<T> consumer);

    @FunctionalInterface
    interface IntBiConsumer<T2> {
        void consume(int key, T2 value);
    }
}
