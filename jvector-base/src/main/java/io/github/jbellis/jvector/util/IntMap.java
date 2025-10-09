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

/**
 * A map with integer keys that provides atomic compare-and-put operations.
 *
 * @param <T> the type of values stored in the map
 */
public interface IntMap<T> {
    /**
     * Atomically sets the value for the given key if the current value matches the expected existing value.
     *
     * @param key ordinal
     * @param existing the expected current value (may be null)
     * @param value the new value to set
     * @return true if successful, false if the current value != {@code existing}
     */
    boolean compareAndPut(int key, T existing, T value);

    /**
     * Returns the number of items that have been added to this map.
     *
     * @return number of items that have been added
     */
    int size();

    /**
     * Returns the value associated with the given key.
     *
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    T get(int key);

    /**
     * Removes the mapping for the given key from this map if present.
     *
     * @param key the key to remove
     * @return the former value of the key, or null if it was not set
     */
    T remove(int key);

    /**
     * Checks if this map contains a mapping for the given key.
     *
     * @param key the key to check
     * @return true iff the given key is set in the map
     */
    boolean containsKey(int key);

    /**
     * Iterates keys in ascending order and calls the consumer for each non-null key-value pair.
     *
     * @param consumer the consumer to call for each key-value pair
     */
    void forEach(IntBiConsumer<T> consumer);

    /**
     * A functional interface for consuming key-value pairs where the key is an int.
     *
     * @param <T2> the type of the value
     */
    @FunctionalInterface
    interface IntBiConsumer<T2> {
        /**
         * Consumes a key-value pair.
         *
         * @param key the integer key
         * @param value the value associated with the key
         */
        void consume(int key, T2 value);
    }
}
