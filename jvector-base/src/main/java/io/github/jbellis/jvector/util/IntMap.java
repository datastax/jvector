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
 * A specialized map interface using primitive int keys instead of Integer objects, providing
 * better performance and lower memory overhead for integer-keyed collections.
 * @param <T> the type of values stored in the map
 */
public interface IntMap<T> {
    /**
     * Compare and put.
     * @param key ordinal
     * @param existing the existing
     * @param value the value
     * @return true if successful, false if the current value != `existing`
     */
    boolean compareAndPut(int key, T existing, T value);

    /**
     * Size of the map.
     * @return number of items that have been added
     */
    int size();

    /**
     * Get value for key.
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    T get(int key);

    /**
     * Remove key from map.
     * @param key the key
     * @return the former value of the key, or null if it was not set
     */
    T remove(int key);

    /**
     * Check if key is in map.
     * @param key the key
     * @return true iff the given key is set in the map
     */
    boolean containsKey(int key);

    /**
     * Iterates keys in ascending order and calls the consumer for each non-null key-value pair.
     * @param consumer the consumer
     */
    void forEach(IntBiConsumer<T> consumer);

    /**
     * Functional interface for consuming key-value pairs where keys are primitive ints.
     * @param <T2> the type of values being consumed
     */
    @FunctionalInterface
    interface IntBiConsumer<T2> {
        /**
         * Processes a key-value pair from the map.
         * @param key the primitive int key
         * @param value the value associated with the key
         */
        void consume(int key, T2 value);
    }
}
