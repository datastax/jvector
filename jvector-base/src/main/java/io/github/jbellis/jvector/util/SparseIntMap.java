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

import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

public class SparseIntMap<T> implements IntMap<T> {
    private final ConcurrentHashMap<Integer, T> map;

    public SparseIntMap() {
        this.map = new ConcurrentHashMap<>();
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }

        if (existing == null) {
            T result = map.putIfAbsent(key, value);
            return result == null;
        }

        return map.replace(key, existing, value);
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public T get(int key) {
        return map.get(key);
    }

    @Override
    public T remove(int key) {
        return map.remove(key);
    }

    @Override
    public boolean containsKey(int key) {
        return map.containsKey(key);
    }

    public IntStream keysStream() {
        return map.keySet().stream().mapToInt(key -> key);
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        map.forEach(consumer::consume);
    }
}
