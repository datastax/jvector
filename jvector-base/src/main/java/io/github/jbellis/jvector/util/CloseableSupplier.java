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

import java.util.function.Supplier;

/**
 * A {@link Supplier} that owns resources associated with values it creates.
 */
public interface CloseableSupplier<T> extends Supplier<T>, AutoCloseable {
    @Override
    void close();

    static <T> CloseableSupplier<T> noOp(Supplier<T> supplier) {
        return new CloseableSupplier<>() {
            @Override
            public T get() {
                return supplier.get();
            }

            @Override
            public void close() {
            }
        };
    }
}
