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

import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * The standard {@link ThreadLocal} appears to be designed to be used with relatively
 * short-lived Threads.  Specifically, it uses a ThreadLocalMap to store ThreadLocal key/value
 * Entry objects, and there are no guarantees as to when Entry references are expunged unless
 * you can explicitly call remove() on the ThreadLocal instance.  This means that objects
 * referenced by ThreadLocals will not be able to be GC'd for the lifetime of the Thread,
 * effectively "leaking" these objects even if there are no other references.
 * <p>
 * This makes ThreadLocal a bad fit for long-lived threads, such as those in the thread pools
 * used by JVector.
 * <p>
 * Because ExplicitThreadLocal doesn't hook into Thread internals, any referenced values
 * can be GC'd as expected as soon as the ETL instance itself is no longer referenced.
 * ExplicitThreadLocal also implements AutoCloseable to cleanup non-GC'd resources.
 * <p>
 * ExplicitThreadLocal is a drop-in replacement for ThreadLocal, and is used in the same way.
 */
public abstract class ExplicitThreadLocal<U> implements CloseableSupplier<U> {
    // thread id -> instance
    private final ConcurrentHashMap<Long, U> map = new ConcurrentHashMap<>();

    // computeIfAbsent wants a callable that takes a parameter, but if we use a lambda
    // it will be a closure and we'll get a new instance for every call.  So we instantiate
    // it just once here as a field instead.
    private final Function<Long, U> initialSupplier = k -> initialValue();

    public U get() {
        return map.computeIfAbsent(Thread.currentThread().getId(), initialSupplier);
    }

    protected abstract U initialValue();

    /**
     * Invoke the close() method on all AutoCloseable values in the map, and then clear the map.
     * <p>
     * Not threadsafe.
     */
    @Override
    public void close() {
        Throwable thrown = null;
        try {
            for (U value : map.values()) {
                if (value instanceof AutoCloseable) {
                    try {
                        ((AutoCloseable) value).close();
                    } catch (Throwable t) {
                        if (thrown == null) {
                            thrown = t;
                        } else {
                            thrown.addSuppressed(t);
                        }
                    }
                }
            }
        } finally {
            map.clear();
        }

        if (thrown instanceof RuntimeException) {
            throw (RuntimeException) thrown;
        } else if (thrown instanceof Error) {
            throw (Error) thrown;
        } else if (thrown != null) {
            throw new RuntimeException(thrown);
        }
    }

    public static <U> ExplicitThreadLocal<U> withInitial(Supplier<U> initialValue) {
        return new ExplicitThreadLocal<>() {
            @Override
            protected U initialValue() {
                return initialValue.get();
            }
        };
    }
}

