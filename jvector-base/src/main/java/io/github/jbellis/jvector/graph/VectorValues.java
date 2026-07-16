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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.ExplicitThreadLocal;

import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * Common super-interface for random access to vectors by dense ordinal.
 * <p>
 * {@code V} is the vector element type — {@code VectorFloat<?>} for float32 vectors
 * (see {@link RandomAccessVectorValues}) and {@code ByteSequence<?>} for int8 vectors
 * (see {@link RandomAccessByteVectorValues}).
 */
public interface VectorValues<V> {
    Logger LOG = Logger.getLogger(VectorValues.class.getName());

    /** Return the number of vector values. */
    int size();

    /** Return the dimension of the returned vector values. */
    int dimension();

    /**
     * Return the vector value indexed at the given ordinal.
     * <p>
     * For performance, implementations are free to re-use the same object across invocations.
     * If you need to retain the value across calls, make a copy.
     *
     * @param nodeId a valid ordinal, &ge; 0 and &lt; {@link #size()}.
     */
    V getVector(int nodeId);

    /**
     * @return true iff the vector returned by {@link #getVector} is shared across calls.
     * A shared vector is only valid until the next call to {@link #getVector} overwrites it.
     */
    boolean isValueShared();

    /**
     * Creates a new copy of this instance.
     * Un-shared implementations may simply return {@code this}.
     */
    VectorValues<V> copy();

    /**
     * Returns a supplier of thread-local copies of this instance.
     */
    @SuppressWarnings("unchecked")
    default Supplier<VectorValues<V>> threadLocalSupplier() {
        if (!isValueShared()) {
            return () -> this;
        }

        if (this instanceof AutoCloseable) {
            LOG.warning("VectorValues is shared and implements AutoCloseable; threadLocalSupplier() may lead to leaks");
        }
        var tl = ExplicitThreadLocal.withInitial(this::copy);
        return tl::get;
    }
}
