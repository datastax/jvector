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
import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * Provides random access to byte (int8) vectors by dense ordinal.
 * <p>
 * This is the byte-vector parallel to {@link RandomAccessVectorValues}.
 * It is used by graph-based index builders and searchers that operate natively
 * on int8 vectors without a float32 round-trip.
 */
public interface RandomAccessByteVectorValues {
    Logger LOG = Logger.getLogger(RandomAccessByteVectorValues.class.getName());

    /** Return the number of vector values. */
    int size();

    /** Return the dimension of the returned vector values. */
    int dimension();

    /**
     * Return the byte vector indexed at the given ordinal.
     *
     * @param nodeId a valid ordinal, &ge; 0 and &lt; {@link #size()}.
     */
    ByteSequence<?> getVector(int nodeId);

    /**
     * @return true iff the vector returned by {@link #getVector} is shared across calls.
     * A shared vector is only valid until the next call to {@link #getVector} overwrites it.
     */
    boolean isValueShared();

    /**
     * Creates a new copy of this {@link RandomAccessByteVectorValues}.
     * Un-shared implementations may simply return {@code this}.
     */
    RandomAccessByteVectorValues copy();

    /**
     * Returns a supplier of thread-local copies of the RABVV.
     */
    default Supplier<RandomAccessByteVectorValues> threadLocalSupplier() {
        if (!isValueShared()) {
            return () -> this;
        }

        if (this instanceof AutoCloseable) {
            LOG.warning("RABVV is shared and implements AutoCloseable; threadLocalSupplier() may lead to leaks");
        }
        var tl = ExplicitThreadLocal.withInitial(this::copy);
        return tl::get;
    }
}
