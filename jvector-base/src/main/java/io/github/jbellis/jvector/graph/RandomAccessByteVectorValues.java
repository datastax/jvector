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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.util.function.Supplier;

/**
 * Provides random access to byte (int8) vectors by dense ordinal.
 * <p>
 * This is the byte-vector parallel to {@link RandomAccessVectorValues}.
 * Both extend the common super-interface {@link VectorValues}.
 * It is used by graph-based index builders and searchers that operate natively
 * on int8 vectors without a float32 round-trip.
 */
@Experimental
public interface RandomAccessByteVectorValues extends VectorValues<ByteSequence<?>> {

    /**
     * Retrieve the vector associated with a given node, and store it in the destination vector at the given offset.
     * The default implementation copies from {@link #getVector}; implementations that can read directly into the
     * destination (e.g. from disk) should override it.
     */
    @Override
    default void getVectorInto(int node, ByteSequence<?> destinationVector, int offset) {
        destinationVector.copyFrom(getVector(node), 0, offset, dimension());
    }

    /**
     * Creates a new copy of this {@link RandomAccessByteVectorValues}.
     * Un-shared implementations may simply return {@code this}.
     */
    @Override
    RandomAccessByteVectorValues copy();

    /**
     * Returns a supplier of thread-local copies of the RAVV.
     * <p>
     * For shared instances the returned supplier is {@link AutoCloseable}: closing it invokes
     * close() on every AutoCloseable copy created so far and drops the per-thread cache.
     * Callers that hold the supplier across a bounded operation should close it when done;
     * heap-only copies are additionally collected once the supplier itself becomes
     * unreachable.
     */
    @Override
    default Supplier<RandomAccessByteVectorValues> threadLocalSupplier() {
        if (!isValueShared()) {
            return () -> this;
        }

        if (this instanceof AutoCloseable) {
            VectorValues.LOG.warning("VectorValues is shared and implements AutoCloseable; close the supplier returned by threadLocalSupplier() to release per-thread copies");
        }
        var tl = ExplicitThreadLocal.withInitial(this::copy);
        return new ThreadLocalCopies<>(tl);
    }
}
