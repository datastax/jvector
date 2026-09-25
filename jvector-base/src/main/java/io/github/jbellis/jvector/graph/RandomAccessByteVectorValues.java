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

import io.github.jbellis.jvector.vector.types.ByteSequence;

/**
 * Provides random access to byte (int8) vectors by dense ordinal.
 * <p>
 * This is the byte-vector parallel to {@link RandomAccessVectorValues}.
 * Both extend the common super-interface {@link VectorValues}.
 * It is used by graph-based index builders and searchers that operate natively
 * on int8 vectors without a float32 round-trip.
 */
public interface RandomAccessByteVectorValues extends VectorValues<ByteSequence<?>> {

    /**
     * Creates a new copy of this {@link RandomAccessByteVectorValues}.
     * Un-shared implementations may simply return {@code this}.
     */
    @Override
    RandomAccessByteVectorValues copy();
}
