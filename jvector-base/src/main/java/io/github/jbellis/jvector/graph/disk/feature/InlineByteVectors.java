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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;

/**
 * Stores signed int8 (byte) vectors inline in an {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex}.
 * <p>
 * Each vector occupies exactly {@code dimension} bytes on disk — 4× smaller than
 * the float32 {@link InlineVectors} representation.  The on-disk layout is otherwise
 * identical: one contiguous byte block per node record, written and read via
 * {@link VectorTypeSupport#writeByteSequence} and {@link VectorTypeSupport#readByteSequence}.
 * <p>
 * Use {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex.View#getByteVector(int)}
 * to retrieve a stored vector at search time.
 */
public class InlineByteVectors extends AbstractFeature {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimension;

    public InlineByteVectors(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public FeatureId id() {
        return FeatureId.INLINE_BYTE_VECTORS;
    }

    /** No extra header bytes — dimension is already in the {@link CommonHeader}. */
    @Override
    public int headerSize() {
        return 0;
    }

    /** One byte per component. */
    @Override
    public int featureSize() {
        return dimension;
    }

    public int dimension() {
        return dimension;
    }

    static InlineByteVectors load(CommonHeader header, RandomAccessReader reader) {
        return new InlineByteVectors(header.dimension);
    }

    @Override
    public void writeHeader(IndexWriter out) {
        // common header carries dimension; nothing extra needed
    }

    @Override
    public void writeInline(IndexWriter out, Feature.State state) throws IOException {
        vts.writeByteSequence(out, ((State) state).vector);
    }

    public static class State implements Feature.State {
        public final ByteSequence<?> vector;

        public State(ByteSequence<?> vector) {
            this.vector = vector;
        }
    }
}
