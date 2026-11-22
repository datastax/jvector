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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;

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
@Experimental
public class InlineByteVectors extends AbstractFeature {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimension;
    // The largest length of the vectors this feature has written, for the header.
    private final MaxVectorNorm written = new MaxVectorNorm();
    // The largest length stored in a loaded index's header; NaN when unknown.
    private final float stored;

    public InlineByteVectors(int dimension) {
        this(dimension, Float.NaN);
    }

    private InlineByteVectors(int dimension, float stored) {
        this.dimension = dimension;
        this.stored = stored;
    }

    /**
     * The largest length of the stored vectors: from the header of a loaded index, or of the vectors written so far.
     * NaN when unknown, as for a writer that wrote its header first.
     *
     * <p>The length is {@code sqrt(sum of squared components)} of the int8 values exactly as stored, with no scaling;
     * if int8 vectors are later normalized before storage, it is the length of the normalized vectors. It is a
     * length, not a score. Nothing reads it in search yet: it is reserved so int8 adaptive stopping can be added
     * without another format version. That stopping must turn it into a score bound through
     * {@link io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction}'s own mapping (for {@code DOT_PRODUCT},
     * {@code dot + 1}, or {@code 1 / (1 - dot)} when negative), not the float rule {@code (1 + dot) / 2}.
     */
    public float maxVectorNorm() {
        return Float.isNaN(stored) ? written.get() : stored;
    }

    @Override
    public FeatureId id() {
        return FeatureId.INLINE_BYTE_VECTORS;
    }

    @Override
    public int headerSize() {
        return headerSize(OnDiskGraphIndex.CURRENT_VERSION);
    }

    /** The common header holds the dimension; the feature header stores the largest vector length. */
    @Override
    public int headerSize(int version) {
        return MaxVectorNorm.headerSize(version);
    }

    /** One byte per component. */
    @Override
    public int featureSize() {
        return dimension;
    }

    public int dimension() {
        return dimension;
    }

    /** This feature and its record of the largest vector length; the vectors stay on disk. */
    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.shallowSizeOf(this) + written.ramBytesUsed();
    }

    static InlineByteVectors load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new InlineByteVectors(header.dimension, MaxVectorNorm.read(reader, header.version));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Deprecated
    @Override
    public void writeHeader(IndexWriter out) throws IOException {
        writeHeader(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeHeader(IndexWriter out, int version) throws IOException {
        written.write(out, version);
    }

    @Override
    public void writeInline(IndexWriter out, Feature.State state) throws IOException {
        var vector = ((State) state).vector;
        written.record(vector);
        vts.writeByteSequence(out, vector);
    }

    public static class State implements Feature.State {
        public final ByteSequence<?> vector;

        public State(ByteSequence<?> vector) {
            this.vector = vector;
        }
    }
}
